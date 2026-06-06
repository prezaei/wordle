"""Autonomous overnight pipeline — CLEAN (leak-free) end-to-end push (driver, not committed).

Every stage applies the 2026-06-05 audit fixes so results are HONEST:
  - candidates from TRAIN only (consistent_candidates over train, not the full answer pool)  [FM-2]
  - teacher pools TRAIN only (generate_transcripts valid_pool/answer_pool=train)              [FM-3]
  - held-out-free openers (drop 'trace' etc.)                                                  [opener]
  - select best-checkpoint on VAL=held[:96]; REPORT on the disjoint TEST=held[96:]             [FM-1]

Pipeline (sequential, single GPU): SFT base -> DAgger -> DPO -> DPO∘DAgger -> GRPO (+ a 2nd
DAgger iter if time). Each stage: best-by-VAL checkpoint, eval on VAL/TEST/full, logged
incrementally to runs/overnight_results.json. Robust: per-stage try/except + a wall-clock budget.
At the end: pick the best setup by VAL, multi-seed TEST eval of the winner (beat MPS ±2-3pt noise),
and write OVERNIGHT_ANALYSIS.md. Runs unattended; never stops on a single-stage failure.
"""

from __future__ import annotations

import copy
import json
import statistics
import time
import traceback
from random import Random

import torch
import torch.nn.functional as F

from wordle_slm.baselines.policies import InfoMaxGuesser
from wordle_slm.config import ModelConfig, RewardConfig, SFTConfig
from wordle_slm.data import is_valid, load_answers, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import consistent_candidates
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.rl.reward import compute_reward
from wordle_slm.sft import pretrain_lm, pretrain_words
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
K_CANDS, AUX_LAMBDA = 3, 0.5
ANSWERS = load_answers()

train, held = split(seed=0)
TRAINP = tuple(train)
CAND_POOL = TRAINP  # FM-2: candidates from train only
TEACHER = InfoMaxGuesser(pool=TRAINP)  # FM-3: DAgger labels over train only
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
SAFE_OPENERS = tuple(o for o in OPENERS if o not in set(held))  # drop held-out openers
VAL = tuple(held[:96])  # FM-1: select on VAL ...
TEST = tuple(held[96:])  # ... report on disjoint TEST
FULL = tuple(held)
START, BUDGET_S = time.time(), 6.7 * 3600
RESULTS: list[dict] = []


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def budget_left():
    return BUDGET_S - (time.time() - START)


# ---- shared helpers (all clean) -------------------------------------------------------------
def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(t):
    return [tok.gray_id] * 5 if t.feedback is None else [tok.token_to_id(_COLOR[c]) for c in t.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for t in turns:
        ids += [tok.guess_id, *_letters(t.guess), *_fb(t), tok.sep_id]
    return ids


def pick_cands(history, guess, rng):
    cons = [w for w in consistent_candidates(history, CAND_POOL) if w != guess]
    rng.shuffle(cons)
    cands = [guess] + cons[: K_CANDS - 1]
    rng.shuffle(cands)
    return cands


def game_examples(game, rng):
    """Per-turn ephemeral examples: board-only history (no loss) + think+guess (loss)."""
    exs = []
    for k, turn in enumerate(game.turns):
        ids = board_only(game.turns[:k])
        mask = [False] * len(ids)
        for c in pick_cands(game.turns[:k], turn.guess, rng):
            ids.append(THINK)
            mask.append(True)
            ids += _letters(c)
            mask += [True] * 5
        ids.append(tok.guess_id)
        mask.append(True)
        ids += _letters(turn.guess)
        mask += [True] * 5
        ids.append(tok.eos_id)
        mask.append(False)
        exs.append((ids, mask))
    return exs


def cot_valid_mask(seqs):
    trie = _valid_trie()
    L = max(len(s) for s in seqs)
    vmask = torch.zeros((len(seqs), L, 26))
    for i, seq in enumerate(seqs):
        for t, tokid in enumerate(seq):
            if tokid not in WORD_STARTS or t + 5 >= len(seq):
                continue
            node = trie
            for j in range(5):
                for child in node:
                    vmask[i, t + j, child] = 1.0
                node = node.get(seq[t + 1 + j] - LETTER_LO, {})
    return vmask


@torch.no_grad()
def batched_generate(model, prompts, temp, max_new=44):
    B = len(prompts)
    lens0 = [len(p) for p in prompts]
    maxL = max(lens0) + max_new
    ids = torch.full((B, maxL), tok.pad_id, dtype=torch.long, device=DEV)
    for i, p in enumerate(prompts):
        ids[i, : lens0[i]] = torch.tensor(p, device=DEV)
    cur = torch.tensor(lens0, device=DEV)
    committed, collected, gen, done = [False] * B, [[] for _ in range(B)], [[] for _ in range(B)], [False] * B
    ar = torch.arange(B, device=DEV)
    for _ in range(max_new):
        if all(done):
            break
        logits = model.forward(ids)[ar, cur - 1]
        probs = torch.softmax(logits[:, ALLOWED_GEN] / temp, dim=-1).cpu()
        choice = torch.multinomial(probs, 1).squeeze(1)
        for i in range(B):
            if done[i]:
                continue
            t = int(ALLOWED_GEN[int(choice[i])])
            ids[i, int(cur[i])] = t
            cur[i] = int(cur[i]) + 1
            gen[i].append(t)
            if committed[i]:
                if t in LETTER_SET:
                    collected[i].append(t)
                if len(collected[i]) >= 5:
                    done[i] = True
            elif t == tok.guess_id:
                committed[i] = True
            if int(cur[i]) >= maxL:
                done[i] = True
    return gen, collected


@torch.no_grad()
def rollout_batch(model, secrets, temp, rows=6):
    games = [Game(s, max_guesses=rows) for s in secrets]
    recs: list[list] = [[] for _ in secrets]
    for _ in range(rows):
        active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
        if not active:
            break
        prompts = [board_only(games[i].turns) for i in active]
        gen, collected = batched_generate(model, prompts, temp)
        for j, i in enumerate(active):
            word = "".join(tok.id_to_token(t) for t in collected[j][:5])
            games[i].guess(word if len(word) == 5 else "zzzzz")
            recs[i].append((prompts[j] + gen[j], [False] * len(prompts[j]) + [True] * len(gen[j])))
    return games, recs


@torch.no_grad()
def play(model, secret, rows=6):
    g = Game(secret, max_guesses=rows)
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)
        guess, committed = [], False
        for _ in range(48):
            nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
            seq.append(nxt)
            if committed:
                if nxt in LETTER_SET:
                    guess.append(nxt)
                if len(guess) >= 5:
                    break
            elif nxt == tok.guess_id:
                committed = True
        word = "".join(tok.id_to_token(t) for t in guess[:5])
        g.guess(word if len(word) == 5 else "zzzzz")
    return g


def win_rate(model, secrets):
    model.eval()
    games = [play(model, s) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def record(name, model, note=""):
    val = win_rate(model, VAL)
    test = win_rate(model, TEST)
    full = win_rate(model, FULL)
    r = {"name": name, "val": round(val["win"], 4), "test": round(test["win"], 4),
         "full": round(full["win"], 4), "valid": round(test["valid"], 3),
         "avg": round(test["avg"], 2) if test["avg"] == test["avg"] else None,
         "note": note, "t": round((time.time() - START) / 60, 1)}
    RESULTS.append(r)
    with open("runs/overnight_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    log(f"RESULT {name}: VAL={r['val']} TEST={r['test']} full={r['full']} valid={r['valid']} avg={r['avg']}  {note}")
    return r


# ---- trainers --------------------------------------------------------------------------------
def sft_aux(model, exs, epochs, lr, ckpt, label, cosine=True):
    """imit + gated aux-validity; best-by-VAL checkpoint; returns the best model loaded."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 10) if cosine else None
    rng = Random(0)
    best = -1.0
    save_checkpoint(ckpt, model, opt, 0, SFTConfig())
    for ep in range(epochs):
        model.train()
        for idx in _batches(len(exs), 128, rng):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tm = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tm[i, : len(m)] = torch.tensor([float(x) for x in m])
            vmask = cot_valid_mask([s for s, _ in bs]).to(DEV)
            ids, tm = ids.to(DEV), tm.to(DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            imit = (nll * tm[:, 1:]).sum() / tm[:, 1:].sum()
            lp = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
            vm = vmask[:, :-1]
            ap = (vm.sum(-1) > 0).float() * tm[:, 1:]
            aux = (-(lp.exp() * vm).sum(-1).clamp_min(1e-9).log() * ap).sum() / ap.sum().clamp_min(1.0)
            loss = imit + AUX_LAMBDA * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched:
            sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            w = win_rate(model, VAL)["win"]
            if w > best:
                best = w
                save_checkpoint(ckpt, model, opt, ep, SFTConfig())
            log(f"  [{label}] ep{ep} loss={float(loss.detach()):.3f} VAL={w:.3f}{'  *' if w == best else ''}")
    load_checkpoint(ckpt, model)
    return model


def seq_logp_full(model, seqs, starts):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_id, dtype=torch.long, device=DEV)
    rm = torch.zeros((len(seqs), L), device=DEV)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=DEV)
        rm[i, starts[i] : len(s)] = 1.0
    logp = torch.log_softmax(model.forward(ids)[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (logp * rm[:, 1:]).sum(-1)


def dpo(base_ckpt, ckpt, label, secrets, n_roll=8, beta=0.1, lr=5e-6, epochs=4):
    policy = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(base_ckpt, policy)
    ref = copy.deepcopy(policy).to(DEV)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.eval()
    rng = Random(1)
    pairs = []
    for c0 in range(0, len(secrets), 8):
        secs = secrets[c0 : c0 + 8]
        games, recs = rollout_batch(policy, [s for s in secs for _ in range(n_roll)], 0.9)
        for si in range(len(secs)):
            idxs = [si * n_roll + k for k in range(n_roll)]
            won = [i for i in idxs if games[i].won]
            lost = [i for i in idxs if not games[i].won]
            made = 0
            for wi in won:
                for li in lost:
                    if made >= 4:
                        break
                    gw = [t.guess for t in games[wi].turns]
                    gl = [t.guess for t in games[li].turns]
                    t = next((k for k in range(min(len(gw), len(gl))) if gw[k] != gl[k]), None)
                    if t is None or recs[wi][t][0] != recs[li][t][0]:
                        continue
                    pw = recs[wi][t][0]
                    pairs.append((pw, recs[li][t][0], len([x for x in recs[wi][t][1] if not x])))
                    made += 1
    log(f"  [{label}] {len(pairs)} DPO pairs")
    if not pairs:
        load_checkpoint(base_ckpt, policy)
        return policy
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    best = win_rate(policy, VAL)["win"]
    save_checkpoint(ckpt, policy, opt, 0, SFTConfig())
    for ep in range(epochs):
        rng.shuffle(pairs)
        policy.eval()
        for b0 in range(0, len(pairs), 16):
            bp = pairs[b0 : b0 + 16]
            ch = [c for c, _, s in bp]
            rj = [r for _, r, s in bp]
            cs = [s for _, _, s in bp]
            lp_c, lp_r = seq_logp_full(policy, ch, cs), seq_logp_full(policy, rj, cs)
            with torch.no_grad():
                rp_c, rp_r = seq_logp_full(ref, ch, cs), seq_logp_full(ref, rj, cs)
            loss = -F.logsigmoid(beta * ((lp_c - rp_c) - (lp_r - rp_r))).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        w = win_rate(policy, VAL)["win"]
        if w > best:
            best = w
            save_checkpoint(ckpt, policy, opt, ep, SFTConfig())
        else:
            load_checkpoint(ckpt, policy)
        log(f"  [{label}] ep{ep} VAL={w:.3f}")
    load_checkpoint(ckpt, policy)
    return policy


def is_bad(turn, history):
    if not turn.valid:
        return True
    if any(turn.guess == pt.guess for pt in history):
        return True
    greens = {}
    for pt in history:
        if pt.valid and pt.feedback:
            for i, c in enumerate(pt.feedback):
                if c is Color.GREEN:
                    greens[i] = pt.guess[i]
    return any(turn.guess[i] != ch for i, ch in greens.items())


def dagger(base_ckpt, teacher_exs, ckpt, label, secrets, epochs=8):
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(base_ckpt, model)
    model.eval()
    rng = Random(0)
    fails = []
    for c0 in range(0, len(secrets), 16):
        secs = secrets[c0 : c0 + 16]
        games = [Game(s, max_guesses=6) for s in secs]
        for _ in range(6):
            active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
            if not active:
                break
            _, cols = batched_generate(model, [board_only(games[i].turns) for i in active], 0.02)
            for j, i in enumerate(active):
                w = "".join(tok.id_to_token(t) for t in cols[j][:5])
                games[i].guess(w if len(w) == 5 else "zzzzz")
        for g in games:
            for k, turn in enumerate(g.turns):
                hist = g.turns[:k]
                if not is_bad(turn, hist) or not hist:
                    continue
                tgt = TEACHER.choose(hist, consistent_candidates(hist, CAND_POOL))
                if is_valid(tgt):
                    dids = board_only(hist)
                    dmask = [False] * len(dids)
                    for c in pick_cands(hist, tgt, rng):
                        dids.append(THINK)
                        dmask.append(True)
                        dids += _letters(c)
                        dmask += [True] * 5
                    dids.append(tok.guess_id)
                    dmask.append(True)
                    dids += _letters(tgt)
                    dmask += [True] * 5
                    dids.append(tok.eos_id)
                    dmask.append(False)
                    fails.append((dids, dmask))
    log(f"  [{label}] {len(fails)} failure boards")
    pool = fails * 4 + (teacher_exs[: len(fails) * 4] if teacher_exs else [])
    rng.shuffle(pool)
    return sft_aux(model, pool, epochs, 6e-5, ckpt, label, cosine=True)


def grpo(base_ckpt, ckpt, label, secrets, updates=30, lr=5e-6, beta_kl=0.05, eps=0.2, B=8, G=8):
    rcfg = RewardConfig()
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(base_ckpt, model)
    ref = copy.deepcopy(model).to(DEV)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best = win_rate(model, VAL)["win"]
    save_checkpoint(ckpt, model, opt, 0, SFTConfig())
    order = list(secrets)
    Random(7).shuffle(order)
    ptr = 0
    for u in range(updates):
        secs = [order[(ptr + i) % len(order)] for i in range(B)]
        ptr += B
        model.eval()
        games, recs = rollout_batch(model, [s for s in secs for _ in range(G)], 1.0)
        rewards = [compute_reward(g, rcfg).total for g in games]
        sa = []
        for si in range(B):
            rs = [rewards[si * G + k] for k in range(G)]
            mean_r = sum(rs) / len(rs)
            if max(rs) - min(rs) < 1e-6:
                continue
            for k in range(G):
                a = rs[k] - mean_r
                for seq, am in recs[si * G + k]:
                    sa.append((seq, am, a))
        if sa:
            model.eval()
            tot = sum(sum(m) for _, m, _ in sa) or 1
            opt.zero_grad()
            for c0 in range(0, len(sa), 64):
                ch = sa[c0 : c0 + 64]
                L = max(len(s) for s, _, _ in ch)
                ids = torch.full((len(ch), L), tok.pad_id, dtype=torch.long)
                am = torch.zeros((len(ch), L))
                adv = torch.zeros((len(ch), 1))
                for i, (s, m, a) in enumerate(ch):
                    ids[i, : len(s)] = torch.tensor(s)
                    am[i, : len(m)] = torch.tensor([float(x) for x in m])
                    adv[i, 0] = a
                ids, am, adv = ids.to(DEV), am.to(DEV), adv.to(DEV)
                logp = torch.log_softmax(model.forward(ids)[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    rlp = torch.log_softmax(ref.forward(ids)[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                m1 = am[:, 1:]
                ratio = torch.exp(logp - logp.detach())
                surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
                d = rlp - logp
                kl = torch.exp(d) - d - 1.0
                ((-(surr * m1).sum() + beta_kl * (kl * m1).sum()) / tot).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if u % 8 == 0 or u == updates - 1:
            w = win_rate(model, VAL)["win"]
            if w > best:
                best = w
                save_checkpoint(ckpt, model, opt, u, SFTConfig())
            log(f"  [{label}] upd{u} VAL={w:.3f}")
    load_checkpoint(ckpt, model)
    return model


def stage(name, fn):
    if budget_left() < 600:
        log(f"SKIP {name} (budget)")
        return None
    try:
        log(f"=== STAGE {name} (budget {budget_left() / 60:.0f}m left) ===")
        return fn()
    except Exception:
        log(f"!!! STAGE {name} FAILED:\n{traceback.format_exc()}")
        return None


# ---- orchestrate -----------------------------------------------------------------------------
log(f"openers={SAFE_OPENERS} |VAL|={len(VAL)} |TEST|={len(TEST)} budget={BUDGET_S/3600:.1f}h")
log("pretrain spell warm-up …")
base0 = WordleGenerator(CFG, VOCAB).to(DEV)
pretrain_lm(base0, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=10, batch_size=256, device=DEV, seed=0)
save_checkpoint("runs/on_pretrain.pt", base0, torch.optim.AdamW(base0.parameters()), 0, SFTConfig())

log("clean teacher games (train-only pools) …")
tgames = []
for s in range(5):
    tgames += [t.game for t in generate_transcripts(TRAINP, weak_frac=0.2, openers=SAFE_OPENERS, seed=s,
                                                    valid_pool=TRAINP, answer_pool=TRAINP)]
rng0 = Random(0)
EXS = [e for g in tgames for e in game_examples(g, rng0)]
TEACHER_EXS = EXS  # reuse for DAgger teacher-mix
log(f"{len(tgames)} games -> {len(EXS)} examples")

best_ckpt, best_val = None, -1.0


def consider(ckpt, val):
    global best_ckpt, best_val
    if val > best_val:
        best_val, best_ckpt = val, ckpt


# Stage A: clean SFT base
def _a():
    m = copy.deepcopy(base0).to(DEV)
    sft_aux(m, EXS, 50, 4e-4, "runs/on_sft.pt", "SFT")
    r = record("SFT (ephemeral-CoT+aux, clean)", m)
    consider("runs/on_sft.pt", r["val"])
    return "runs/on_sft.pt"


sft_ckpt = stage("A:SFT", _a) or "runs/on_pretrain.pt"


# Stage B: DAgger on SFT
def _b():
    dg = dagger(sft_ckpt, TEACHER_EXS, "runs/on_dagger.pt", "DAgger", train, epochs=8)
    r = record("DAgger (on SFT, clean)", dg)
    consider("runs/on_dagger.pt", r["val"])


stage("B:DAgger", _b)


# Stage C: DPO on SFT
def _c():
    dp = dpo(sft_ckpt, "runs/on_dpo.pt", "DPO", train[:600])
    r = record("DPO (on SFT, clean)", dp)
    consider("runs/on_dpo.pt", r["val"])


stage("C:DPO", _c)


# Stage D: DPO on DAgger (combo)
def _d():
    import os
    if not os.path.exists("runs/on_dagger.pt"):
        return
    dp = dpo("runs/on_dagger.pt", "runs/on_dagger_dpo.pt", "DAgger+DPO", train[:600])
    r = record("DPO∘DAgger (clean)", dp)
    consider("runs/on_dagger_dpo.pt", r["val"])


stage("D:DPO∘DAgger", _d)


# Stage E: GRPO on best-so-far
def _e():
    gp = grpo(best_ckpt, "runs/on_grpo.pt", "GRPO", train, updates=30)
    r = record("GRPO (on best, clean reward)", gp)
    consider("runs/on_grpo.pt", r["val"])


stage("E:GRPO", _e)


# Stage F: 2nd DAgger iteration (if budget)
def _f():
    import os
    if not os.path.exists("runs/on_dagger.pt"):
        return
    dg = dagger("runs/on_dagger.pt", TEACHER_EXS, "runs/on_dagger2.pt", "DAgger2", train, epochs=6)
    r = record("DAgger×2 (clean)", dg)
    consider("runs/on_dagger2.pt", r["val"])


stage("F:DAgger2", _f)


# ---- final: multi-seed TEST eval of the winner + analysis -----------------------------------
log(f"\n=== WINNER by VAL: {best_ckpt} (VAL={best_val:.3f}) ===")
seed_tests = []
if best_ckpt:
    try:
        bm = WordleGenerator(CFG, VOCAB).to(DEV)
        load_checkpoint(best_ckpt, bm)
        bm.eval()
        for sd in range(3):  # multi-seed TEST to beat MPS ±2-3pt noise
            torch.manual_seed(sd)
            seed_tests.append(round(win_rate(bm, TEST)["win"], 4))
        log(f"winner TEST over 3 seeds: {seed_tests}  mean={statistics.mean(seed_tests):.4f}")
    except Exception:
        log(traceback.format_exc())

ranked = sorted(RESULTS, key=lambda r: r["val"], reverse=True)
lines = ["# Overnight run — results (CLEAN / leak-free; honest TEST = held[96:])\n",
         f"_ran {time.strftime('%Y-%m-%d %H:%M')}; VAL=held[:96] (selection), TEST=held[96:] (report, never selected); FULL=held[463]._\n",
         "| setup | VAL | TEST | full | valid | avg |", "|---|---|---|---|---|---|"]
for r in ranked:
    lines.append(f"| {r['name']} | {r['val']} | **{r['test']}** | {r['full']} | {r['valid']} | {r['avg']} |")
lines.append("")
if best_ckpt:
    w = ranked[0]
    lines.append(f"**Best by VAL: {w['name']}** — honest TEST = **{w['test']}**"
                 + (f"; multi-seed TEST {seed_tests} mean **{statistics.mean(seed_tests):.4f}**" if seed_tests else "")
                 + f". Checkpoint: `{best_ckpt}`.")
    lines.append("\nMPS greedy eval is ±2-3pt noisy on these slice sizes (TEST=367 is more stable than the 96-curve);"
                 " treat sub-2pt gaps as ties. All stages clean (train-only pools, safe openers, disjoint VAL/TEST).")
with open("OVERNIGHT_ANALYSIS.md", "w") as f:
    f.write("\n".join(lines))
log("wrote OVERNIGHT_ANALYSIS.md")
log("\n[OVERNIGHT DONE]")
