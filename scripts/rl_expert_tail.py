"""RL Phase 1b — higher-ceiling expert-iteration: full coverage + tail-focused high-K (driver).

Phase 1 plateaued (held10 +3.1 -> +1.1 -> revert) because expert-iteration's ceiling is pass@K, and
re-winning easy secrets wastes budget. This RAISES the ceiling: pass 0 samples ALL train secrets once
(coverage + identify the unsolved tail); later passes pour high-K / high-temp sampling ONLY at the
secrets still unsolved (the hard tail), to surface wins that low-K never finds. Winning trajectories
are accumulated (union over passes, clean teacher-think rebuild, the model's achievable guesses) and
SFT'd with aux-validity + a teacher mix. Honest: train TRAIN answers, eval held-out, no rules.
Starts from runs/rl_expert.pt (Phase-1 best 0.646) -> runs/rl_expert_tail.pt.
"""

from __future__ import annotations

import statistics
import time
from random import Random

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_answers, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import consistent_candidates
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
ANSWERS = load_answers()
K_CANDS = 3
AUX_LAMBDA = 0.5
ROWS = 10
MAX_WIN_TURNS = 9
SAMPLE_S = 8
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
# (scope, K rollouts/secret, temperature): pass 0 = full coverage; rest = hard tail, hotter + higher K
PASSES = [("all", 10, 1.0), ("tail", 24, 1.1), ("tail", 32, 1.2), ("tail", 48, 1.3)]


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def pick_cands(history, guess, rng):
    cons = [w for w in consistent_candidates(history, ANSWERS) if w != guess]
    rng.shuffle(cons)
    cands = [guess] + cons[: K_CANDS - 1]
    rng.shuffle(cands)
    return cands


def teacher_examples(game, rng):
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
    return collected


@torch.no_grad()
def rollout_batch(model, secrets, temp, rows=ROWS):
    games = [Game(s, max_guesses=rows) for s in secrets]
    for _ in range(rows):
        active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
        if not active:
            break
        prompts = [board_only(games[i].turns) for i in active]
        collected = batched_generate(model, prompts, temp)
        for j, i in enumerate(active):
            word = "".join(tok.id_to_token(t) for t in collected[j][:5])
            games[i].guess(word if len(word) == 5 else "zzzzz")
    return games


@torch.no_grad()
def play(model, secret, rows):
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


def evaluate(model, secrets, rows):
    model.eval()
    games = [play(model, s, rows) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def sft_pass(model, opt, exs, epochs, rng):
    model.train()
    last = 0.0
    for _ in range(epochs):
        for idx in _batches(len(exs), 128, rng):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            vmask = cot_valid_mask([s for s, _ in bs]).to(DEV)
            ids, tmask = ids.to(DEV), tmask.to(DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            imit = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
            vm = vmask[:, :-1]
            aux_pos = (vm.sum(-1) > 0).float() * tmask[:, 1:]
            valid_mass = (logp_let.exp() * vm).sum(-1).clamp_min(1e-9)
            aux = (-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
            loss = imit + AUX_LAMBDA * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.detach())
    return last


train, held = split(seed=0)
curve = tuple(held[:120])
print("[load] runs/rl_expert.pt (Phase-1 best 0.646) …", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/rl_expert.pt", model)

print("[teacher-mix] 1 InfoMax pass …", flush=True)
trng = Random(0)
tgames = [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=0)]
TEACHER_POOL = [e for g in tgames for e in teacher_examples(g, trng)]

b6 = evaluate(model, curve, 6)
b10 = evaluate(model, curve, ROWS)
print(f"[base] held6={b6['win']:.3f}  held{ROWS}={b10['win']:.3f} valid={b10['valid']:.3f}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
rngS = Random(11)
best = b10["win"]
save_checkpoint("runs/rl_expert_tail.pt", model, opt, 0, SFTConfig())
solved: set[str] = set()
wins_by_secret: dict[str, list] = {}

for pi, (scope, K, temp) in enumerate(PASSES):
    t0 = time.time()
    targets = list(train) if scope == "all" else [s for s in train if s not in solved]
    rngS.shuffle(targets)
    if not targets:
        print(f"[pass {pi}] tail empty — stopping", flush=True)
        break
    model.eval()
    n_games, n_chunks = 0, (len(targets) + SAMPLE_S - 1) // SAMPLE_S
    for ci, c0 in enumerate(range(0, len(targets), SAMPLE_S)):
        secs = targets[c0 : c0 + SAMPLE_S]
        batch_secrets = [s for s in secs for _ in range(K)]
        games = rollout_batch(model, batch_secrets, temp)
        n_games += len(games)
        for si, s in enumerate(secs):
            grp = [games[si * K + k] for k in range(K)]
            wins = sorted([g for g in grp if g.won and g.guesses_used <= MAX_WIN_TURNS],
                          key=lambda g: g.guesses_used)
            if wins:
                solved.add(s)
                wins_by_secret[s] = [e for g in wins[:2] for e in teacher_examples(g, rngS)]
        if ci % 10 == 0 or ci == n_chunks - 1:
            print(f"    [pass {pi} {scope} K{K}] sampling {c0 + len(secs)}/{len(targets)} secrets  "
                  f"solved={len(solved)}/{len(train)}  {n_games} games  ({time.time() - t0:.0f}s)", flush=True)
    win_examples = [e for exs in wins_by_secret.values() for e in exs]
    teacher_sample = [TEACHER_POOL[i] for i in torch.randint(0, len(TEACHER_POOL), (len(win_examples) // 2 + 1,)).tolist()]
    pool = win_examples + teacher_sample
    rngS.shuffle(pool)
    loss = sft_pass(model, opt, pool, epochs=3, rng=rngS)
    e6 = evaluate(model, curve, 6)
    e10 = evaluate(model, curve, ROWS)
    flag = ""
    if e10["win"] > best:
        best = e10["win"]
        save_checkpoint("runs/rl_expert_tail.pt", model, opt, pi + 1, SFTConfig())
        flag = "  <- best, saved"
    else:
        load_checkpoint("runs/rl_expert_tail.pt", model)
        flag = "  (regressed -> reverted)"
    print(f"[pass {pi} {scope} K{K} T{temp}] {n_games} games  solved={len(solved)}/{len(train)}  "
          f"pool={len(pool)}  loss={loss:.3f}  held6={e6['win']:.3f}  held{ROWS}={e10['win']:.3f} "
          f"valid={e10['valid']:.3f} avg={e10['avg']:.2f}  ({time.time() - t0:.0f}s){flag}", flush=True)

print(f"\n=== higher-ceiling expert-iteration: full held-out ({len(held)}), best ckpt ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/rl_expert_tail.pt", b)
f6 = evaluate(b, tuple(held), 6)
f10 = evaluate(b, tuple(held), ROWS)
print(f"  held-out @6-row  : win {f6['win']:.3f}  ({int(round(f6['win'] * len(held)))}/{len(held)})  valid {f6['valid']:.3f} avg {f6['avg']:.2f}   [base 0.616, Phase-1 0.646 curve]", flush=True)
print(f"  held-out @{ROWS}-row : win {f10['win']:.3f}  ({int(round(f10['win'] * len(held)))}/{len(held)})  valid {f10['valid']:.3f} avg {f10['avg']:.2f}", flush=True)
print(f"  reachable (solved by sampling): {len(solved)}/{len(train)} train secrets", flush=True)
print("\n=== sample 10-row games ===", flush=True)
b.eval()
for s in held[:6]:
    g = play(b, s, ROWS)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[TAIL DONE]", flush=True)
