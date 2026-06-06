"""RL Phase 1 — expert-iteration (ReST/STaR) on the 0.616 base, 10-row games (driver, not committed).

pass@10 ~= 0.79 means the winning lines are already in the policy's sampling distribution; the job is
to make GREEDY commit to them. Expert-iteration does exactly that: sample K 10-row rollouts/secret with
the current policy, KEEP the winning trajectories (the model's OWN think+guess), and SFT on them (with
the aux-validity loss + a teacher-mix for breadth), then iterate. Honest: train on TRAIN answers only,
eval held-out; ephemeral CoT (board-only history, think regenerated+discarded), ZERO rules at inference.
Batched multi-game rollout so 10-row sampling is minutes not hours. -> runs/rl_expert.pt.
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
from wordle_slm.sft import pretrain_lm, pretrain_words  # noqa: F401  (kept for parity; base is loaded)
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
TEMP = 0.9
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


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
    """Teacher-mix SFT examples (board-only history + teacher think+guess)."""
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
def batched_generate(model, prompts, max_new=44, temp=TEMP):
    """Sample think+guess for a batch of variable-length board-only prompts (right-padded, causal)."""
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
        logits = model.forward(ids)[ar, cur - 1]  # [B, V] at each seq's last real position
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
def rollout_batch(model, secrets, rows=ROWS):
    """Batched 10-row games. Returns games + per-game per-turn (seq, action_mask) records."""
    games = [Game(s, max_guesses=rows) for s in secrets]
    records: list[list] = [[] for _ in secrets]
    for _ in range(rows):
        active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
        if not active:
            break
        prompts = [board_only(games[i].turns) for i in active]
        gen, collected = batched_generate(model, prompts)
        for j, i in enumerate(active):
            word = "".join(tok.id_to_token(t) for t in collected[j][:5])
            games[i].guess(word if len(word) == 5 else "zzzzz")
            records[i].append((prompts[j] + gen[j], [False] * len(prompts[j]) + [True] * len(gen[j])))
    return games, records


@torch.no_grad()
def play(model, secret, rows):
    """Greedy ephemeral play (eval)."""
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
    return float(loss.detach())


train, held = split(seed=0)
curve = tuple(held[:96])
print("[load] runs/cot_eph_aux.pt (the 0.616 base) …", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_aux.pt", model)

print("[teacher-mix] 1 InfoMax pass for breadth …", flush=True)
trng = Random(0)
tgames = [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=0)]
TEACHER_POOL = [e for g in tgames for e in teacher_examples(g, trng)]
print(f"      {len(TEACHER_POOL)} teacher examples", flush=True)

b0 = evaluate(model, curve, 6)
b1 = evaluate(model, curve, ROWS)
print(f"[base] held6 win={b0['win']:.3f}  held{ROWS} win={b1['win']:.3f} valid={b1['valid']:.3f}", flush=True)

ITERS, SECRETS_PER_ITER, K, SAMPLE_S, MAX_WIN_TURNS = 4, 400, 12, 8, 8
opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
rngS = Random(1)
best = b1["win"]
save_checkpoint("runs/rl_expert.pt", model, opt, 0, SFTConfig())  # seed with the base
order = list(train)
rngS.shuffle(order)
ptr = 0
for it in range(ITERS):
    t0 = time.time()
    chunk = [order[(ptr + i) % len(order)] for i in range(SECRETS_PER_ITER)]
    ptr += SECRETS_PER_ITER
    win_games, n_win, n_games = [], 0, 0
    n_chunks = (len(chunk) + SAMPLE_S - 1) // SAMPLE_S
    model.eval()
    for ci, c0 in enumerate(range(0, len(chunk), SAMPLE_S)):
        secs = chunk[c0 : c0 + SAMPLE_S]
        batch_secrets = [s for s in secs for _ in range(K)]
        games, _ = rollout_batch(model, batch_secrets)
        n_games += len(games)
        for si in range(len(secs)):
            grp = [games[si * K + k] for k in range(K)]
            wins = sorted([g for g in grp if g.won and g.guesses_used <= MAX_WIN_TURNS],
                          key=lambda g: g.guesses_used)
            for g in wins[:2]:  # up to 2 shortest achievable wins/secret
                win_games.append(g)
                n_win += 1
        if ci % 10 == 0 or ci == n_chunks - 1:
            print(f"    [iter {it}] sampling {c0 + len(secs)}/{len(chunk)} secrets  "
                  f"{n_win} wins  {n_games} games  ({time.time() - t0:.0f}s)", flush=True)
    # rebuild winning games with CLEAN teacher think + the model's achievable winning guesses (RAFT)
    win_examples = [e for g in win_games for e in teacher_examples(g, rngS)]
    teacher_sample = [TEACHER_POOL[i] for i in torch.randint(0, len(TEACHER_POOL), (len(win_examples) // 2 + 1,)).tolist()]
    pool = win_examples + teacher_sample
    rngS.shuffle(pool)
    loss = sft_pass(model, opt, pool, epochs=3, rng=rngS)
    e6 = evaluate(model, curve, 6)
    e10 = evaluate(model, curve, ROWS)
    flag = ""
    if e10["win"] > best:
        best = e10["win"]
        save_checkpoint("runs/rl_expert.pt", model, opt, it + 1, SFTConfig())
        flag = "  <- best, saved"
    else:
        load_checkpoint("runs/rl_expert.pt", model)  # revert: never compound a regression
        flag = "  (regressed -> reverted to best)"
    print(f"[iter {it}] {n_win} wins / {n_games} games  pool={len(pool)}  loss={loss:.3f}  "
          f"held6={e6['win']:.3f}  held{ROWS}={e10['win']:.3f} valid={e10['valid']:.3f} avg={e10['avg']:.2f}"
          f"  ({time.time() - t0:.0f}s){flag}", flush=True)

print(f"\n=== expert-iteration: full held-out ({len(held)}), best checkpoint ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/rl_expert.pt", b)
f6 = evaluate(b, tuple(held), 6)
f10 = evaluate(b, tuple(held), ROWS)
print(f"  held-out @6-row  : win {f6['win']:.3f}  ({int(round(f6['win'] * len(held)))}/{len(held)})  valid {f6['valid']:.3f} avg {f6['avg']:.2f}   [base 0.616]", flush=True)
print(f"  held-out @{ROWS}-row : win {f10['win']:.3f}  ({int(round(f10['win'] * len(held)))}/{len(held)})  valid {f10['valid']:.3f} avg {f10['avg']:.2f}", flush=True)
print("\n=== sample 10-row games ===", flush=True)
b.eval()
for s in held[:6]:
    g = play(b, s, ROWS)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[EXPERT DONE]", flush=True)
