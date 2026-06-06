"""DPO on DECISIVE-board commit pairs (driver, not committed): clean credit, fully honest greedy.

The noisy version (dpo_commit.py) paired at the first win/loss divergence — but the outcome isn't caused
by that first guess, so pref_acc capped at 0.73 and the gain at +1.5pts. This fixes the CREDIT: find
DECISIVE boards (where the secret is reachable — taken from winning rollouts' final turns), RE-SAMPLE M
responses at each, and pair chosen = "commits the secret" vs rejected = "commits a wrong CONSISTENT word"
from the SAME board. The label is unambiguous (secret vs not), so the preference is clean.

Honesty: the secret + engine consistency are TRAINING-time labels on TRAIN secrets only (same as the
teacher / expert-iteration / aux-validity). Inference = greedy ephemeral-CoT on HELD-OUT, ZERO rules.
Base runs/cot_eph_aux.pt -> runs/dpo_decisive.pt.
"""

from __future__ import annotations

import copy
import statistics
import time
from random import Random

import torch
import torch.nn.functional as F

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint, save_checkpoint

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
ROWS, TEMP, BETA = 6, 0.9, 0.1
N_SECRETS, N_FIND, M_RESAMPLE, BOARD_S, PAIRS_PER_BOARD = 700, 8, 14, 8, 4


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


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


def _word(c):
    w = "".join(tok.id_to_token(t) for t in c[:5])
    return w if len(w) == 5 else "zzzzz"


@torch.no_grad()
def rollout_batch(model, secrets, temp, rows=ROWS):
    games = [Game(s, max_guesses=rows) for s in secrets]
    for _ in range(rows):
        active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
        if not active:
            break
        prompts = [board_only(games[i].turns) for i in active]
        _, collected = batched_generate(model, prompts, temp)
        for j, i in enumerate(active):
            games[i].guess(_word(collected[j]))
    return games


@torch.no_grad()
def play_greedy(model, secret, rows=ROWS):
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
        g.guess(_word(guess))
    return g


def evaluate(model, secrets, rows=ROWS):
    model.eval()
    games = [play_greedy(model, s, rows) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


@torch.no_grad()
def find_decisive_boards(model, secrets, rng):
    """Decisive boards = the last 2 boards of winning rollouts (secret reachable). Dedup."""
    boards, seen = [], set()
    for c0 in range(0, len(secrets), BOARD_S):
        secs = secrets[c0 : c0 + BOARD_S]
        batch = [s for s in secs for _ in range(N_FIND)]
        games = rollout_batch(model, batch, TEMP)
        for i, g in enumerate(games):
            if not g.won:
                continue
            tw = len(g.turns) - 1
            for t in {tw, max(0, tw - 1)}:
                turns = g.turns[:t]
                key = (g.secret, t, tuple(x.guess for x in turns))
                if key in seen:
                    continue
                seen.add(key)
                boards.append((board_only(turns), turns, g.secret))
        if (c0 // BOARD_S) % 10 == 0:
            print(f"    [find] {c0 + len(secs)}/{len(secrets)} secrets  decisive_boards={len(boards)}", flush=True)
    rng.shuffle(boards)
    return boards


@torch.no_grad()
def build_pairs(model, boards, rng):
    """Resample M responses per decisive board; chosen=commits secret, rejected=valid consistent !=secret."""
    pairs = []
    for c0 in range(0, len(boards), BOARD_S):
        chunk = boards[c0 : c0 + BOARD_S]
        prompts = [b for b, _, _ in chunk for _ in range(M_RESAMPLE)]
        gen, collected = batched_generate(model, prompts, TEMP)
        for bi, (board, turns, secret) in enumerate(chunk):
            chosen, rejected = [], []
            for k in range(M_RESAMPLE):
                idx = bi * M_RESAMPLE + k
                w = _word(collected[idx])
                resp = board + gen[idx]
                if w == secret:
                    chosen.append(resp)
                elif w != secret and is_valid(w) and is_consistent(w, turns):
                    rejected.append(resp)
            made = 0
            for ch in chosen:
                for rj in rejected:
                    if made >= PAIRS_PER_BOARD:
                        break
                    pairs.append((ch, rj, len(board)))
                    made += 1
        if (c0 // BOARD_S) % 10 == 0:
            print(f"    [pairs] {c0 + len(chunk)}/{len(boards)} boards  pairs={len(pairs)}", flush=True)
    rng.shuffle(pairs)
    return pairs


def seq_logp(model, seqs, starts):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_id, dtype=torch.long, device=DEV)
    rmask = torch.zeros((len(seqs), L), device=DEV)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=DEV)
        rmask[i, starts[i] : len(s)] = 1.0
    logits = model.forward(ids)
    logp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (logp * rmask[:, 1:]).sum(-1)


_, held = split(seed=0)
train, _ = split(seed=0)
curve = tuple(held[:96])
print(f"[load] runs/cot_eph_aux.pt (0.616)  beta={BETA} N_find={N_FIND} M={M_RESAMPLE} secrets={N_SECRETS}", flush=True)
policy = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_aux.pt", policy)
ref = copy.deepcopy(policy).to(DEV)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)
policy.eval()

rng = Random(0)
secrets = list(train)
rng.shuffle(secrets)
secrets = secrets[:N_SECRETS]
base = evaluate(policy, curve)
print(f"[base] held6 win={base['win']:.3f} valid={base['valid']:.3f}", flush=True)

t0 = time.time()
print("[find] decisive boards from winning rollouts …", flush=True)
boards = find_decisive_boards(policy, secrets, rng)
print(f"[find] {len(boards)} decisive boards  ({time.time() - t0:.0f}s)", flush=True)
t0 = time.time()
print("[pairs] resampling at decisive boards -> clean commit pairs …", flush=True)
pairs = build_pairs(policy, boards, rng)
print(f"[pairs] {len(pairs)} clean preference pairs  ({time.time() - t0:.0f}s)", flush=True)

opt = torch.optim.AdamW(policy.parameters(), lr=5e-6, weight_decay=0.0)
best = base["win"]
save_checkpoint("runs/dpo_decisive.pt", policy, opt, 0, SFTConfig())
EPOCHS, BATCH = 5, 16
for ep in range(EPOCHS):
    rng.shuffle(pairs)
    policy.eval()
    acc_n, acc_correct, last = 0, 0, 0.0
    for b0 in range(0, len(pairs), BATCH):
        bp = pairs[b0 : b0 + BATCH]
        ch, rj, cs = [c for c, _, _ in bp], [r for _, r, _ in bp], [s for _, _, s in bp]
        lp_c, lp_r = seq_logp(policy, ch, cs), seq_logp(policy, rj, cs)
        with torch.no_grad():
            rp_c, rp_r = seq_logp(ref, ch, cs), seq_logp(ref, rj, cs)
        delta = BETA * ((lp_c - rp_c) - (lp_r - rp_r))
        loss = -F.logsigmoid(delta).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        acc_n += len(bp)
        acc_correct += int((delta > 0).sum())
        last = float(loss.detach())
    m = evaluate(policy, curve)
    flag = ""
    if m["win"] > best:
        best = m["win"]
        save_checkpoint("runs/dpo_decisive.pt", policy, opt, ep + 1, SFTConfig())
        flag = "  <- best, saved"
    else:
        load_checkpoint("runs/dpo_decisive.pt", policy)
        flag = "  (regressed -> reverted)"
    print(f"[epoch {ep}] loss={last:.3f}  pref_acc={acc_correct / max(1, acc_n):.3f}  "
          f"held6 win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)

print(f"\n=== DECISIVE-board DPO: full held-out ({len(held)}, 6-row), best ckpt ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/dpo_decisive.pt", b)
f6 = evaluate(b, tuple(held))
print(f"  held-out @6-row : win {f6['win']:.3f}  ({int(round(f6['win'] * len(held)))}/{len(held)})  valid {f6['valid']:.3f} avg {f6['avg']:.2f}   [base 0.616, noisy-DPO 0.631]", flush=True)
print("\n[DPOD DONE]", flush=True)
