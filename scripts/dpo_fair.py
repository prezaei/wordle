"""DPO on self-play commit pairs (driver, not committed): sharpen the commit, fully honest greedy.

The commit gap: pass@12=0.95 but greedy 0.62 — the model can find the line, greedy commits wrong, and
plain distillation plateaued. DPO targets it directly: per train secret, sample rollouts; where a
WINNING line and a LOSING line diverge from the SAME board, the winning think+guess is "chosen" and the
losing one "rejected". DPO's contrastive loss raises P(winning commit) over P(losing commit) relative to
a frozen ref. Outcome labels (win/loss) come from the engine at TRAINING time only (honest, like the
teacher); inference stays greedy ephemeral-CoT with ZERO rules. Base runs/cot_eph_aux.pt -> runs/cot_eph_aux_dpofair.pt.
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
N_SECRETS, N_ROLL, SAMPLE_S, PAIRS_PER_SECRET = 600, 8, 8, 4
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
def rollout_batch(model, secrets, temp, rows=ROWS):
    """Returns games + per-game per-turn (prompt_ids, gen_ids)."""
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
            recs[i].append((prompts[j], gen[j]))
    return games, recs


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
        word = "".join(tok.id_to_token(t) for t in guess[:5])
        g.guess(word if len(word) == 5 else "zzzzz")
    return g


def evaluate(model, secrets, rows=ROWS):
    model.eval()
    games = [play_greedy(model, s, rows) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def build_pairs(model, secrets, rng):
    """Collect (chosen_seq, rejected_seq, resp_start) DPO pairs at win/loss divergence boards."""
    pairs = []
    for c0 in range(0, len(secrets), SAMPLE_S):
        secs = secrets[c0 : c0 + SAMPLE_S]
        batch = [s for s in secs for _ in range(N_ROLL)]
        games, recs = rollout_batch(model, batch, TEMP)
        for si in range(len(secs)):
            idxs = [si * N_ROLL + k for k in range(N_ROLL)]
            won = [i for i in idxs if games[i].won]
            lost = [i for i in idxs if not games[i].won]
            if not won or not lost:
                continue
            made = 0
            for wi in won:
                for li in lost:
                    if made >= PAIRS_PER_SECRET:
                        break
                    gw = [t.guess for t in games[wi].turns]
                    gl = [t.guess for t in games[li].turns]
                    t = next((k for k in range(min(len(gw), len(gl))) if gw[k] != gl[k]), None)
                    if t is None:
                        continue
                    pw, gen_w = recs[wi][t]
                    pl, gen_l = recs[li][t]
                    if pw != pl:  # boards must match (same prefix) to be a fair pair
                        continue
                    pairs.append((pw + gen_w, pw + gen_l, len(pw)))
                    made += 1
        print(f"    [dpo] sampled {c0 + len(secs)}/{len(secrets)} secrets  pairs={len(pairs)}", flush=True)
    rng.shuffle(pairs)
    return pairs


def seq_logp(model, seqs, starts):
    """Sum log p over response tokens (positions >= start) per sequence. seqs: list[list[int]]."""
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
print(f"[load] runs/cot_eph_aux.pt (fair 0.281 base)  beta={BETA} N_roll={N_ROLL} secrets={N_SECRETS}", flush=True)
policy = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_aux_fair.pt", policy)
ref = copy.deepcopy(policy).to(DEV)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)
policy.eval()  # no dropout for clean DPO

rng = Random(0)
secrets = list(train)
rng.shuffle(secrets)
secrets = secrets[:N_SECRETS]

base = evaluate(policy, curve)
print(f"[base] held6 win={base['win']:.3f} valid={base['valid']:.3f} avg={base['avg']:.2f}", flush=True)
print("[dpo] sampling rollouts -> win/loss divergence pairs …", flush=True)
t0 = time.time()
pairs = build_pairs(policy, secrets, rng)
print(f"[dpo] {len(pairs)} preference pairs  ({time.time() - t0:.0f}s)", flush=True)

opt = torch.optim.AdamW(policy.parameters(), lr=5e-6, weight_decay=0.0)
best = base["win"]
save_checkpoint("runs/cot_eph_aux_dpofair.pt", policy, opt, 0, SFTConfig())
EPOCHS, BATCH = 4, 16
for ep in range(EPOCHS):
    rng.shuffle(pairs)
    policy.eval()
    acc_n, acc_correct, last = 0, 0, 0.0
    for b0 in range(0, len(pairs), BATCH):
        bp = pairs[b0 : b0 + BATCH]
        ch = [c for c, _, _ in bp]
        rj = [r for _, r, _ in bp]
        cs = [s for _, _, s in bp]
        lp_c = seq_logp(policy, ch, cs)
        lp_r = seq_logp(policy, rj, cs)
        with torch.no_grad():
            rp_c = seq_logp(ref, ch, cs)
            rp_r = seq_logp(ref, rj, cs)
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
        save_checkpoint("runs/cot_eph_aux_dpofair.pt", policy, opt, ep + 1, SFTConfig())
        flag = "  <- best, saved"
    else:
        load_checkpoint("runs/cot_eph_aux_dpofair.pt", policy)
        flag = "  (regressed -> reverted)"
    print(f"[epoch {ep}] loss={last:.3f}  pref_acc={acc_correct / max(1, acc_n):.3f}  "
          f"held6 win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)

print(f"\n=== DPO commit-sharpening: full held-out ({len(held)}, 6-row), best ckpt ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_aux_dpofair.pt", b)
f6 = evaluate(b, tuple(held))
print(f"  held-out @6-row : win {f6['win']:.3f}  ({int(round(f6['win'] * len(held)))}/{len(held)})  valid {f6['valid']:.3f} avg {f6['avg']:.2f}   [fair base 0.281]", flush=True)
print("\n[DPO DONE]", flush=True)
