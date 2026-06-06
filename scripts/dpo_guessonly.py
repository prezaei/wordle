"""DPO scored on GUESS tokens only (driver, not committed): isolate the commit, fully honest greedy.

Both prior DPOs scored logp over the whole `<think>`+guess response, so the long think (3×6=18 tokens)
diluted/hijacked the 5-token commit — decisive-DPO's loss fell while held6 dropped. This scores the DPO
preference on the 5 GUESS-letter tokens ONLY (the actual commit), so the gradient sharpens the decision,
not the scratchpad. Pairs from the fast noisy pipeline (first win/loss divergence). CoT still generated
at inference (ephemeral); inference greedy on held-out, no rules. Base runs/cot_eph_aux.pt -> runs/dpo_go.pt.
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
    recs: list[list] = [[] for _ in secrets]
    for _ in range(rows):
        active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
        if not active:
            break
        prompts = [board_only(games[i].turns) for i in active]
        gen, collected = batched_generate(model, prompts, temp)
        for j, i in enumerate(active):
            games[i].guess(_word(collected[j]))
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


def build_pairs(model, secrets, rng):
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
                    if pw != pl:
                        continue
                    pairs.append((pw + gen_w, pw + gen_l))
                    made += 1
        print(f"    [dpo-go] sampled {c0 + len(secs)}/{len(secrets)} secrets  pairs={len(pairs)}", flush=True)
    rng.shuffle(pairs)
    return pairs


def _guess_mask_row(seq):
    """1.0 at the 5 committed guess-letter target positions (after the last <GUESS>)."""
    gi = max((i for i, t in enumerate(seq) if t == tok.guess_id), default=None)
    m = [0.0] * len(seq)
    if gi is None:
        return m
    got = 0
    for i in range(gi + 1, len(seq)):
        if seq[i] in LETTER_SET:
            m[i] = 1.0
            got += 1
        if got >= 5:
            break
    return m


def guess_logp(model, seqs):
    """Sum log p over the 5 GUESS-letter tokens only (the commit) per sequence."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_id, dtype=torch.long, device=DEV)
    gmask = torch.zeros((len(seqs), L), device=DEV)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=DEV)
        gmask[i, : len(s)] = torch.tensor(_guess_mask_row(s), device=DEV)
    logits = model.forward(ids)
    logp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (logp * gmask[:, 1:]).sum(-1)


_, held = split(seed=0)
train, _ = split(seed=0)
curve = tuple(held[:96])
print(f"[load] runs/cot_eph_aux.pt (0.616)  GUESS-ONLY DPO  beta={BETA} N_roll={N_ROLL} secrets={N_SECRETS}", flush=True)
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
pairs = build_pairs(policy, secrets, rng)
print(f"[dpo-go] {len(pairs)} pairs  ({time.time() - t0:.0f}s)", flush=True)

opt = torch.optim.AdamW(policy.parameters(), lr=5e-6, weight_decay=0.0)
best = base["win"]
save_checkpoint("runs/dpo_go.pt", policy, opt, 0, SFTConfig())
EPOCHS, BATCH = 5, 16
for ep in range(EPOCHS):
    rng.shuffle(pairs)
    policy.eval()
    acc_n, acc_correct, last = 0, 0, 0.0
    for b0 in range(0, len(pairs), BATCH):
        bp = pairs[b0 : b0 + BATCH]
        ch, rj = [c for c, _ in bp], [r for _, r in bp]
        lp_c, lp_r = guess_logp(policy, ch), guess_logp(policy, rj)
        with torch.no_grad():
            rp_c, rp_r = guess_logp(ref, ch), guess_logp(ref, rj)
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
        save_checkpoint("runs/dpo_go.pt", policy, opt, ep + 1, SFTConfig())
        flag = "  <- best, saved"
    else:
        load_checkpoint("runs/dpo_go.pt", policy)
        flag = "  (regressed -> reverted)"
    print(f"[epoch {ep}] loss={last:.3f}  pref_acc={acc_correct / max(1, acc_n):.3f}  "
          f"held6 win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)

print(f"\n=== GUESS-ONLY DPO: full held-out ({len(held)}, 6-row), best ckpt ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/dpo_go.pt", b)
f6 = evaluate(b, tuple(held))
print(f"  held-out @6-row : win {f6['win']:.3f}  ({int(round(f6['win'] * len(held)))}/{len(held)})  valid {f6['valid']:.3f} avg {f6['avg']:.2f}   [base 0.616, full-resp DPO 0.631]", flush=True)
print("\n[DPOGO DONE]", flush=True)
