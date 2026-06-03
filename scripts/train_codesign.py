"""Co-design payoff (driver, not committed): the redesigned 50M model + the diverse curriculum.

The control (99M on the OLD answer-only data) plateaued — scale alone doesn't help. This pairs the
redesigned `large` preset (~50M, depth-emphasized, dropout 0.15) with the redesigned diverse data:
strong InfoMax on the answer distribution + Consistent on rarer valid words (from the curriculum's
diverse pool), so the bigger model has diverse data to GENERALIZE from instead of memorize.
Tracks held-out win/valid/consistent + the probe gap. Best-by-held-out -> runs/sft_codesign.pt.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from random import Random

import torch

logging.disable(logging.CRITICAL)

from wordle_slm.baselines.policies import ConsistentGuesser, play  # noqa: E402
from wordle_slm.config import ModelConfig, SFTConfig  # noqa: E402
from wordle_slm.data import is_valid, load_answers, load_valid_guesses, split, train_probe  # noqa: E402
from wordle_slm.engine import Color  # noqa: E402
from wordle_slm.engine.constraints import is_consistent  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.rl.curriculum import build_curriculum_pool  # noqa: E402
from wordle_slm.rl.grpo import eval_win_rate  # noqa: E402
from wordle_slm.rl.rollout import letter_id_tensor, play_game  # noqa: E402
from wordle_slm.sft.pretrain import pretrain_lm, pretrain_words  # noqa: E402
from wordle_slm.sft.train import _batches, load_checkpoint, make_batch, save_checkpoint, sft_loss  # noqa: E402
from wordle_slm.teacher import generate_transcripts  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
LARGE = ModelConfig.preset("large")  # ~50M, depth 16, dropout 0.15
train, held = split(seed=0)
curve = tuple(held[:80])
probe = train_probe(seed=0, size=80)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def evaluate(model, secrets) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [g for g in games if g.won]
    valid = total = consistent = vc = 0
    for g in games:
        prior = []
        for turn in g.turns:
            total += 1
            v = is_valid(turn.guess)
            valid += int(v)
            if v:
                vc += 1
                consistent += int(is_consistent(turn.guess, prior))
            prior.append(turn)
    return {"win": len(wins) / len(games), "valid": valid / total,
            "consistent": consistent / vc if vc else 0.0,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


print(f"[1/3] deep spell warm-up (LARGE preset, {LARGE.estimated_params() / 1e6:.0f}M params) …", flush=True)
model = WordleGenerator(LARGE, tok.vocab_size).to(DEV)
pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=10, batch_size=256, device=DEV, seed=0)

print("[2/3] DIVERSE teacher data (InfoMax on answers + Consistent on rarer valid) …", flush=True)
t0 = time.perf_counter()
pool = build_curriculum_pool(seed=0)
answer_set = set(load_answers())
answer_secrets = tuple(w for w in pool if w in answer_set)  # = train answers, difficulty-ordered
rare_secrets = [w for w in pool if w not in answer_set]  # rarer valid words
games = []
for s in range(3):  # strong InfoMax on the eval distribution
    games += [tr.game for tr in generate_transcripts(answer_secrets, weak_frac=0.2, openers=OPENERS, seed=s)]
n_strong = len(games)
valid = load_valid_guesses()
rng = Random(0)
for i, secret in enumerate(rng.sample(rare_secrets, 5500)):  # diversity from rarer valid words
    games.append(play(ConsistentGuesser(opener=rng.choice(OPENERS), seed=i, pool=valid), secret))
print(f"      {len(games)} games ({n_strong} InfoMax-answers + {len(games) - n_strong} Consistent-rare) in {time.perf_counter() - t0:.0f}s", flush=True)

print("[3/3] SFT (warmup+cosine, save best-by-held-out) …", flush=True)
letter_ids = letter_id_tensor(tok, DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
EPOCHS, BATCH, EVAL_EVERY = 45, 96, 5
steps_per_epoch = (len(games) + BATCH - 1) // BATCH
total_steps, warmup_steps = EPOCHS * steps_per_epoch, 400
PEAK, FLOOR = 3e-4, 3e-5


def lr_at(s):
    if s < warmup_steps:
        return PEAK * (s + 1) / warmup_steps
    p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
    return FLOOR + 0.5 * (PEAK - FLOOR) * (1 + math.cos(math.pi * p))


rng2 = Random(0)
best_win = -1.0
step = 0
model.train()
for epoch in range(EPOCHS):
    losses = []
    for idx in _batches(len(games), BATCH, rng2):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        ids, tgt, mask = make_batch([games[i] for i in idx], tok, DEV)
        loss = sft_loss(model, ids, tgt, mask, letter_ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        step += 1
    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve)
        extra = ""
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            pw = eval_win_rate(model, tok, probe, device=DEV)
            extra = f"  probe={pw:.3f} gap={pw - m['win']:+.3f}"
        print(f"  epoch {epoch:>2}  loss={statistics.mean(losses):.3f}  win={m['win']:.3f} valid={m['valid']:.3f} consistent={m['consistent']:.3f} avg={m['avg']:.2f}{extra}", flush=True)
        if m["win"] > best_win:
            best_win = m["win"]
            save_checkpoint("runs/sft_codesign.pt", model, opt, epoch, SFTConfig())

print(f"\n=== CO-DESIGN milestone: full held-out ({len(held)}) — 50M + diverse data ===", flush=True)
b = WordleGenerator(LARGE, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_codesign.pt", b)
f = evaluate(b, tuple(held))
pw = eval_win_rate(b, tok, train_probe(seed=0, size=128), device=DEV)
print(f"  held-out win : {f['win']:.3f}  ({int(round(f['win'] * len(held)))}/{len(held)})   [25M+old-data was 0.391]", flush=True)
print(f"  valid-rate   : {f['valid']:.3f}   consistent : {f['consistent']:.3f}   avg : {f['avg']:.2f}", flush=True)
print(f"  probe(train) : {pw:.3f}   gap : {pw - f['win']:+.3f}   [25M gap was ~+0.38]", flush=True)
print("\n=== sample greedy held-out games ===", flush=True)
b.eval()
for s in held[:10]:
    g = play_game(b, tok, s, sample=False, device=DEV)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[CODESIGN DONE]", flush=True)
