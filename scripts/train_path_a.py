"""Path A: scale up + imitate hard (driver script, not committed).

A ~25M decoder, deep vocab pretrain, and heavy imitation of the near-optimal InfoMax teacher
(99% / 3.55 on held-out). Goal: see how far PURE imitation climbs toward the teacher ceiling,
before any RL. Saves the best-by-held-out checkpoint to runs/sft_xl.pt.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from random import Random

import torch

logging.disable(logging.CRITICAL)

from wordle_slm.config import ModelConfig, SFTConfig  # noqa: E402
from wordle_slm.data import is_valid, split, train_probe  # noqa: E402
from wordle_slm.engine import Color  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.rl.grpo import eval_win_rate  # noqa: E402
from wordle_slm.rl.rollout import letter_id_tensor, play_game  # noqa: E402
from wordle_slm.sft.pretrain import pretrain_lm, pretrain_words  # noqa: E402
from wordle_slm.sft.train import _batches, load_checkpoint, make_batch, save_checkpoint, sft_loss  # noqa: E402
from wordle_slm.teacher import generate_transcripts  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
XL = ModelConfig(d_model=512, n_layers=8, n_heads=8, d_ff=2048)  # ~25M
train, held = split(seed=0)
curve_secrets = tuple(held[:96])
probe = train_probe(seed=0, size=96)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def evaluate(model: WordleGenerator, secrets: tuple[str, ...]) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [g for g in games if g.won]
    per_turn: dict[int, list[int]] = {}
    v = t = 0
    for g in games:
        for i, turn in enumerate(g.turns):
            ok = int(is_valid(turn.guess))
            per_turn.setdefault(i, [0, 0])
            per_turn[i][0] += ok
            per_turn[i][1] += 1
            v += ok
            t += 1
    return {
        "win": len(wins) / len(games),
        "avg_g": statistics.mean(g.guesses_used for g in wins) if wins else float("nan"),
        "valid": v / t if t else 0.0,
        "per_turn": {i: c[0] / c[1] for i, c in sorted(per_turn.items())},
    }


def render(g) -> str:
    lines = [f"  secret={g.secret}  [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        fb = "❌ not a word" if t.feedback is None else "".join(SQ[c] for c in t.feedback)
        lines.append(f"      {t.guess}  {fb}")
    return "\n".join(lines)


print(f"[1/3] deep spell warm-up ({XL.estimated_params() / 1e6:.0f}M params, 12 epochs) …", flush=True)
model = WordleGenerator(XL, tok.vocab_size).to(DEV)
pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=12, batch_size=512, device=DEV, seed=0)

print("[2/3] teacher transcripts (5 passes, 80% InfoMax) …", flush=True)
t0 = time.perf_counter()
games = []
for s in range(5):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
print(f"      {len(games)} games in {time.perf_counter() - t0:.0f}s", flush=True)

print("[3/3] SFT (warmup+cosine, save best-by-held-out) …", flush=True)
letter_ids = letter_id_tensor(tok, DEV)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
EPOCHS, BATCH, EVAL_EVERY = 100, 128, 5
steps_per_epoch = (len(games) + BATCH - 1) // BATCH
total_steps, warmup_steps = EPOCHS * steps_per_epoch, 300
PEAK, FLOOR = 4e-4, 4e-5


def lr_at(s: int) -> float:
    if s < warmup_steps:
        return PEAK * (s + 1) / warmup_steps
    p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
    return FLOOR + 0.5 * (PEAK - FLOOR) * (1 + math.cos(math.pi * p))


rng = Random(0)
best_win = -1.0
step = 0
model.train()
for epoch in range(EPOCHS):
    losses = []
    for idx in _batches(len(games), BATCH, rng):
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
        m = evaluate(model, curve_secrets)
        pt = " ".join(f"t{i + 1}={r:.2f}" for i, r in m["per_turn"].items())
        extra = ""
        if epoch % 20 == 0 or epoch == EPOCHS - 1:  # probe gap occasionally (it's extra eval cost)
            pw = eval_win_rate(model, tok, probe, device=DEV)
            extra = f"  probe={pw:.3f} gap={pw - m['win']:+.3f}"
        print(
            f"  epoch {epoch:>3}  loss={statistics.mean(losses):.3f}  lr={lr_at(step):.1e}  "
            f"win={m['win']:.3f} avg_g={m['avg_g']:.2f} valid={m['valid']:.3f}{extra}  [{pt}]",
            flush=True,
        )
        if m["win"] > best_win:
            best_win = m["win"]
            save_checkpoint("runs/sft_xl.pt", model, opt, epoch, SFTConfig())

print(f"\n=== Path A milestone: full held-out ({len(held)}) — best-by-held-out checkpoint ===", flush=True)
best = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_xl.pt", best)
final = evaluate(best, tuple(held))
pt = "  ".join(f"turn{i + 1}={r:.3f}" for i, r in final["per_turn"].items())
print(f"  held-out win rate : {final['win']:.3f}  ({int(round(final['win'] * len(held)))}/{len(held)})", flush=True)
print(f"  avg guesses/win   : {final['avg_g']:.2f}   (teacher ceiling: 3.55)", flush=True)
print(f"  valid-word rate   : {final['valid']:.3f}", flush=True)
print(f"  per-turn valid    : {pt}", flush=True)
print("\n=== sample greedy held-out games (final play) ===", flush=True)
best.eval()
for s in held[:12]:
    print(render(play_game(best, tok, s, sample=False, device=DEV)), flush=True)
print("\n[PATH A DONE]", flush=True)
