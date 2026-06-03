"""Strong-SFT training run with a held-out learning curve (not committed; a driver script).

pretrain (spell warm-up) -> augmented teacher transcripts -> SFT with periodic held-out eval.
Milestone goal: greedy play valid on ALL turns + wins some held-out games. Saves the
best-by-held-out checkpoint to runs/sft_strong.pt for the subsequent GRPO run.
"""

from __future__ import annotations

import logging
import statistics
import time
from random import Random

import torch

logging.disable(logging.CRITICAL)  # silence INFO; this script prints its own curve

from wordle_slm.config import ModelConfig, SFTConfig  # noqa: E402
from wordle_slm.data import is_valid, split  # noqa: E402
from wordle_slm.engine import Color  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.rl.rollout import letter_id_tensor, play_game  # noqa: E402
from wordle_slm.sft.pretrain import pretrain_lm, pretrain_words  # noqa: E402
from wordle_slm.sft.train import _batches, make_batch, save_checkpoint, sft_loss  # noqa: E402
from wordle_slm.teacher import generate_transcripts  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
train, held = split(seed=0)
curve_secrets = tuple(held[:100])  # fixed held-out subsample for the curve (comparable)
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def evaluate(model: WordleGenerator, secrets: tuple[str, ...]) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [g for g in games if g.won]
    per_turn: dict[int, list[int]] = {}
    v_all = t_all = 0
    for g in games:
        for i, turn in enumerate(g.turns):
            ok = int(is_valid(turn.guess))
            per_turn.setdefault(i, [0, 0])
            per_turn[i][0] += ok
            per_turn[i][1] += 1
            v_all += ok
            t_all += 1
    return {
        "win": len(wins) / len(games),
        "avg_g": statistics.mean(g.guesses_used for g in wins) if wins else float("nan"),
        "valid": v_all / t_all if t_all else 0.0,
        "per_turn": {i: (c[0] / c[1], c[1]) for i, c in sorted(per_turn.items())},
        "n": len(games),
    }


def render(g) -> str:
    head = f"  secret={g.secret}  [{g.status.value} in {g.guesses_used}]"
    lines = [head]
    for t in g.turns:
        fb = "❌ not a word" if t.feedback is None else "".join(SQ[c] for c in t.feedback)
        lines.append(f"      {t.guess}  {fb}")
    return "\n".join(lines)


print("[1/3] spell warm-up …", flush=True)
model = WordleGenerator(ModelConfig(), tok.vocab_size).to(DEV)
pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=4, batch_size=512, device=DEV, seed=0)

print("[2/3] teacher transcripts (3 augmentation passes) …", flush=True)
t0 = time.perf_counter()
games = []
for s in range(3):
    games += [t.game for t in generate_transcripts(tuple(train), weak_frac=0.5, seed=s)]
print(f"      {len(games)} games in {time.perf_counter() - t0:.0f}s", flush=True)

print("[3/3] SFT with held-out curve …", flush=True)
letter_ids = letter_id_tensor(tok, DEV)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
rng = Random(0)
EPOCHS, BATCH, EVAL_EVERY = 60, 96, 3
best_win = -1.0
model.train()
for epoch in range(EPOCHS):
    losses = []
    for idx in _batches(len(games), BATCH, rng):
        ids, tgt, mask = make_batch([games[i] for i in idx], tok, DEV)
        loss = sft_loss(model, ids, tgt, mask, letter_ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve_secrets)
        pt = " ".join(f"t{i + 1}={r:.2f}" for i, (r, _) in m["per_turn"].items())
        print(
            f"  epoch {epoch:>2}  loss={statistics.mean(losses):.3f}  "
            f"held-out win={m['win']:.3f} avg_g={m['avg_g']:.2f}  valid={m['valid']:.3f}  [{pt}]",
            flush=True,
        )
        if m["win"] > best_win:
            best_win = m["win"]
            save_checkpoint("runs/sft_strong.pt", model, opt, epoch, SFTConfig())

print(f"\n=== FINAL milestone: full held-out ({len(held)}) ===", flush=True)
final = evaluate(model, tuple(held))
pt = "  ".join(f"turn{i + 1}={r:.3f} (n={n})" for i, (r, n) in final["per_turn"].items())
print(f"  held-out win rate : {final['win']:.3f}  ({int(final['win'] * len(held))}/{len(held)})", flush=True)
print(f"  avg guesses/win   : {final['avg_g']:.2f}", flush=True)
print(f"  valid-word rate   : {final['valid']:.3f}", flush=True)
print(f"  per-turn valid    : {pt}", flush=True)
print(f"  best curve win    : {best_win:.3f}  (checkpoint runs/sft_strong.pt)", flush=True)

print("\n=== sample greedy held-out games (final play) ===", flush=True)
model.eval()
for s in held[:12]:
    print(render(play_game(model, tok, s, sample=False, device=DEV)), flush=True)
