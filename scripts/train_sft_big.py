"""Phase A: strong SFT on the bigger 4.8M model (driver script, not committed).

Deeper pretrain (full valid vocab) + more augmented teacher data + cosine-decay SFT, to raise
late-game valid-word rate before RL. Saves the best-by-held-out checkpoint to runs/sft_big.pt.
"""

from __future__ import annotations

import logging
import statistics
import time
from random import Random

import torch

logging.disable(logging.CRITICAL)

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
BIG = ModelConfig(d_model=256, n_layers=6, n_heads=8, d_ff=1024)  # ~4.8M
train, held = split(seed=0)
curve_secrets = tuple(held[:100])
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
        "per_turn": {i: (c[0] / c[1], c[1]) for i, c in sorted(per_turn.items())},
    }


def render(g) -> str:
    lines = [f"  secret={g.secret}  [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        fb = "❌ not a word" if t.feedback is None else "".join(SQ[c] for c in t.feedback)
        lines.append(f"      {t.guess}  {fb}")
    return "\n".join(lines)


print(f"[1/3] deep spell warm-up (4.8M model, ~{BIG.estimated_params() / 1e6:.1f}M params) …", flush=True)
model = WordleGenerator(BIG, tok.vocab_size).to(DEV)
pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=8, batch_size=512, device=DEV, seed=0)

print("[2/3] teacher transcripts (4 augmentation passes) …", flush=True)
t0 = time.perf_counter()
games = []
for s in range(4):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.45, seed=s)]
print(f"      {len(games)} games in {time.perf_counter() - t0:.0f}s", flush=True)

print("[3/3] SFT (cosine LR, save best-by-held-out) …", flush=True)
letter_ids = letter_id_tensor(tok, DEV)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
EPOCHS, BATCH, EVAL_EVERY = 70, 96, 3
steps_per_epoch = (len(games) + BATCH - 1) // BATCH
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch, eta_min=6e-5)
rng = Random(0)
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
        sched.step()
        losses.append(float(loss.detach()))
    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve_secrets)
        pt = " ".join(f"t{i + 1}={r:.2f}" for i, (r, _) in m["per_turn"].items())
        print(
            f"  epoch {epoch:>2}  loss={statistics.mean(losses):.3f}  lr={sched.get_last_lr()[0]:.1e}  "
            f"win={m['win']:.3f} avg_g={m['avg_g']:.2f}  valid={m['valid']:.3f}  [{pt}]",
            flush=True,
        )
        if m["win"] > best_win:
            best_win = m["win"]
            save_checkpoint("runs/sft_big.pt", model, opt, epoch, SFTConfig())

print(f"\n=== Phase A milestone: full held-out ({len(held)}) ===", flush=True)
# reload the best-by-held-out checkpoint for the milestone report
from wordle_slm.sft.train import load_checkpoint  # noqa: E402

best = WordleGenerator(BIG, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_big.pt", best)
final = evaluate(best, tuple(held))
pt = "  ".join(f"turn{i + 1}={r:.3f}" for i, (r, _) in final["per_turn"].items())
print(f"  held-out win rate : {final['win']:.3f}  ({int(round(final['win'] * len(held)))}/{len(held)})", flush=True)
print(f"  avg guesses/win   : {final['avg_g']:.2f}", flush=True)
print(f"  valid-word rate   : {final['valid']:.3f}", flush=True)
print(f"  per-turn valid    : {pt}", flush=True)
print("\n=== sample greedy held-out games ===", flush=True)
best.eval()
for s in held[:10]:
    print(render(play_game(best, tok, s, sample=False, device=DEV)), flush=True)
print("\n[PHASE A DONE]", flush=True)
