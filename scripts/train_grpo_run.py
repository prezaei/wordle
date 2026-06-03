"""Phase B: GRPO on the 4.8M SFT base (driver script, not committed).

Loads runs/sft_big.pt, freezes a copy as π_ref, runs real GRPO over the full train set with a
config strong enough to actually move greedy play (lr 8e-5, loose KL). Streams the full diagnostic
curve (reward / win / valid / gen-gap / KL), keeps the best-by-held-out model in runs/best.pt.
"""

from __future__ import annotations

import logging
import statistics
from random import Random

import torch

logging.disable(logging.CRITICAL)

from wordle_slm.config import CurriculumConfig, GRPOConfig, ModelConfig, RewardConfig  # noqa: E402
from wordle_slm.data import is_valid, load_valid_guesses, split, train_probe  # noqa: E402
from wordle_slm.engine import Color  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.rl.curriculum import Curriculum  # noqa: E402
from wordle_slm.rl.grpo import eval_win_rate, grpo_update  # noqa: E402
from wordle_slm.rl.rollout import play_game  # noqa: E402
from wordle_slm.rl.tracer import make_reference  # noqa: E402
from wordle_slm.sft.train import load_checkpoint, save_checkpoint  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
BIG = ModelConfig(d_model=256, n_layers=6, n_heads=8, d_ff=1024)
train, held = split(seed=0)
eval_secrets = tuple(held[:128])
probe = train_probe(seed=0, size=128)
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def evaluate(model: WordleGenerator, secrets: tuple[str, ...]) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    wins = [g for g in games if g.won]
    v = t = 0
    for g in games:
        for turn in g.turns:
            v += int(is_valid(turn.guess))
            t += 1
    return {"win": len(wins) / len(games), "valid": v / t if t else 0.0}


def render(g) -> str:
    lines = [f"  secret={g.secret}  [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        fb = "❌ not a word" if t.feedback is None else "".join(SQ[c] for c in t.feedback)
        lines.append(f"      {t.guess}  {fb}")
    return "\n".join(lines)


print("[load] SFT base runs/sft_big.pt → policy + frozen π_ref", flush=True)
model = WordleGenerator(BIG, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_big.pt", model)
ref = make_reference(model)
base = evaluate(model, eval_secrets)
print(f"[start] held-out win={base['win']:.3f} valid={base['valid']:.3f} (greedy, before GRPO)", flush=True)

# Generalization test: draw RL secrets from the FULL valid list (8x more than the 1,852 answers,
# so a tiny model can't memorize them in 200 updates), with the held-out answers EXCLUDED so the
# gap measurement stays honest, and hard-word replay OFF (replay re-feeds specific secrets).
held_set = set(held)
pool = tuple(w for w in load_valid_guesses() if w not in held_set)
curriculum = Curriculum(pool, CurriculumConfig(tiers=(None,), replay_prob=0.0))
grpo = GRPOConfig(group_size=16, secrets_per_update=8, inner_epochs=1, lr=8e-5, kl_beta=0.005)
N_UPDATES, EVAL_EVERY = 200, 20
print(f"[pool] {len(pool)} RL secrets (full valid list minus {len(held)} held-out), replay OFF", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=grpo.lr)
generator = torch.Generator().manual_seed(0)
rng = Random(0)
warmup = max(1, int(N_UPDATES * grpo.warmup_ratio))
best = base["win"]  # only overwrite best.pt if GRPO beats the SFT base
save_checkpoint("runs/best.pt", model, optimizer, -1, grpo)  # seed best.pt with the SFT base
print(f"[grpo] {N_UPDATES} updates, G={grpo.group_size}, lr={grpo.lr}, kl_beta={grpo.kl_beta}", flush=True)
recent: list = []
for update in range(N_UPDATES):
    secrets = tuple(curriculum.sample(rng) for _ in range(grpo.secrets_per_update))
    lr = grpo.lr * min(1.0, (update + 1) / warmup)
    for grp in optimizer.param_groups:
        grp["lr"] = lr
    stats = grpo_update(
        model, ref, tok, secrets, grpo=grpo, reward=RewardConfig(),
        optimizer=optimizer, group_size=grpo.group_size, device=DEV, generator=generator,
    )
    recent.append(stats)  # replay OFF for this generalization test
    if (update + 1) % EVAL_EVERY == 0:
        m = evaluate(model, eval_secrets)
        probe_win = eval_win_rate(model, tok, probe, device=DEV)
        rr = statistics.mean(s.reward_mean for s in recent[-EVAL_EVERY:])
        kf = statistics.mean(s.kept_secrets / s.n_secrets for s in recent[-EVAL_EVERY:])
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/best.pt", model, optimizer, update, grpo)
            flag = "  <- new best, saved"
        print(
            f"  upd {update + 1:>3}  reward={rr:+.3f}  win={m['win']:.3f}  valid={m['valid']:.3f}  "
            f"probe={probe_win:.3f}  gap={probe_win - m['win']:+.3f}  kl={stats.kl:.4f}  kept={kf:.2f}{flag}",
            flush=True,
        )

print(f"\n[grpo] best held-out win during run = {best:.3f}  (SFT base was {base['win']:.3f})", flush=True)
print(f"\n=== FINAL (best.pt) full held-out ({len(held)}) ===", flush=True)
final_model = WordleGenerator(BIG, tok.vocab_size).to(DEV)
load_checkpoint("runs/best.pt", final_model)
f_all = evaluate(final_model, tuple(held))
print(f"  held-out win rate : {f_all['win']:.3f}  ({int(round(f_all['win'] * len(held)))}/{len(held)})", flush=True)
print(f"  valid-word rate   : {f_all['valid']:.3f}", flush=True)
print("\n=== sample greedy held-out games (final play) ===", flush=True)
final_model.eval()
for s in held[:12]:
    print(render(play_game(final_model, tok, s, sample=False, device=DEV)), flush=True)
print("\n[PHASE B DONE]", flush=True)
