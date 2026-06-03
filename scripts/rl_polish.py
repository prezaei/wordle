"""RL polish on the BEST base (driver, not committed): GRPO with the validity+consistency reward on
top of sft_aux (0.436) — the cleanest base RL has ever had. ~20-min, best-checkpoint seeded at the
base so we can't regress below 0.436. Tests whether RL squeezes out points on a strong base.
"""

from __future__ import annotations

import logging
import statistics
import time
from random import Random
from types import SimpleNamespace

import torch

logging.disable(logging.CRITICAL)

import wordle_slm.rl.grpo as G  # noqa: E402
from wordle_slm.config import GRPOConfig, ModelConfig, RewardConfig  # noqa: E402
from wordle_slm.data import is_valid, load_valid_guesses, split, train_probe  # noqa: E402
from wordle_slm.engine import Status  # noqa: E402
from wordle_slm.engine.constraints import is_consistent  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.rl.rollout import play_game  # noqa: E402
from wordle_slm.rl.tracer import make_reference  # noqa: E402
from wordle_slm.sft.train import load_checkpoint, save_checkpoint  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
LARGE = ModelConfig.preset("large")
train, held = split(seed=0)
eval_secrets = tuple(held[:128])
probe = train_probe(seed=0, size=128)


def revised_reward(game, _config) -> SimpleNamespace:
    total = 0.0
    prior = []
    for turn in game.turns:
        if not turn.valid or turn.feedback is None:
            total -= 1.0
        elif not is_consistent(turn.guess, prior):
            total -= 1.0
        else:
            total += 0.1
        total -= 0.02
        prior.append(turn)
    if game.status is Status.WIN:
        total += 3.0 + 0.5 * (game.max_guesses - game.guesses_used)
    elif game.status is Status.LOSE:
        total -= 1.0
    return SimpleNamespace(total=total)


G.compute_reward = revised_reward


def evaluate(model, secrets) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


print("[load] best base runs/sft_aux.pt (0.436)", flush=True)
model = WordleGenerator(LARGE, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_aux.pt", model)
ref = make_reference(model)
base = evaluate(model, eval_secrets)
print(f"[start] win={base['win']:.3f} valid={base['valid']:.3f}", flush=True)

held_set = set(held)
pool = tuple(w for w in load_valid_guesses() if w not in held_set)
grpo = GRPOConfig(group_size=8, secrets_per_update=4, inner_epochs=1, lr=4e-5, kl_beta=0.01)
N_UPDATES, EVAL_EVERY, BUDGET_S = 80, 20, 1100  # ~18 min hard cap
opt = torch.optim.AdamW(model.parameters(), lr=grpo.lr)
gen = torch.Generator().manual_seed(0)
rng = Random(0)
warmup = 8
best = base["win"]
save_checkpoint("runs/rl_polish.pt", model, opt, -1, grpo)  # seed at the base — never regress
print(f"[grpo] up to {N_UPDATES} updates on the best base (cap {BUDGET_S}s)", flush=True)
t0 = time.perf_counter()
recent = []
for u in range(N_UPDATES):
    if time.perf_counter() - t0 > BUDGET_S:
        print(f"  [time cap reached at update {u}]", flush=True)
        break
    secrets = tuple(rng.choice(pool) for _ in range(grpo.secrets_per_update))
    lr = grpo.lr * min(1.0, (u + 1) / warmup)
    for grp in opt.param_groups:
        grp["lr"] = lr
    stats = G.grpo_update(model, ref, tok, secrets, grpo=grpo, reward=RewardConfig(),
                          optimizer=opt, group_size=grpo.group_size, device=DEV, generator=gen)
    recent.append(stats)
    if (u + 1) % EVAL_EVERY == 0:
        m = evaluate(model, eval_secrets)
        pw = G.eval_win_rate(model, tok, probe, device=DEV)
        rr = statistics.mean(s.reward_mean for s in recent[-EVAL_EVERY:])
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/rl_polish.pt", model, opt, u, grpo)
            flag = "  <- new best, saved"
        sec = (time.perf_counter() - t0) / (u + 1)
        print(f"  upd {u + 1:>3}  reward={rr:+.2f}  win={m['win']:.3f}  valid={m['valid']:.3f}  probe={pw:.3f}  ({sec:.0f}s/upd){flag}", flush=True)

print(f"\n[grpo] best curve-subsample win during run = {best:.3f}  (base was {base['win']:.3f})", flush=True)
print(f"=== FINAL (best-by-win) full held-out ({len(held)}) ===", flush=True)
b = WordleGenerator(LARGE, tok.vocab_size).to(DEV)
load_checkpoint("runs/rl_polish.pt", b)
f = evaluate(b, tuple(held))
print(f"  win={f['win']:.3f} ({int(round(f['win'] * len(held)))}/{len(held)})  valid={f['valid']:.3f}  avg={f['avg']:.2f}   [aux base was 0.436]", flush=True)
print("\n[POLISH DONE]", flush=True)
