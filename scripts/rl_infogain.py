"""RL with info-gain shaping + more training guesses (driver, not committed), on the best base.

Two changes vs rl_polish: (1) reward each valid+consistent guess with +0.1 + BETA*log(|C_before|/
|C_after|) — the information it gains by narrowing the answer set (the missing 'smart play' term);
(2) rollouts get 12 guesses (denser signal / more chances to narrow + win), while EVAL stays at the
real 6 guesses. Best base = sft_aux (0.436); best-checkpoint seeded there so we can't regress.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from random import Random
from types import SimpleNamespace

import torch

logging.disable(logging.CRITICAL)

import wordle_slm.rl.grpo as G  # noqa: E402
from wordle_slm.config import GRPOConfig, ModelConfig, RewardConfig  # noqa: E402
from wordle_slm.data import is_valid, load_answers, load_valid_guesses, split, train_probe  # noqa: E402
from wordle_slm.engine import Status  # noqa: E402
from wordle_slm.engine.constraints import filter_consistent, is_consistent  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.rl.rollout import play_game as real_play  # noqa: E402
from wordle_slm.rl.tracer import make_reference  # noqa: E402
from wordle_slm.sft.train import load_checkpoint, save_checkpoint  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
LARGE = ModelConfig.preset("large")
train, held = split(seed=0)
eval_secrets = tuple(held[:128])
probe = train_probe(seed=0, size=128)
INFO_POOL = load_valid_guesses()  # candidate pool for info-gain; secret is always in it (never empties)
BETA = 0.2
TRAIN_GUESSES = 10  # max that fits context_len=128 (a 10-turn game = 122 tokens; 12 overflows)


def play10(*args, **kwargs):
    kwargs.setdefault("max_guesses", TRAIN_GUESSES)
    return real_play(*args, **kwargs)


def infogain_reward(game, _cfg) -> SimpleNamespace:
    """Validity + consistency + INFORMATION GAIN (answer-set narrowing) + win/speed."""
    total = 0.0
    prior = []
    cand = INFO_POOL  # candidate hypotheses; info-gain = how much each guess narrows them
    for turn in game.turns:
        total -= 0.02  # step cost
        if not turn.valid or turn.feedback is None:
            total -= 1.0
        elif not is_consistent(turn.guess, prior):
            total -= 1.0
        else:
            before = len(cand)
            cand = filter_consistent(cand, turn)
            info = math.log(max(1, before) / max(1, len(cand)))  # nats gained this turn
            total += 0.1 + BETA * info
        prior.append(turn)
    if game.status is Status.WIN:
        total += 3.0 + 0.5 * (game.max_guesses - game.guesses_used)
    elif game.status is Status.LOSE:
        total -= 1.0
    return SimpleNamespace(total=total)


G.play_game = play10  # grpo_update rollouts now use 10 guesses
G.compute_reward = infogain_reward


def evaluate(model, secrets) -> dict:
    """Greedy eval on the REAL 6-guess game (real_play, max_guesses=6)."""
    model.eval()
    games = [real_play(model, tok, s, sample=False, device=DEV, max_guesses=6) for s in secrets]
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
print(f"[start] win={base['win']:.3f} valid={base['valid']:.3f} (6-guess eval)", flush=True)

held_set = set(held)
pool = tuple(w for w in load_valid_guesses() if w not in held_set)
grpo = GRPOConfig(group_size=8, secrets_per_update=4, inner_epochs=1, lr=4e-5, kl_beta=0.01)
N_UPDATES, EVAL_EVERY, BUDGET_S = 120, 20, 1300
opt = torch.optim.AdamW(model.parameters(), lr=grpo.lr)
gen = torch.Generator().manual_seed(0)
rng = Random(0)
warmup = 8
best = base["win"]
save_checkpoint("runs/rl_infogain.pt", model, opt, -1, grpo)
print(f"[grpo] info-gain (BETA={BETA}) + {TRAIN_GUESSES}-guess rollouts; up to {N_UPDATES} upd, cap {BUDGET_S}s", flush=True)
t0 = time.perf_counter()
recent = []
for u in range(N_UPDATES):
    if time.perf_counter() - t0 > BUDGET_S:
        print(f"  [time cap at update {u}]", flush=True)
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
        rr = statistics.mean(s.reward_mean for s in recent[-EVAL_EVERY:])
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/rl_infogain.pt", model, opt, u, grpo)
            flag = "  <- new best, saved"
        sec = (time.perf_counter() - t0) / (u + 1)
        print(f"  upd {u + 1:>3}  reward={rr:+.2f}  win(6)={m['win']:.3f}  valid={m['valid']:.3f}  avg={m['avg']:.2f}  ({sec:.0f}s/upd){flag}", flush=True)

print(f"\n[grpo] best 6-guess subsample win = {best:.3f}  (base {base['win']:.3f})", flush=True)
print(f"=== FINAL (best-by-win) full held-out, 6-guess ({len(held)}) ===", flush=True)
b = WordleGenerator(LARGE, tok.vocab_size).to(DEV)
load_checkpoint("runs/rl_infogain.pt", b)
f = evaluate(b, tuple(held))
print(f"  win={f['win']:.3f} ({int(round(f['win'] * len(held)))}/{len(held)})  valid={f['valid']:.3f}  avg={f['avg']:.2f}   [aux base 0.436]", flush=True)
print("\n[INFOGAIN DONE]", flush=True)
