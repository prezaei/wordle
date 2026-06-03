"""RL to LEARN validity + consistency (+ no-repeat) into the weights (driver, not committed).

Revised reward (the key change from the runs that overfit on win): validity + consistency dominate.
  invalid guess           -> -1.0
  valid but inconsistent  -> -1.0   (subsumes no-repeat: a repeated wrong guess is inconsistent)
  valid + consistent      -> +0.1
  win +3.0 + 0.5*(6-t),  lose -1.0,  -0.02/step
Injected via monkeypatch so the tested GRPO machinery is reused unchanged. Starts from the strong
SFT base; diverse secrets (full valid minus held-out) so it can't memorize. Measures held-out
win / valid-rate / consistency-rate / repeat-rate (not just win). Best-by-win -> runs/rl_cons.pt.
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
from wordle_slm.engine import Color, Status  # noqa: E402
from wordle_slm.engine.constraints import is_consistent  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.rl.rollout import play_game  # noqa: E402
from wordle_slm.rl.tracer import make_reference  # noqa: E402
from wordle_slm.sft.train import load_checkpoint, save_checkpoint  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
XL = ModelConfig(d_model=512, n_layers=8, n_heads=8, d_ff=2048)
train, held = split(seed=0)
eval_secrets = tuple(held[:128])
probe = train_probe(seed=0, size=128)
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def revised_reward(game, _config) -> SimpleNamespace:
    total = 0.0
    prior = []
    for turn in game.turns:
        if not turn.valid or turn.feedback is None:
            total -= 1.0  # invalid word
        elif not is_consistent(turn.guess, prior):
            total -= 1.0  # violates a known clue / repeats a dead guess
        else:
            total += 0.1  # valid + consistent
        total -= 0.02  # step cost
        prior.append(turn)
    if game.status is Status.WIN:
        total += 3.0 + 0.5 * (game.max_guesses - game.guesses_used)
    elif game.status is Status.LOSE:
        total -= 1.0
    return SimpleNamespace(total=total)


G.compute_reward = revised_reward  # inject the revised reward into the tested GRPO update


def evaluate(model, secrets) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [g for g in games if g.won]
    valid = total = consistent = valid_count = repeat = 0
    for g in games:
        prior = []
        seen = set()
        for turn in g.turns:
            total += 1
            v = is_valid(turn.guess)
            valid += int(v)
            if v:
                valid_count += 1
                consistent += int(is_consistent(turn.guess, prior))
            repeat += int(turn.guess in seen)
            seen.add(turn.guess)
            prior.append(turn)
    return {
        "win": len(wins) / len(games),
        "valid": valid / total,
        "consistent": consistent / valid_count if valid_count else 0.0,  # of valid guesses
        "repeat": repeat / total,
        "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan"),
    }


print("[load] base runs/sft_distill.pt -> policy + frozen pi_ref", flush=True)
model = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_distill.pt", model)
ref = make_reference(model)
base = evaluate(model, eval_secrets)
print(f"[start] win={base['win']:.3f} valid={base['valid']:.3f} consistent={base['consistent']:.3f} repeat={base['repeat']:.3f}", flush=True)

held_set = set(held)
pool = tuple(w for w in load_valid_guesses() if w not in held_set)
grpo = GRPOConfig(group_size=8, secrets_per_update=4, inner_epochs=1, lr=5e-5, kl_beta=0.01)
N_UPDATES, EVAL_EVERY = 80, 12
opt = torch.optim.AdamW(model.parameters(), lr=grpo.lr)
gen = torch.Generator().manual_seed(0)
rng = Random(0)
warmup = 8
best = base["win"]
save_checkpoint("runs/rl_cons.pt", model, opt, -1, grpo)  # seed with the base
print(f"[grpo] {N_UPDATES} updates G={grpo.group_size} x{grpo.secrets_per_update} secrets, lr={grpo.lr} (revised reward)", flush=True)
t0 = time.perf_counter()
recent = []
for u in range(N_UPDATES):
    secrets = tuple(rng.choice(pool) for _ in range(grpo.secrets_per_update))
    lr = grpo.lr * min(1.0, (u + 1) / warmup)
    for grp in opt.param_groups:
        grp["lr"] = lr
    stats = G.grpo_update(model, ref, tok, secrets, grpo=grpo, reward=RewardConfig(), optimizer=opt, group_size=grpo.group_size, device=DEV, generator=gen)
    recent.append(stats)
    if (u + 1) % EVAL_EVERY == 0:
        m = evaluate(model, eval_secrets)
        pw = G.eval_win_rate(model, tok, probe, device=DEV)
        rr = statistics.mean(s.reward_mean for s in recent[-EVAL_EVERY:])
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/rl_cons.pt", model, opt, u, grpo)
            flag = "  <- best"
        sec = (time.perf_counter() - t0) / (u + 1)
        print(f"  upd {u + 1:>3}  reward={rr:+.2f}  win={m['win']:.3f}  valid={m['valid']:.3f}  consistent={m['consistent']:.3f}  repeat={m['repeat']:.3f}  probe={pw:.3f}  kl={stats.kl:.3f}  ({sec:.0f}s/upd){flag}", flush=True)

print(f"\n=== FINAL (best-by-win) full held-out ({len(held)}) ===", flush=True)
b = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/rl_cons.pt", b)
f = evaluate(b, tuple(held))
print(f"  win={f['win']:.3f} ({int(round(f['win'] * len(held)))}/{len(held)})  valid={f['valid']:.3f}  consistent={f['consistent']:.3f}  repeat={f['repeat']:.3f}  avg={f['avg']:.2f}", flush=True)
print(f"  [base was: win {base['win']:.3f}, valid {base['valid']:.3f}, consistent {base['consistent']:.3f}, repeat {base['repeat']:.3f}]", flush=True)
print("\n=== sample greedy games ===", flush=True)
b.eval()
for s in held[:10]:
    g = play_game(b, tok, s, sample=False, device=DEV)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[RLCONS DONE]", flush=True)
