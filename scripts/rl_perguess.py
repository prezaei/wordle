"""Per-GUESS GRPO (driver, not committed): clean per-guess credit for validity + consistency.

Trajectory RL gave one advantage to a whole game (muddy credit). Here the "episode" is a SINGLE
guess: for each board the model reaches, sample G candidate guesses, score EACH on its own merits,
mean-center -> each guess gets its own advantage. Directly optimizes "make this one guess valid +
consistent + informative", which is exactly the skill we want and the cleanest signal for it.

per-guess reward (board known, so secret known): invalid -> -1 ; valid but inconsistent -> -1 ;
valid+consistent -> +0.2 + 0.2*greens + (2.0 if it solves). Best-by-win -> runs/rl_pg.pt.
"""

from __future__ import annotations

import logging
import statistics
import time
from random import Random

import torch

logging.disable(logging.CRITICAL)

from wordle_slm.config import GRPOConfig, ModelConfig  # noqa: E402
from wordle_slm.data import is_valid, load_valid_guesses, split, train_probe  # noqa: E402
from wordle_slm.engine import Color, Status  # noqa: E402
from wordle_slm.engine.constraints import is_consistent  # noqa: E402
from wordle_slm.engine.scoring import score  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.model.serialization import build_prompt, decode_word  # noqa: E402
from wordle_slm.rl.grpo import eval_win_rate  # noqa: E402
from wordle_slm.rl.rollout import letter_id_tensor, play_game  # noqa: E402
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
letter_ids = letter_id_tensor(tok, DEV)
LIDX = {int(letter_ids[i]): i for i in range(26)}  # letter token id -> 0..25


def guess_reward(word: str, history, secret: str) -> float:
    if not is_valid(word):
        return -1.0
    if not is_consistent(word, history):
        return -1.0
    greens = sum(c is Color.GREEN for c in score(word, secret))
    return 0.2 + 0.2 * greens + (2.0 if word == secret else 0.0)


def per_pos_logp(model, prompt_ids: list[int], guess_token_rows: torch.Tensor) -> torch.Tensor:
    """Per-position log-prob [G,5] of each sampled guess given the board prompt (grad-enabled)."""
    p = len(prompt_ids)
    prompt = torch.tensor(prompt_ids, device=DEV).unsqueeze(0).expand(guess_token_rows.size(0), -1)
    seq = torch.cat([prompt, guess_token_rows], dim=1)  # [G, P+5]
    logits = model.forward(seq)[:, p - 1 : p + 4, :]  # predict the 5 guess letters
    logp = torch.log_softmax(logits[:, :, letter_ids], dim=-1)  # [G,5,26]
    idx = torch.tensor([[LIDX[int(t)] for t in row] for row in guess_token_rows], device=DEV)
    return logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # [G,5]


def evaluate(model, secrets) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [g for g in games if g.won]
    valid = total = consistent = valid_count = 0
    for g in games:
        prior = []
        for turn in g.turns:
            total += 1
            v = is_valid(turn.guess)
            valid += int(v)
            if v:
                valid_count += 1
                consistent += int(is_consistent(turn.guess, prior))
            prior.append(turn)
    return {"win": len(wins) / len(games), "valid": valid / total,
            "consistent": consistent / valid_count if valid_count else 0.0,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


print("[load] base runs/sft_distill.pt", flush=True)
model = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_distill.pt", model)
ref = make_reference(model)
base = evaluate(model, eval_secrets)
print(f"[start] win={base['win']:.3f} valid={base['valid']:.3f} consistent={base['consistent']:.3f}", flush=True)

held_set = set(held)
pool = tuple(w for w in load_valid_guesses() if w not in held_set)
G, S_SECRETS, N_UPDATES, EVAL_EVERY = 8, 5, 70, 12
LR, KL_BETA, CLIP = 5e-5, 0.01, 0.2
opt = torch.optim.AdamW(model.parameters(), lr=LR)
gen = torch.Generator().manual_seed(0)
rng = Random(0)
best = base["win"]
save_checkpoint("runs/rl_pg.pt", model, opt, -1, GRPOConfig())  # seed with the base
print(f"[per-guess grpo] {N_UPDATES} updates, G={G} guesses/board, {S_SECRETS} rollouts/update", flush=True)
t0 = time.perf_counter()
for u in range(N_UPDATES):
    # collect on-policy boards from sampled rollouts
    boards = []  # (prompt_ids, history_turns, secret)
    model.eval()
    with torch.no_grad():
        for _ in range(S_SECRETS):
            secret = rng.choice(pool)
            g = play_game(model, tok, secret, sample=True, generator=gen, device=DEV)
            for k in range(len(g.turns)):
                boards.append((build_prompt(g.turns[:k], tok), g.turns[:k], secret))
    # per-board: sample G guesses, score, mean-center, accumulate loss
    model.train()
    opt.zero_grad()
    total_loss = torch.zeros((), device=DEV)
    n_kept = 0
    kl_acc = 0.0
    for prompt_ids, history, secret in boards:
        with torch.no_grad():
            rows, words = [], []
            pt = torch.tensor(prompt_ids, device=DEV)
            for _ in range(G):
                gtoks = model.generate(pt, letter_ids, sample=True, generator=gen)
                rows.append(gtoks)
                words.append(decode_word(gtoks.tolist(), tok))
            rows_t = torch.stack(rows)  # [G,5]
            rewards = torch.tensor([guess_reward(w, history, secret) for w in words], device=DEV)
            adv = rewards - rewards.mean()
            if bool((adv.abs() < 1e-9).all()):
                continue  # zero-variance board, no signal
            old_lp = per_pos_logp(model, prompt_ids, rows_t).sum(-1)  # [G]
            ref_lp = per_pos_logp(ref, prompt_ids, rows_t)  # [G,5]
        cur = per_pos_logp(model, prompt_ids, rows_t)  # [G,5] grad
        cur_sum = cur.sum(-1)
        ratio = torch.exp(cur_sum - old_lp)
        surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv)
        kl = (torch.exp(ref_lp - cur) - (ref_lp - cur) - 1.0).sum(-1)  # [G] k3
        total_loss = total_loss + (-surr + KL_BETA * kl).sum()
        n_kept += G
        kl_acc += float(kl.sum().detach())
    if n_kept == 0:
        continue
    (total_loss / n_kept).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (u + 1) % EVAL_EVERY == 0:
        m = evaluate(model, eval_secrets)
        pw = eval_win_rate(model, tok, probe, device=DEV)
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/rl_pg.pt", model, opt, u, GRPOConfig())
            flag = "  <- best"
        sec = (time.perf_counter() - t0) / (u + 1)
        print(f"  upd {u + 1:>3}  win={m['win']:.3f}  valid={m['valid']:.3f}  consistent={m['consistent']:.3f}  probe={pw:.3f}  kl/g={kl_acc / max(1, n_kept):.3f}  ({sec:.0f}s/upd){flag}", flush=True)

print(f"\n=== FINAL (best-by-win) full held-out ({len(held)}) ===", flush=True)
b = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/rl_pg.pt", b)
f = evaluate(b, tuple(held))
print(f"  win={f['win']:.3f} ({int(round(f['win'] * len(held)))}/{len(held)})  valid={f['valid']:.3f}  consistent={f['consistent']:.3f}  avg={f['avg']:.2f}", flush=True)
print(f"  [base: win {base['win']:.3f}, valid {base['valid']:.3f}, consistent {base['consistent']:.3f}]", flush=True)
print("\n[PG DONE]", flush=True)
