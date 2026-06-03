"""Consistency-constrained RL (driver, not committed): the FULL constraint in the training loop.

Dict-only sampling surfaced the answer ~4% of boards (model mass favors common valid words, not the
rare answer). Here the behavior policy samples from the still-CONSISTENT candidate set (shrinks
toward the answer each turn), so the answer is surfaced far more -> strong winning signal. Each
candidate is rewarded (greens + solve), mean-centered, and the model's FREE-GEN log-prob is pushed
toward the high-advantage ones (advantage-weighted, KL-anchored). The consistency filter is used
ONLY in training; eval/inference is free-gen, unaided. (= distilling the v3 candidate teacher into
the free-gen weights.) Best-by-win -> runs/rl_constr.pt.
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
from wordle_slm.engine import Color, Game, Status  # noqa: E402
from wordle_slm.engine.constraints import filter_consistent, is_consistent  # noqa: E402
from wordle_slm.engine.scoring import score  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.model.serialization import build_prompt  # noqa: E402
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
letter_ids = letter_id_tensor(tok, DEV)
LIDX = {int(letter_ids[i]): i for i in range(26)}
VALID = load_valid_guesses()
CAP = 48  # cap candidates scored per board (subsample when the consistent set is large)


def free_logp(model, prompt_ids: list[int], rows: torch.Tensor) -> torch.Tensor:
    """FREE-gen per-position log-prob [N,5] of each candidate word (no constraint) — grad if model is."""
    p = len(prompt_ids)
    prompt = torch.tensor(prompt_ids, device=DEV).unsqueeze(0).expand(rows.size(0), -1)
    logits = model.forward(torch.cat([prompt, rows], dim=1))[:, p - 1 : p + 4, :]
    logp = torch.log_softmax(logits[:, :, letter_ids], dim=-1)
    idx = torch.tensor([[LIDX[int(t)] for t in row] for row in rows], device=DEV)
    return logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def rows_of(words: list[str]) -> torch.Tensor:
    return torch.tensor([tok.encode_letters(w) for w in words], device=DEV)


def evaluate(model, secrets) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
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


print("[load] base runs/sft_distill.pt", flush=True)
model = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_distill.pt", model)
model.eval()
ref = make_reference(model)
base = evaluate(model, eval_secrets)
print(f"[start] win={base['win']:.3f} valid={base['valid']:.3f} consistent={base['consistent']:.3f}", flush=True)

held_set = set(held)
secret_pool = tuple(w for w in VALID if w not in held_set)
G, S_SECRETS, N_UPDATES, EVAL_EVERY = 8, 5, 70, 12
LR, KL_BETA = 3e-5, 0.02
opt = torch.optim.AdamW(model.parameters(), lr=LR)
gen = torch.Generator().manual_seed(0)
rng = Random(0)
best = base["win"]
save_checkpoint("runs/rl_constr.pt", model, opt, -1, GRPOConfig())
print(f"[constrained-RL] {N_UPDATES} updates, consistent-set sampling, G={G}/board, {S_SECRETS} rollouts/update", flush=True)
t0 = time.perf_counter()
for u in range(N_UPDATES):
    boards = []  # (prompt, rows[G,5], words, secret, old_lp[G])
    with torch.no_grad():
        for _ in range(S_SECRETS):
            secret = rng.choice(secret_pool)
            game = Game(secret)
            cand = VALID  # consistent set, filtered incrementally
            while game.status is Status.ONGOING:
                prompt = build_prompt(game.turns, tok)
                if not game.turns:  # natural opener (free-gen greedy); no training data turn 0
                    word = "".join(tok.id_to_token(int(i)) for i in model.generate(torch.tensor(prompt, device=DEV), letter_ids, sample=False))
                    game.guess(word)
                    cand = filter_consistent(cand, game.turns[-1])
                    continue
                pool_list = list(cand) if cand else [secret]
                sub = rng.sample(pool_list, CAP) if len(pool_list) > CAP else pool_list
                if secret not in sub and secret in pool_list:
                    sub = sub[:-1] + [secret]  # keep the answer reachable when it's still consistent
                sub_rows = rows_of(sub)
                lp = free_logp(model, prompt, sub_rows).sum(-1)  # [|sub|]
                probs = torch.softmax(lp, dim=-1).cpu()
                k = min(G, len(sub))
                picks = torch.multinomial(probs, k, replacement=False, generator=gen).tolist()
                words = [sub[i] for i in picks]
                boards.append((prompt, rows_of(words), words, secret, lp[picks].detach()))
                game.guess(words[0])  # advance with the model's top consistent pick
                cand = filter_consistent(cand, game.turns[-1])
    opt.zero_grad()
    total_loss = torch.zeros((), device=DEV)
    n = 0
    solved = []
    for prompt, rows, words, secret, old_lp in boards:
        greens = torch.tensor([sum(c is Color.GREEN for c in score(w, secret)) for w in words], device=DEV)
        solve = torch.tensor([1.0 if w == secret else 0.0 for w in words], device=DEV)
        rewards = 0.1 + 0.15 * greens + solve
        solved.append(float(solve.mean()))
        adv = (rewards - rewards.mean()).clamp(-1.5, 1.5)
        if bool((adv.abs() < 1e-9).all()):
            continue
        with torch.no_grad():
            ref_lp = free_logp(ref, prompt, rows)
        cur = free_logp(model, prompt, rows)
        surr = -(adv * cur.sum(-1))
        kl = (torch.exp(ref_lp - cur) - (ref_lp - cur) - 1.0).sum(-1)
        total_loss = total_loss + (surr + KL_BETA * kl).sum()
        n += len(words)
    if n == 0:
        continue
    (total_loss / n).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (u + 1) % EVAL_EVERY == 0:
        m = evaluate(model, eval_secrets)
        pw = eval_win_rate(model, tok, probe, device=DEV)
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/rl_constr.pt", model, opt, u, GRPOConfig())
            flag = "  <- best"
        sec = (time.perf_counter() - t0) / (u + 1)
        print(f"  upd {u + 1:>3}  win={m['win']:.3f}  valid={m['valid']:.3f}  consistent={m['consistent']:.3f}  probe={pw:.3f}  solved/board={statistics.mean(solved):.2f}  ({sec:.0f}s/upd){flag}", flush=True)

print(f"\n=== FINAL (best-by-win, FREE-GEN) full held-out ({len(held)}) ===", flush=True)
b = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/rl_constr.pt", b)
f = evaluate(b, tuple(held))
print(f"  win={f['win']:.3f} ({int(round(f['win'] * len(held)))}/{len(held)})  valid={f['valid']:.3f}  consistent={f['consistent']:.3f}  avg={f['avg']:.2f}", flush=True)
print(f"  [base: win {base['win']:.3f}, valid {base['valid']:.3f}, consistent {base['consistent']:.3f}]", flush=True)
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
print("\n=== sample free-gen games ===", flush=True)
b.eval()
for s in held[:10]:
    g = play_game(b, tok, s, sample=False, device=DEV)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[CONSTR DONE]", flush=True)
