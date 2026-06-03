"""Dict-in-the-loop RL (driver, not committed): the dictionary surfaces valid words during TRAINING
so RL can reinforce words free-gen can't yet produce. Inference stays free (no dict).

Per update: roll out via trie-constrained SAMPLING (valid, diverse boards). At each board, draw G
trie-sampled candidate words; reward each (consistent? greens? solves?); mean-center -> advantage;
then push the model's FREE-GENERATION log-prob toward the high-advantage words (advantage-weighted,
KL-anchored to the base). Over updates, free-gen learns to produce the valid+consistent+winning
words the trie surfaced. Eval is free-gen greedy. Best-by-win -> runs/rl_dict.pt.
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
letter_ids = letter_id_tensor(tok, DEV)
LETTERS = "abcdefghijklmnopqrstuvwxyz"
LIDX = {int(letter_ids[i]): i for i in range(26)}
TRIE: dict = {}
for w in load_valid_guesses():
    node = TRIE
    for ch in w:
        node = node.setdefault(ch, {})


@torch.no_grad()
def trie_sample(model, prompt_ids: list[int], generator) -> tuple[torch.Tensor, str]:
    """Sample a 5-letter word from the model, constrained to the dictionary (trie). Returns tokens, word."""
    node = TRIE
    ids = list(prompt_ids)
    toks = []
    for _ in range(5):
        logits = model.forward(torch.tensor([ids], device=DEV))[0, -1, letter_ids]  # [26]
        mask = torch.full((26,), float("-inf"))
        for ch in node:
            mask[LETTERS.index(ch)] = 0.0
        probs = torch.softmax(logits.cpu() + mask, dim=-1)
        j = int(torch.multinomial(probs, 1, generator=generator))
        node = node[LETTERS[j]]
        toks.append(int(letter_ids[j]))
        ids.append(toks[-1])
    return torch.tensor(toks, device=DEV), decode_word(toks, tok)


def free_logp(model, prompt_ids: list[int], rows: torch.Tensor) -> torch.Tensor:
    """FREE-generation per-position log-prob [G,5] of each word (no trie mask) — grad-enabled."""
    p = len(prompt_ids)
    prompt = torch.tensor(prompt_ids, device=DEV).unsqueeze(0).expand(rows.size(0), -1)
    logits = model.forward(torch.cat([prompt, rows], dim=1))[:, p - 1 : p + 4, :]
    logp = torch.log_softmax(logits[:, :, letter_ids], dim=-1)
    idx = torch.tensor([[LIDX[int(t)] for t in row] for row in rows], device=DEV)
    return logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # [G,5]


def word_reward(word: str, history, secret: str) -> float:
    if not is_consistent(word, history):
        return -0.5  # valid (trie) but violates a clue / repeats
    greens = sum(c is Color.GREEN for c in score(word, secret))
    return 0.1 + 0.15 * greens + (1.0 if word == secret else 0.0)


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
pool = tuple(w for w in load_valid_guesses() if w not in held_set)
G, S_SECRETS, N_UPDATES, EVAL_EVERY = 6, 5, 70, 12
LR, KL_BETA = 3e-5, 0.02
opt = torch.optim.AdamW(model.parameters(), lr=LR)
gen = torch.Generator().manual_seed(0)  # CPU generator for multinomial
rng = Random(0)
best = base["win"]
save_checkpoint("runs/rl_dict.pt", model, opt, -1, GRPOConfig())
print(f"[dict-RL] {N_UPDATES} updates, G={G} trie-samples/board, {S_SECRETS} rollouts/update, lr={LR}", flush=True)
t0 = time.perf_counter()
for u in range(N_UPDATES):
    boards = []  # (prompt, history, rows[G,5], words, secret)
    with torch.no_grad():
        for _ in range(S_SECRETS):
            secret = rng.choice(pool)
            game = Game(secret)
            while game.status is Status.ONGOING:
                prompt = build_prompt(game.turns, tok)
                rows, words = [], []
                for _ in range(G):
                    t, w = trie_sample(model, prompt, gen)
                    rows.append(t)
                    words.append(w)
                boards.append((prompt, list(game.turns), torch.stack(rows), words, secret))
                game.guess(words[0])  # advance with the first trie-sample
    opt.zero_grad()
    total_loss = torch.zeros((), device=DEV)
    n = 0
    solved_frac = []
    for prompt, history, rows, words, secret in boards:
        rewards = torch.tensor([word_reward(w, history, secret) for w in words], device=DEV)
        solved_frac.append(float((rewards >= 1.0).float().mean()))
        adv = (rewards - rewards.mean()).clamp(-1.5, 1.5)
        if bool((adv.abs() < 1e-9).all()):
            continue
        with torch.no_grad():
            ref_lp = free_logp(ref, prompt, rows)  # [G,5]
        cur = free_logp(model, prompt, rows)  # [G,5] grad
        surr = -(adv * cur.sum(-1))  # advantage-weighted push on free-gen logp
        kl = (torch.exp(ref_lp - cur) - (ref_lp - cur) - 1.0).sum(-1)
        total_loss = total_loss + (surr + KL_BETA * kl).sum()
        n += G
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
            save_checkpoint("runs/rl_dict.pt", model, opt, u, GRPOConfig())
            flag = "  <- best"
        sec = (time.perf_counter() - t0) / (u + 1)
        print(f"  upd {u + 1:>3}  win={m['win']:.3f}  valid={m['valid']:.3f}  consistent={m['consistent']:.3f}  probe={pw:.3f}  solved/board={statistics.mean(solved_frac):.2f}  ({sec:.0f}s/upd){flag}", flush=True)

print(f"\n=== FINAL (best-by-win, FREE-GEN) full held-out ({len(held)}) ===", flush=True)
b = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/rl_dict.pt", b)
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
print("\n[DICTRL DONE]", flush=True)
