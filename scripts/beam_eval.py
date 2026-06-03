"""Decoding experiment (driver script, not committed): how much win rate is greedy leaving behind?

Compares, on the SAME held-out subsample with the existing best model (runs/sft_xl.pt):
  - greedy        : current eval (argmax letter-by-letter).
  - beam          : beam search over the model's OWN distribution (no dictionary) — pure decoding.
  - beam+dict     : beam search constrained to real words via a trie — the strategy ceiling
                    (guarantees valid guesses; model still chooses WHICH word).
"""

from __future__ import annotations

import logging
import statistics

import torch

logging.disable(logging.CRITICAL)

from wordle_slm.config import ModelConfig  # noqa: E402
from wordle_slm.data import is_valid, load_valid_guesses, split  # noqa: E402
from wordle_slm.engine import Color, Game, Status  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.model.serialization import build_prompt, decode_word  # noqa: E402
from wordle_slm.rl.rollout import letter_id_tensor  # noqa: E402
from wordle_slm.sft.train import load_checkpoint  # noqa: E402

DEV = "mps"
tok = Tokenizer()
XL = ModelConfig(d_model=512, n_layers=8, n_heads=8, d_ff=2048)
train, held = split(seed=0)
EVAL = tuple(held[:250])
WIDTH = 12
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}

# trie of valid words (length-5) over letters a-z
TrieNode = dict
TRIE: TrieNode = {}
for w in load_valid_guesses():
    node = TRIE
    for ch in w:
        node = node.setdefault(ch, {})
LETTERS = "abcdefghijklmnopqrstuvwxyz"


@torch.no_grad()
def beam_decode(model: WordleGenerator, prompt: list[int], letter_ids: torch.Tensor, *, use_trie: bool) -> str:
    """Return the model's best 5-letter word by beam search; optionally trie-constrained to real words."""
    # each beam: (cum_logprob, [letter_idx...], trie_node_or_None)
    beams = [(0.0, [], TRIE if use_trie else None)]
    for _ in range(5):
        seqs = [prompt + [int(letter_ids[i]) for i in letters] for _, letters, _ in beams]
        batch = torch.tensor(seqs, device=DEV)
        logp = torch.log_softmax(model.forward(batch)[:, -1, letter_ids], dim=-1)  # [B,26]
        cand: list[tuple[float, list[int], TrieNode | None]] = []
        for b, (cum, letters, node) in enumerate(beams):
            if use_trie:
                allowed = [(LETTERS.index(ch), child) for ch, child in node.items()]
            else:
                allowed = [(j, None) for j in range(26)]
            for j, child in allowed:
                cand.append((cum + float(logp[b, j]), letters + [j], child))
        cand.sort(key=lambda x: x[0], reverse=True)
        beams = cand[:WIDTH]
    best = beams[0][1]
    return decode_word([int(letter_ids[i]) for i in best], tok)


def play(model, secret, mode) -> Game:
    letter_ids = letter_id_tensor(tok, DEV)
    game = Game(secret)
    while game.status is Status.ONGOING:
        prompt = build_prompt(game.turns, tok)
        if mode == "greedy":
            ids = torch.tensor(prompt, device=DEV)
            word = decode_word(model.generate(ids, letter_ids, sample=False).tolist(), tok)
        else:
            word = beam_decode(model, prompt, letter_ids, use_trie=(mode == "beam+dict"))
        game.guess(word)
    return game


def summarize(model, mode) -> dict:
    model.eval()
    games = [play(model, s, mode) for s in EVAL]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {
        "win": len(wins) / len(games),
        "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan"),
        "valid": v / n,
        "games": games,
    }


model = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_xl.pt", model)
print(f"=== decoding comparison on {len(EVAL)} held-out (model sft_xl.pt, beam width {WIDTH}) ===", flush=True)
results = {}
for mode in ("greedy", "beam", "beam+dict"):
    r = summarize(model, mode)
    results[mode] = r
    print(f"  {mode:10s}  win={r['win']:.3f}  avg/win={r['avg']:.2f}  valid-rate={r['valid']:.3f}", flush=True)


def render(g) -> str:
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        fb = "❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)
        out.append(f"      {t.guess}  {fb}")
    return "\n".join(out)


print("\n=== beam+dict sample games (the strategy ceiling) ===", flush=True)
for g in results["beam+dict"]["games"][:12]:
    print(render(g), flush=True)
print("\n[BEAM DONE]", flush=True)
