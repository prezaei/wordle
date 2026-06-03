"""No-repeat decode experiment (driver, not committed): what does forbidding duplicate guesses do?

Same model (runs/sft_xl.pt), same 250 held-out words, beam width 10. Four conditions:
  beam            : beam over the model's own distribution (no dictionary).
  beam +norepeat  : same, but never re-emit a word already guessed this game.
  beam+dict       : beam constrained to real words (the strategy ceiling).
  beam+dict+norep : the above + never repeat a prior guess.
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
WIDTH = 10
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
LETTERS = "abcdefghijklmnopqrstuvwxyz"
TRIE: dict = {}
for w in load_valid_guesses():
    node = TRIE
    for ch in w:
        node = node.setdefault(ch, {})


@torch.no_grad()
def beam_word(model, prompt, letter_ids, *, use_trie: bool, forbidden: set[str]) -> str:
    """Beam-search a 5-letter word; optionally trie-constrained; skip words in `forbidden` if possible."""
    beams = [(0.0, [], TRIE if use_trie else None)]
    for _ in range(5):
        seqs = [prompt + [int(letter_ids[i]) for i in ls] for _, ls, _ in beams]
        logp = torch.log_softmax(model.forward(torch.tensor(seqs, device=DEV))[:, -1, letter_ids], dim=-1)
        cand = []
        for b, (cum, ls, node) in enumerate(beams):
            allowed = ([(LETTERS.index(c), ch) for c, ch in node.items()] if use_trie
                       else [(j, None) for j in range(26)])
            for j, child in allowed:
                cand.append((cum + float(logp[b, j]), ls + [j], child))
        cand.sort(key=lambda x: x[0], reverse=True)
        beams = cand[:WIDTH]
    words = [decode_word([int(letter_ids[i]) for i in ls], tok) for _, ls, _ in beams]
    for w in words:  # highest-prob word not already guessed
        if w not in forbidden:
            return w
    return words[0]


def play(model, secret, *, use_trie: bool, no_repeat: bool) -> Game:
    letter_ids = letter_id_tensor(tok, DEV)
    g = Game(secret)
    while g.status is Status.ONGOING:
        forbidden = {t.guess for t in g.turns} if no_repeat else set()
        g.guess(beam_word(model, build_prompt(g.turns, tok), letter_ids, use_trie=use_trie, forbidden=forbidden))
    return g


def summarize(model, *, use_trie: bool, no_repeat: bool) -> dict:
    model.eval()
    games = [play(model, s, use_trie=use_trie, no_repeat=no_repeat) for s in EVAL]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan"), "valid": v / n, "games": games}


model = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_xl.pt", model)
print(f"=== no-repeat decode on {len(EVAL)} held-out (sft_xl.pt, width {WIDTH}) ===", flush=True)
conds = [
    ("beam", dict(use_trie=False, no_repeat=False)),
    ("beam +norepeat", dict(use_trie=False, no_repeat=True)),
    ("beam+dict", dict(use_trie=True, no_repeat=False)),
    ("beam+dict+norep", dict(use_trie=True, no_repeat=True)),
]
results = {}
for name, kw in conds:
    r = summarize(model, **kw)
    results[name] = r
    print(f"  {name:16s}  win={r['win']:.3f}  avg/win={r['avg']:.2f}  valid={r['valid']:.3f}", flush=True)


def render(g):
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    return "\n".join(out)


print("\n=== beam+dict+norepeat sample games (was repeating before) ===", flush=True)
for g in results["beam+dict+norep"]["games"][:12]:
    print(render(g), flush=True)
print("\n[NOREPEAT DONE]", flush=True)
