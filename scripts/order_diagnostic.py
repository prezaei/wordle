"""Diagnostic (not committed): do free-gen failures correlate with GREEN POSITION?

Tests the "spell backwards / generate constrained-position-first" hypothesis. For stage-1 free-gen on the
held-out TEST split, per turn we record the greens KNOWN from the board (position+letter), then check the
committed guess for: (a) validity (real word?), (b) green-respect (does it keep every known green?). We
bucket turns by whether a known green sits LATE (position 4-5) vs only EARLY/none, and report failure
rates. If late-green turns fail more, left-to-right generation is the culprit and an infill/any-order
mechanism is worth building. Honest: read-only eval on disjoint TEST, free-gen greedy, zero rules.
"""

from __future__ import annotations

import os

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
CKPT = os.environ.get("EVAL_CKPT", "runs/cot_eph_aux_fair.pt")


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def known_greens(turns):
    """Map position -> green letter, accumulated over all prior feedback."""
    g = {}
    for t in turns:
        if t.feedback is None:
            continue
        for i, c in enumerate(t.feedback):
            if c is Color.GREEN:
                g[i] = t.guess[i]
    return g


@torch.no_grad()
def gen_guess(model, turns):
    seq = board_only(turns)
    guess, committed = [], False
    for _ in range(60):
        nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
        seq.append(nxt)
        if committed:
            if nxt in LETTER_SET:
                guess.append(nxt)
            if len(guess) >= 5:
                break
        elif nxt == tok.guess_id:
            committed = True
    return "".join(tok.id_to_token(t) for t in guess[:5])


def main():
    _, held = split(seed=0)
    TEST = tuple(held[96:])
    print(f"[order] ckpt={CKPT}  free-gen on |TEST|={len(TEST)}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()
    # buckets: turns with a LATE green (pos 4-5) vs an EARLY-only green vs NO green
    stats = {"late": [0, 0, 0], "early": [0, 0, 0], "none": [0, 0, 0]}  # [n, invalid, green_violation]
    for secret in TEST:
        g = Game(secret)
        while g.status is Status.ONGOING:
            greens = known_greens(g.turns)
            word = gen_guess(model, g.turns)
            if len(word) != 5:
                word = "zzzzz"
            bucket = "none" if not greens else ("late" if any(p >= 3 for p in greens) else "early")
            invalid = 0 if is_valid(word) else 1
            violation = 0 if all(word[p] == ch for p, ch in greens.items()) else 1
            stats[bucket][0] += 1
            stats[bucket][1] += invalid
            stats[bucket][2] += violation
            g.guess(word)
    print("  bucket    turns  invalid%  green-violation%", flush=True)
    for b in ("none", "early", "late"):
        n, inv, vio = stats[b]
        if n:
            print(f"  {b:7}  {n:5}   {100 * inv / n:6.1f}    {100 * vio / n:6.1f}", flush=True)
    print("\n  [read] if 'late' invalid%/violation% >> 'early', L->R generation is the culprit "
          "-> infill/any-order worth building. If flat, order is NOT the lever.", flush=True)
    print("\n[ORDER DONE]", flush=True)


if __name__ == "__main__":
    main()
