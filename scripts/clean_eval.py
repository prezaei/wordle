"""Clean-protocol honest eval (the correct inference): pure free-gen, NO dictionary, non-words counted
but NOT fed back to the model.

Per the honesty rule: the inference layer gets ZERO dictionary/trie help, and a non-word guess (a) counts
as a used turn (the penalty is fair) but (b) is NOT written back into the model's context — so the model
only ever conditions on its VALID guesses + their real feedback. This also fixes the train/inference
mismatch for free: training contexts (teacher games) are valid-only, and now inference contexts are too.

Consequence under greedy: an invalid guess leaves the context unchanged, so the next greedy guess repeats
it — i.e. one non-word effectively ends the game. That is honest: if the model cannot free-gen a valid
word for a board, it loses. We report win, plus the loss breakdown (stuck-on-nonword vs ran-out-of-turns).
Compare to the OLD eval (0.281) which fed non-words back as fabricated all-gray (an OOD context lottery).
Env: EVAL_CKPT, EVAL_SIZE (large default).
"""

from __future__ import annotations

import os
import statistics

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Color
from wordle_slm.engine.game import Turn
from wordle_slm.engine.scoring import score
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
_SIZES = {"tiny": (128, 6, 4, 512, 0.10), "base": (320, 10, 8, 1280, 0.10),
          "large": (512, 16, 8, 2048, 0.10), "xl": (640, 20, 10, 2560, 0.15)}
_d, _l, _h, _ff, _dr = _SIZES[os.environ.get("EVAL_SIZE", "large")]
CFG = ModelConfig(d_model=_d, n_layers=_l, n_heads=_h, d_ff=_ff, context_len=256, dropout=_dr)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
CKPT = os.environ.get("EVAL_CKPT", "runs/cot_eph_aux_fair.pt")
MAX_GUESSES = 6


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.token_to_id(_COLOR[c]) for c in turn.feedback]  # clean protocol: only VALID turns here


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


@torch.no_grad()
def gen_guess(model, visible):
    seq = board_only(visible)
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


def play_clean(model, secret):
    """Pure free-gen; non-words count as a turn but are NOT added to the visible context."""
    visible = []
    for attempt in range(1, MAX_GUESSES + 1):
        word = gen_guess(model, visible)
        if len(word) != 5 or not is_valid(word):
            return {"won": False, "guesses": attempt, "reason": "nonword"}  # greedy repeats -> game lost
        fb = score(word, secret)
        visible.append(Turn(guess=word, feedback=fb, valid=True))
        if all(c is Color.GREEN for c in fb):
            return {"won": True, "guesses": attempt, "reason": "win"}
    return {"won": False, "guesses": MAX_GUESSES, "reason": "exhausted"}


def main():
    _, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    print(f"[clean] ckpt={CKPT}  pure free-gen, no dict, non-words counted+not-fed-back", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()
    for name, secrets in (("TEST", TEST), ("VAL", VAL)):
        res = [play_clean(model, s) for s in secrets]
        wins = [r for r in res if r["won"]]
        nonword = sum(1 for r in res if r["reason"] == "nonword")
        exhausted = sum(1 for r in res if r["reason"] == "exhausted")
        avg = statistics.mean(r["guesses"] for r in wins) if wins else float("nan")
        print(f"  {name}: win {len(wins) / len(res):.3f} ({len(wins)}/{len(res)})  avg {avg:.2f}  "
              f"| losses: stuck-on-nonword {nonword}, ran-out {exhausted}", flush=True)
    print(f"  [ref] OLD eval (fed non-words back as all-gray): TEST 0.281", flush=True)
    print("\n[CLEAN DONE]", flush=True)


if __name__ == "__main__":
    main()
