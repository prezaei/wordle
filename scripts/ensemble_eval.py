"""Honest ensemble committee (new experiment): several trained models vote per board — NO dictionary.

Each turn, every model free-gens its greedy guess from the shared board; the committee commits the
MAJORITY-VOTE word (self-consistency across diverse models). Pure test-time compute, ZERO dictionary/clue
enforcement — just multiple models agreeing. Clean protocol: a non-word commit is counted as a turn and
NOT fed back. The hope: diverse models agree on REAL words more often than any one model's greedy (which
emits non-words). Honest: no dict at inference, train-only secrets, disjoint TEST. This is a committee,
labeled as such (a different tier from a single model). Env: ENS_CKPTS (comma-sep, default the validity
family + stage-1), EVAL_SIZE (large).
"""

from __future__ import annotations

import os
import statistics
from collections import Counter

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
LIDS = torch.tensor(LETTER_IDS, device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
CKPTS = os.environ.get("ENS_CKPTS", "runs/cot_eph_aux_fair.pt,runs/validity_max.pt,"
                       "runs/validity_max_v2.pt,runs/validity_max_v3.pt").split(",")


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


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


def play_ensemble(models, secret):
    visible = []
    for attempt in range(1, 7):
        guesses = [gen_guess(m, visible) for m in models]
        cnt = Counter(g for g in guesses if len(g) == 5)
        word = cnt.most_common(1)[0][0] if cnt else "zzzzz"  # majority vote, NO dict
        if len(word) != 5 or not is_valid(word):
            return {"won": False, "guesses": attempt, "reason": "nonword"}
        fb = score(word, secret)
        visible.append(Turn(guess=word, feedback=fb, valid=True))
        if all(c is Color.GREEN for c in fb):
            return {"won": True, "guesses": attempt, "reason": "win"}
    return {"won": False, "guesses": 6, "reason": "exhausted"}


def main():
    _, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    print(f"[ensemble] {len(CKPTS)} models: {CKPTS}", flush=True)
    models = []
    for c in CKPTS:
        m = WordleGenerator(CFG, VOCAB).to(DEV)
        load_checkpoint(c.strip(), m)
        m.eval()
        models.append(m)
    for name, secrets in (("TEST", TEST), ("VAL", VAL)):
        res = [play_ensemble(models, s) for s in secrets]
        wins = [r for r in res if r["won"]]
        nonword = sum(1 for r in res if r["reason"] == "nonword")
        avg = statistics.mean(r["guesses"] for r in wins) if wins else float("nan")
        print(f"  {name}: committee win {len(wins) / len(res):.3f} ({len(wins)}/{len(res)}) avg {avg:.2f} "
              f"| nonword-losses {nonword}", flush=True)
    print(f"  [ref] best single model (clean): validity-max v2 0.302 ; stage-1 0.243", flush=True)
    print("\n[ENSEMBLE DONE]", flush=True)


if __name__ == "__main__":
    main()
