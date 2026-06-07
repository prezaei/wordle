"""Free-gen greedy eval for any checkpoint (honest path: ephemeral CoT, ZERO inference rules).

Reports held-out TEST (held[96:]) + VAL (held[:96]) win/valid/avg — the honest headline metric. Reusable
across experiments. Env: EVAL_CKPT (required), EVAL_SIZE (tiny|base|large|xl, default large).
"""

from __future__ import annotations

import os
import statistics

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
_SIZES = {"tiny": (128, 6, 4, 512, 0.10), "base": (320, 10, 8, 1280, 0.10),
          "large": (512, 16, 8, 2048, 0.10), "xl": (640, 20, 10, 2560, 0.15)}
_d, _l, _h, _ff, _dr = _SIZES[os.environ.get("EVAL_SIZE", "large")]
CFG = ModelConfig(d_model=_d, n_layers=_l, n_heads=_h, d_ff=_ff, context_len=256, dropout=_dr)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
CKPT = os.environ["EVAL_CKPT"]


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


@torch.no_grad()
def play(model, secret):
    g = Game(secret)
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)
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
        word = "".join(tok.id_to_token(t) for t in guess[:5])
        g.guess(word if len(word) == 5 else "zzzzz")
    return g


def metrics(model, secrets):
    games = [play(model, s) for s in secrets]
    wins = [g for g in games if g.won]
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": sum(is_valid(t.guess) for g in games for t in g.turns) / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def main():
    _, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    print(f"[eval] ckpt={CKPT} size={os.environ.get('EVAL_SIZE', 'large')}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()
    mt, mv = metrics(model, TEST), metrics(model, VAL)
    print(f"  TEST  win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  VAL   win {mv['win']:.3f} valid {mv['valid']:.3f}", flush=True)
    print(f"  [ref] stage-1 free-gen TEST 0.281/0.662", flush=True)
    print("\n[EVAL DONE]", flush=True)


if __name__ == "__main__":
    main()
