"""Best-of-N self-consistency eval (diagnostic, not committed): test-time compute, honest-aided.

Per turn, sample N free-gen guesses, keep the VALID ones (a spelling-only filter — public dictionary,
NOT clue-consistency, so it's not the rejected candidate-ranking), and play the MAJORITY VOTE (the most
common valid word). This is test-time compute + self-consistency on the model's own distribution. It is
*aided* (uses is_valid to filter samples), so it's labeled as such — between pure free-gen (0.281) and
spelling-constrained decode (0.436). Eval on the stage-1 model. Env: BON_N (default 16), BON_CKPT.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter

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
_d, _l, _h, _ff, _dr = _SIZES[os.environ.get("SIZE", "large")]  # must match the checkpoint's architecture
CFG = ModelConfig(d_model=_d, n_layers=_l, n_heads=_h, d_ff=_ff, context_len=256, dropout=_dr)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
N = int(os.environ.get("BON_N", "16"))
TEMP = 1.0
CKPT = os.environ.get("BON_CKPT", "runs/cot_eph_aux_fair.pt")
FILTER = os.environ.get("BON_FILTER", "valid")  # "valid" (spelling-aided) | "none" (pure free-gen vote)


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
def sample_guesses(model, prompt, n, temp=TEMP, max_new=44):
    """Sample n free-gen guesses for one prompt (batched). Returns list of 5-letter words."""
    maxL = len(prompt) + max_new
    ids = torch.full((n, maxL), tok.pad_id, dtype=torch.long, device=DEV)
    for i in range(n):
        ids[i, : len(prompt)] = torch.tensor(prompt, device=DEV)
    cur = torch.full((n,), len(prompt), device=DEV)
    committed, collected, done = [False] * n, [[] for _ in range(n)], [False] * n
    ar = torch.arange(n, device=DEV)
    for _ in range(max_new):
        if all(done):
            break
        logits = model.forward(ids)[ar, cur - 1]
        probs = torch.softmax(logits[:, ALLOWED_GEN] / temp, dim=-1).cpu()
        choice = torch.multinomial(probs, 1).squeeze(1)
        for i in range(n):
            if done[i]:
                continue
            t = int(ALLOWED_GEN[int(choice[i])])
            ids[i, int(cur[i])] = t
            cur[i] = int(cur[i]) + 1
            if committed[i]:
                if t in LETTER_SET:
                    collected[i].append(t)
                if len(collected[i]) >= 5:
                    done[i] = True
            elif t == tok.guess_id:
                committed[i] = True
            if int(cur[i]) >= maxL:
                done[i] = True
    return ["".join(tok.id_to_token(t) for t in c[:5]) for c in collected if len(c) >= 5]


@torch.no_grad()
def play_bon(model, secret, n):
    g = Game(secret)
    while g.status is Status.ONGOING:
        words = sample_guesses(model, board_only(g.turns), n)
        pool = [w for w in words if is_valid(w)] if FILTER == "valid" else words  # spelling filter optional
        if pool:
            guess = Counter(pool).most_common(1)[0][0]  # majority vote (self-consistency)
        elif words:
            guess = Counter(words).most_common(1)[0][0]
        else:
            guess = "zzzzz"
        g.guess(guess)
    return g


def metrics(games):
    wins = [g for g in games if g.won]
    nn = sum(len(g.turns) for g in games)
    return {
        "win": len(wins) / len(games),
        "valid": sum(is_valid(t.guess) for g in games for t in g.turns) / nn if nn else 0.0,
        "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan"),
    }


def main():
    _, held = split(seed=0)
    val, test = tuple(held[:96]), tuple(held[96:])
    print(f"[bon] ckpt={CKPT} N={N}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()
    mv = metrics([play_bon(model, s, N) for s in val])
    mt = metrics([play_bon(model, s, N) for s in test])
    print(f"  best-of-{N}  VAL : win {mv['win']:.3f} valid {mv['valid']:.3f}", flush=True)
    print(f"  best-of-{N}  TEST: win {mt['win']:.3f} valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print("  [ref] free-gen TEST 0.281/0.662 ; constrained-decode TEST 0.436/1.000", flush=True)
    print("\n[BON DONE]", flush=True)


if __name__ == "__main__":
    main()
