"""Diagnostic (not committed): does the model CASCADE after its first invalid (non-word) guess?

Motivated by a real flaw: invalid guesses (a) waste a turn and (b) are written into the inference
context with FABRICATED all-gray feedback — an out-of-distribution context the model never saw in
training (teacher games are always valid). Hypothesis: once the model emits one non-word, the poisoned
OOD context makes it emit more → a cascade that loses games. We measure, on stage-1 free-gen TEST:
overall invalid rate, recovery rate (validity of guesses AFTER the first invalid in a game), and how many
LOSSES are dominated by non-words. Honest: read-only, free-gen greedy, disjoint TEST, zero rules.
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
    print(f"[cascade] ckpt={CKPT}  free-gen on |TEST|={len(TEST)}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()
    n_guesses = n_invalid = 0
    games_with_invalid = 0
    post_first_invalid_total = post_first_invalid_valid = 0
    losses = loss_last_invalid = 0
    repeat_after_invalid = 0  # did it re-emit a word it already played?
    for secret in TEST:
        g = Game(secret)
        flags, words = [], []
        seen_invalid = False
        while g.status is Status.ONGOING:
            w = gen_guess(model, g.turns) or "zzzzz"
            if len(w) != 5:
                w = "zzzzz"
            v = is_valid(w)
            if seen_invalid:
                post_first_invalid_total += 1
                post_first_invalid_valid += int(v)
                if w in words:
                    repeat_after_invalid += 1
            if not v:
                seen_invalid = True
            flags.append(v)
            words.append(w)
            g.guess(w)
        n = len(flags)
        n_guesses += n
        inv = sum(1 for f in flags if not f)
        n_invalid += inv
        games_with_invalid += int(inv > 0)
        if not g.won:
            losses += 1
            loss_last_invalid += int(not flags[-1])
    print(f"  invalid guesses overall      : {n_invalid}/{n_guesses} = {100 * n_invalid / n_guesses:.1f}%", flush=True)
    print(f"  games with >=1 invalid       : {games_with_invalid}/{len(TEST)} = {100 * games_with_invalid / len(TEST):.1f}%", flush=True)
    if post_first_invalid_total:
        rec = 100 * post_first_invalid_valid / post_first_invalid_total
        print(f"  RECOVERY: validity of guesses AFTER the 1st invalid: {rec:.1f}%  "
              f"(low => cascade; vs overall valid {100 * (1 - n_invalid / n_guesses):.1f}%)", flush=True)
        print(f"  re-emitted an already-played word after an invalid : {repeat_after_invalid}", flush=True)
    print(f"  losses whose LAST guess was invalid: {loss_last_invalid}/{losses} = "
          f"{100 * loss_last_invalid / losses:.1f}%" if losses else "  no losses", flush=True)
    print("\n  [read] if recovery-validity << overall-validity, the OOD all-gray context IS poisoning play "
          "-> teaching invalid-recovery (or omitting invalids from context) should help.", flush=True)
    print("\n[CASCADE DONE]", flush=True)


if __name__ == "__main__":
    main()
