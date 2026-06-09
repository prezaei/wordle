"""Ambiguity diagnostic: WHY does G lose on held-out, and can ANY honest tiebreaker help?

The wall is "can't pin THE answer among consistent candidates." This quantifies it: play G on held-out,
and on each turn measure |consistent set| = #valid words still consistent with the clues. The decision:
  - If at G's losing guesses the consistent set is LARGE -> the answer is genuinely ambiguous; no honest
    prior (frequency, etc.) can reliably pick it -> the wall is FUNDAMENTAL, don't pursue the freq prior.
  - If the consistent set is SMALL (<=3) but G still guessed wrong -> a tiebreaker (e.g. common-word prior)
    could rescue those games -> the freq prior has real EV, worth a greenlight.

Honest analysis only (uses is_consistent for measurement — NOT at inference). No model changes.
Env: AD_G(runs/validity_max_v4.pt) AD_N(150 held-out secrets) AD_SAMPLE(0 = exact scan of full dict).
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

import poe as P
from wordle_slm.data import split
from wordle_slm.engine import Game, Status
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.engine.game import Turn
from wordle_slm.model import WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = P.DEV
tok = P.tok
LIDS = P.LIDS
ALLOWED_GEN = P.ALLOWED_GEN
VALID = P.VALID


@torch.no_grad()
def _g_guess(G, visible):
    seq = P.board_only(visible)
    committed, guess = False, []
    for _ in range(60):
        nxt = int(ALLOWED_GEN[int(torch.argmax(G.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
        seq.append(nxt)
        if committed:
            if nxt in P.LETTER_SET:
                guess.append(nxt)
            if len(guess) >= 5:
                break
        elif nxt == tok.guess_id:
            committed = True
    return "".join(tok.id_to_token(t) for t in guess[:5]) if len(guess) >= 5 else "zzzzz"


def _consistent_count(turns, pool):
    return sum(1 for w in pool if is_consistent(w, turns))


def main():
    _, held = split(seed=0)
    G_PATH = os.environ.get("AD_G", "runs/validity_max_v4.pt")
    NSEC = int(os.environ.get("AD_N", "150"))
    secrets = held[96:96 + NSEC]
    pool = VALID  # exact scan of the full valid-guess dictionary
    G = WordleGenerator(P.CFG, P.VOCAB_G).to(DEV)
    load_checkpoint(G_PATH, G)
    G.eval()
    print(f"[diag] G={G_PATH} secrets={len(secrets)} (held-out) pool=|valid|={len(pool)}", flush=True)

    lost_final_sizes = []          # |consistent set| at the turn G made its losing guess
    lost_small_rescuable = 0       # lost games where the losing-turn consistent set <= 3 (tiebreaker could win)
    lost_secret_in_small = 0       # ...and the secret is in that small set (it always is, but for clarity)
    won = 0
    for s in secrets:
        g = Game(s)
        visible: list[Turn] = []
        last_consistent_before = None
        while g.status is Status.ONGOING:
            # consistent set the model is choosing FROM this turn (before guessing)
            csize = _consistent_count(visible, pool)
            w = _g_guess(G, visible)
            t = g.guess(w if len(w) == 5 else "zzzzz")
            last_consistent_before = csize
            if t.valid:
                visible.append(t)
            else:
                break
        if g.won:
            won += 1
        else:
            lost_final_sizes.append(last_consistent_before if last_consistent_before is not None else len(pool))
            if last_consistent_before is not None and last_consistent_before <= 3:
                lost_small_rescuable += 1
                lost_secret_in_small += 1  # secret is always consistent -> in the set
    nlost = len(lost_final_sizes)
    print(f"\n[diag] held-out: won {won}/{len(secrets)} ({won / len(secrets):.3f}), lost {nlost}", flush=True)
    if lost_final_sizes:
        med = statistics.median(lost_final_sizes)
        mean = statistics.mean(lost_final_sizes)
        buckets = {"1": 0, "2-3": 0, "4-10": 0, "11-50": 0, ">50": 0}
        for c in lost_final_sizes:
            buckets["1" if c == 1 else "2-3" if c <= 3 else "4-10" if c <= 10 else "11-50" if c <= 50 else ">50"] += 1
        print(f"[diag] |consistent set| at the LOSING guess: median {med:.0f}  mean {mean:.1f}", flush=True)
        print(f"[diag] distribution: {buckets}", flush=True)
        print(f"[diag] RESCUABLE by a tiebreaker (consistent set <=3 but G missed): "
              f"{lost_small_rescuable}/{nlost} lost = {lost_small_rescuable / nlost:.1%} of losses "
              f"(= +{lost_small_rescuable / len(secrets):.3f} win ceiling if a prior always picked right)", flush=True)
        print("\n[verdict] If most losses have LARGE consistent sets -> ambiguity is fundamental, a freq prior", flush=True)
        print("          won't help. If many are <=3 -> a common-word tiebreaker has real EV.", flush=True)
    print("\n[DIAG DONE]", flush=True)


if __name__ == "__main__":
    main()
