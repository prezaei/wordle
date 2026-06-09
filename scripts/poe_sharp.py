"""Sharp consistency expert: train C on REALISTIC TIGHT states (the endgame), then PoE + C-solo.

The ambiguity diagnostic reframed the wall: 61% of G's held-out losses have <=3 consistent words (30 are
UNIQUELY determined) — G fails on near-determined positions, NOT on genuine ambiguity (ceiling +0.41). PoE's
C was too soft because it trained on LOOSE states (random prior guesses). This trains C on a curriculum of
TIGHT states produced by realistic narrowing (consistent prior guesses, like a real solver), so C learns the
ENDGAME: given clues that nearly/fully determine the word, produce THAT word. Honest: answer-AGNOSTIC
(secrets + targets from the full valid-guess dict, never the answer set; the consistency RULE, not a lookup),
eval clean held-out, no engine/dict at inference.

Decisive metrics:
  - C-solo held-out win (C generates from constraint-state alone; clean) — did C learn endgame deduction?
  - C size-1 accuracy: on held-out states where exactly ONE valid word is consistent, does C generate it?
  - PoE (G x C^beta) held-out TEST + memorization audit (TRAIN vs HELD).
Env: PS_STATES(30000) PS_CEPOCHS(35) PS_CPRE(20) PS_LR(3e-4) PS_BETAS("0,1,2,4,8") PS_EVAL_N(48) PS_OUT.
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

import poe as P
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Game, Status
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.engine.game import Turn
from wordle_slm.model import WordleGenerator
from wordle_slm.sft.train import load_checkpoint, save_checkpoint
from wordle_slm.config import SFTConfig

DEV = P.DEV
tok = P.tok
LIDS = P.LIDS
VALID = P.VALID


def tight_states(n_states, rng):
    """Realistic narrowing: play CONSISTENT prior guesses (like a solver) so constraints TIGHTEN, then take
    consistent words as targets. Answer-agnostic (full dict). Produces the endgame states C must master."""
    exs = []
    for _ in range(n_states):
        secret = rng.choice(VALID)
        g = Game(secret)
        turns: list[Turn] = []
        steps = rng.randint(1, 4)  # more steps -> tighter constraints (the endgame curriculum)
        for _ in range(steps):
            if g.status is not Status.ONGOING:
                break
            cand = None
            for _ in range(40):  # a guess consistent with clues so far -> realistic narrowing
                w = rng.choice(VALID)
                if is_consistent(w, turns):
                    cand = w
                    break
            if cand is None:
                break
            t = g.guess(cand)
            if not t.valid or g.won:
                break
            turns.append(t)
        cons = []
        if is_consistent(secret, turns):
            cons.append(secret)
        for _ in range(200):
            w = rng.choice(VALID)
            if w not in cons and is_consistent(w, turns):
                cons.append(w)
            if len(cons) >= 6:
                break
        pre = P.encode_constraint(*P.clue_state(turns)) + [tok.guess_id]
        for w in cons:  # ALL consistent words as targets -> C learns the exact (small) set
            exs.append((pre + P._letters(w) + [tok.eos_id], [False] * len(pre) + [True] * 5 + [False]))
    return exs


@torch.no_grad()
def play_c_solo(C, secret):
    """C generates from the constraint state ALONE (clean protocol) — does the sharp C deduce endgames?"""
    g = Game(secret)
    visible: list[Turn] = []
    while g.status is Status.ONGOING:
        seq = P.encode_constraint(*P.clue_state(visible)) + [tok.guess_id]
        letters = []
        for _ in range(5):
            tid = int(LIDS[int(torch.argmax(C.forward(torch.tensor([seq], device=DEV))[0, -1][LIDS]))])
            letters.append(tok.id_to_token(tid))
            seq.append(tid)
        word = "".join(letters)
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break
    return g


def c_solo_win(C, secrets):
    games = [play_c_solo(C, s) for s in secrets]
    wins = [x for x in games if x.won]
    n = sum(len(x.turns) for x in games)
    return len(wins) / len(games), (sum(is_valid(t.guess) for x in games for t in x.turns) / n if n else 0.0)


@torch.no_grad()
def c_size1_accuracy(C, secrets, rng):
    """On held-out states where EXACTLY ONE valid word is consistent, does C generate it? (clean test of
    endgame deduction generalization)."""
    ok = tot = 0
    for s in secrets:
        g = Game(s)
        turns: list[Turn] = []
        for _ in range(5):
            if g.status is not Status.ONGOING:
                break
            cand = next((w for w in (rng.choice(VALID) for _ in range(60)) if is_consistent(w, turns)), None)
            if cand is None:
                break
            t = g.guess(cand)
            if not t.valid:
                break
            turns.append(t)
            cons = [w for w in VALID if is_consistent(w, turns)]
            if len(cons) == 1:  # uniquely determined -> C must produce cons[0]
                seq = P.encode_constraint(*P.clue_state(turns)) + [tok.guess_id]
                letters = []
                for _ in range(5):
                    tid = int(LIDS[int(torch.argmax(C.forward(torch.tensor([seq], device=DEV))[0, -1][LIDS]))])
                    letters.append(tok.id_to_token(tid))
                    seq.append(tid)
                ok += int("".join(letters) == cons[0])
                tot += 1
                break
    return ok / tot if tot else float("nan"), tot


def main():
    train, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    EVAL_N = int(os.environ.get("PS_EVAL_N", "48"))
    G_PATH = os.environ.get("PS_G", "runs/validity_max_v4.pt")
    OUT = os.environ.get("PS_OUT", "runs/poe_sharp_c.pt")
    N_STATES = int(os.environ.get("PS_STATES", "30000"))
    CEPOCHS = int(os.environ.get("PS_CEPOCHS", "35"))
    CPRE = int(os.environ.get("PS_CPRE", "20"))
    LR = float(os.environ.get("PS_LR", "3e-4"))
    BETAS = [float(b) for b in os.environ.get("PS_BETAS", "0,1,2,4,8").split(",")]
    print(f"[sharp] G={G_PATH} tight-states={N_STATES} C-epochs={CEPOCHS} betas={BETAS}", flush=True)

    G = WordleGenerator(P.CFG, P.VOCAB_G).to(DEV)
    load_checkpoint(G_PATH, G)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    C = WordleGenerator(P.CFG, P.VOCAB_C).to(DEV)
    print(f"[sharp] C spell warm-up ({CPRE} ep)", flush=True)
    P.warmup_C(C, CPRE)
    print(f"[sharp] building TIGHT consistency states ({N_STATES}) …", flush=True)
    exs = tight_states(N_STATES, Random(1))
    Random(0).shuffle(exs)
    print(f"[sharp] C training examples={len(exs)}", flush=True)
    P._train(C, exs, CEPOCHS, LR, with_aux=True, label="sharpC")
    save_checkpoint(OUT, C, torch.optim.AdamW(C.parameters(), lr=LR), CEPOCHS, SFTConfig())
    C.eval()

    # decisive diagnostics: did C learn endgame deduction (generalizing)?
    cs_win, cs_valid = c_solo_win(C, VAL[:EVAL_N])
    s1_acc, s1_n = c_size1_accuracy(C, VAL[:EVAL_N], Random(7))
    print(f"\n[sharp] C-solo VAL win {cs_win:.3f} valid {cs_valid:.3f}  (dense-encode soft-C bar: 0.065)", flush=True)
    print(f"[sharp] C size-1 accuracy (held-out unique-answer states): {s1_acc:.3f} over {s1_n} states", flush=True)

    print("\n[sharp] PoE beta sweep on VAL (beta=0 == G-alone):", flush=True)
    best_beta, best_win = 0.0, -1.0
    for beta in BETAS:
        m = P.evaluate(G, C, VAL[:EVAL_N], beta)
        flag = ""
        if m["win"] > best_win:
            best_win, best_beta = m["win"], beta
            flag = "  <- best"
        print(f"  beta={beta:>4}: VAL win {m['win']:.3f} valid {m['valid']:.3f}{flag}", flush=True)

    tr = P.evaluate(G, C, tuple(train[:EVAL_N]), best_beta)
    mt = P.evaluate(G, C, TEST, best_beta)
    g_only = P.evaluate(G, C, TEST, 0.0)
    print(f"\n=== sharp-PoE: honest TEST (G x C^{best_beta}, no dict) ===", flush=True)
    print(f"  TEST win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f}", flush=True)
    print(f"  G-only TEST {g_only['win']:.3f}  delta {mt['win'] - g_only['win']:+.3f}  | TRAIN-probe {tr['win']:.3f} (audit)", flush=True)
    print(f"  [bar] validity-max v4 clean 0.338", flush=True)
    print("\n[SHARP DONE]", flush=True)


if __name__ == "__main__":
    main()
