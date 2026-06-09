"""PoE re-rank (word-level fusion variant): G proposes, C ranks — reuses the trained consistency expert.

Per-letter PoE (poe.py) can drift off-distribution (the product of two distributions need not be a real word).
Word-level fusion avoids that: G SAMPLES N candidate words (its common-word generation, temperature), and C
scores each whole candidate by its consistency-likelihood; commit argmax. Honest: G and C are both learned
nets (forward passes), candidates are G's OWN samples, C is a learned scorer — NO dict/trie/engine/answer-set,
the only mask is the 26-letter action space. N=1 == greedy G baseline (built-in ablation).

Reuses runs/poe_c.pt (the consistency expert trained by poe.py) + validity_max_v4 (G).
Env: PR_N(16) PR_TEMP(1.0) PR_G(runs/validity_max_v4.pt) PR_C(runs/poe_c.pt) PR_EVAL_N(48).
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

import poe as P  # reuse vocab/encode/clue_state/board_only/CFG/G+C loading
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Game, Status
from wordle_slm.engine.game import Turn
from wordle_slm.model import WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = P.DEV
tok = P.tok
LIDS = P.LIDS
LETTER_SET = P.LETTER_SET
ALLOWED_GEN = P.ALLOWED_GEN
N = int(os.environ.get("PR_N", "16"))
TEMP = float(os.environ.get("PR_TEMP", "1.0"))


@torch.no_grad()
def _sample_word(G, visible, gen):
    """G's ephemeral-CoT play, sampling the 5 letters (temperature). Returns the word string."""
    seq = P.board_only(visible)
    committed = False
    guess = []
    for _ in range(60):
        logits = G.forward(torch.tensor([seq], device=DEV))[0, -1]
        if committed:
            probs = torch.softmax(logits[LIDS] / TEMP, dim=-1).cpu()
            tid = int(LIDS[int(torch.multinomial(probs, 1, generator=gen))])
            seq.append(tid)
            guess.append(tid)
            if len(guess) >= 5:
                break
        else:
            nxt = int(ALLOWED_GEN[int(torch.argmax(logits[ALLOWED_GEN]))])  # greedy CoT, sample only the word
            seq.append(nxt)
            if nxt == tok.guess_id:
                committed = True
    return "".join(tok.id_to_token(t) for t in guess[:5]) if len(guess) >= 5 else ""


@torch.no_grad()
def _c_score(C, constraints_pre, word):
    """C's mean log-prob of `word`'s letters given the constraint state (consistency-likelihood)."""
    seq = list(constraints_pre)
    total = 0.0
    for ch in word:
        lp = torch.log_softmax(C.forward(torch.tensor([seq], device=DEV))[0, -1][LIDS], dim=-1)
        total += float(lp[ord(ch) - 97])
        seq.append(tok.token_to_id(ch))
    return total / max(len(word), 1)


@torch.no_grad()
def play_rerank(G, C, secret, n, gen):
    g = Game(secret)
    visible: list[Turn] = []
    while g.status is Status.ONGOING:
        cands = {}
        for _ in range(n):
            w = _sample_word(G, visible, gen)
            if len(w) == 5:
                cands[w] = cands.get(w, 0) + 1
        if not cands:
            word = "zzzzz"
        elif n == 1 or C is None:
            word = max(cands, key=cands.get)  # n=1 -> the single sample == greedy-ish baseline
        else:
            pre = P.encode_constraint(*P.clue_state(visible)) + [tok.guess_id]
            word = max(cands, key=lambda w: _c_score(C, pre, w))  # C ranks G's candidates
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break
    return g


def evaluate(G, C, secrets, n, gen):
    games = [play_rerank(G, C, s, n, gen) for s in secrets]
    wins = [x for x in games if x.won]
    nn = sum(len(x.turns) for x in games)
    return {"win": len(wins) / len(games),
            "valid": sum(is_valid(t.guess) for x in games for t in x.turns) / nn if nn else 0.0,
            "avg": statistics.mean(x.guesses_used for x in wins) if wins else float("nan")}


def main():
    train, held = split(seed=0)
    EVAL_N = int(os.environ.get("PR_EVAL_N", "48"))
    VAL, TEST = held[:EVAL_N], held[96:]
    TRAIN_PROBE = train[:EVAL_N]
    G = WordleGenerator(P.CFG, P.VOCAB_G).to(DEV)
    load_checkpoint(os.environ.get("PR_G", "runs/validity_max_v4.pt"), G)
    G.eval()
    C = WordleGenerator(P.CFG, P.VOCAB_C).to(DEV)
    load_checkpoint(os.environ.get("PR_C", "runs/poe_c.pt"), C)
    C.eval()
    gen = torch.Generator()
    gen.manual_seed(0)
    print(f"[rerank] N={N} temp={TEMP} — G samples, C ranks (word-level PoE)", flush=True)

    print("[rerank] VAL: N=1 (G greedy-ish baseline) vs N (C re-rank):", flush=True)
    base = evaluate(G, None, VAL, 1, gen)
    rer = evaluate(G, C, VAL, N, gen)
    delta = rer["win"] - base["win"]
    print(f"  N=1  win {base['win']:.3f} valid {base['valid']:.3f}", flush=True)
    print(f"  N={N} win {rer['win']:.3f} valid {rer['valid']:.3f}  (delta {delta:+.3f})", flush=True)

    # Cost guard: the N-sample TEST eval is ~Nx slower; only run it if VAL re-rank actually beats N=1.
    if delta <= 0.02:
        print(f"\n[rerank] no VAL improvement (delta {delta:+.3f} <= 0.02) -> skipping the expensive TEST. "
              f"NULL: C-rerank does not beat G (same wall as per-letter PoE).", flush=True)
        print("\n[RERANK DONE]", flush=True)
        return

    print("\n[rerank] VAL improved -> memorization audit + TEST:", flush=True)
    tr = evaluate(G, C, TRAIN_PROBE, N, gen)
    print(f"  TRAIN-probe win {tr['win']:.3f}  (if >> TEST -> C memorized)", flush=True)
    TEST_SUB = TEST[:150]  # bound the N-sample TEST cost
    mt = evaluate(G, C, TEST_SUB, N, gen)
    print(f"\n=== PoE re-rank: honest clean-protocol TEST[:150] (G samples, C ranks; no dict) ===", flush=True)
    print(f"  TEST win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST_SUB)))}/{len(TEST_SUB)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  [bar] validity-max v4 clean 0.338 ; train-probe {tr['win']:.3f}", flush=True)
    print("\n[RERANK DONE]", flush=True)


if __name__ == "__main__":
    main()
