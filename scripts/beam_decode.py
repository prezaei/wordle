"""Honest beam decode (#2): commit the highest-JOINT-probability 5-letter word under the model's OWN
distribution, instead of greedy per-position argmax. NO dictionary, NO consistency filter — beams are
ranked purely by the model's letter log-probs (masked to the 26 letters). Greedy can commit a
locally-good-but-globally-suboptimal early letter -> a non-word; beam can recover a higher-prob valid word
the greedy path missed. Tests whether better decoding of the SAME 0.338 weights trims the ~14% non-word /
clue-violation losses. B=1 == greedy (sanity). Clean full-TEST.
"""

from __future__ import annotations

import statistics

import torch

import format_sweep as F  # board, ALLOWED_GEN, LIDS, LETTER_SET, tok, CFG, VOCAB
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Game, Status
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.model import WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = F.DEV
tok = F.tok


@torch.no_grad()
def _commit_prefix(model, visible):
    """Greedy ephemeral-CoT up to the commit point (ends at <GUESS>)."""
    seq = F.board(visible)
    for _ in range(60):
        nxt = int(F.ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][F.ALLOWED_GEN]))])
        seq.append(nxt)
        if nxt == tok.guess_id:
            return seq
    seq.append(tok.guess_id)
    return seq


@torch.no_grad()
def beam_word(model, prefix, B):
    """Beam search over the 5 guess letters by joint model log-prob (no dict). Returns the top word."""
    beams = [(list(prefix), 0.0)]  # (seq, cum_logprob)
    for _ in range(5):
        seqs = [s for s, _ in beams]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_id, dtype=torch.long, device=DEV)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, device=DEV)
        last = torch.tensor([len(s) - 1 for s in seqs], device=DEV)
        logp = torch.log_softmax(model.forward(ids)[torch.arange(len(seqs), device=DEV), last][:, F.LIDS], dim=-1)  # [nb,26]
        cand = []
        for bi, (s, lp) in enumerate(beams):
            for li in range(26):
                cand.append((s + [int(F.LIDS[li])], lp + float(logp[bi, li])))
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = cand[:B]
    best = beams[0][0]
    return "".join(tok.id_to_token(t) for t in best[-5:])


@torch.no_grad()
def play_beam(model, secret, B):
    g = Game(secret)
    visible = []
    while g.status is Status.ONGOING:
        word = beam_word(model, _commit_prefix(model, visible), B)
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break
    return g


def metrics(games):
    wins = [g for g in games if g.won]
    n = sum(len(g.turns) for g in games)
    vg = cg = 0
    for g in games:
        for k, t in enumerate(g.turns):
            if not is_valid(t.guess):
                continue
            vg += 1
            cg += is_consistent(t.guess, g.turns[:k])
    return (len(wins) / len(games), sum(is_valid(t.guess) for g in games for t in g.turns) / n if n else 0.0,
            cg / vg if vg else 0.0, statistics.mean(g.guesses_used for g in wins) if wins else float("nan"))


def main():
    _, held = split(seed=0)
    TEST = held[96:]
    m = WordleGenerator(F.CFG, F.VOCAB).to(DEV)
    load_checkpoint("runs/validity_max_v4.pt", m)
    m.eval()
    print(f"[beam] validity_max_v4 on TEST (n={len(TEST)}); B=1 == greedy baseline", flush=True)
    print("  B    win        valid   respect  avg", flush=True)
    for B in (1, 4, 8, 16):
        w, v, r, a = metrics([play_beam(m, s, B) for s in TEST])
        print(f"  {B:>2}  {w:.3f} ({int(round(w*len(TEST)))}/{len(TEST)})  {v:.3f}  {r:.3f}  {a:.2f}", flush=True)
    print("\n[BEAM DONE]  (bar: greedy 0.338)", flush=True)


if __name__ == "__main__":
    main()
