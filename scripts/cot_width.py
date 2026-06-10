"""More/less CoT, the DRIFT-FREE way: force the real 0.338 model to emit exactly N candidate-blocks before
it may commit, at inference. Tests whether wider in-context search helps WITHOUT retraining (so no
fine-tune drift). N=1 = "less CoT" (commit after one candidate); N=3 ~ native; N=6/10 = "more CoT".
Honest: greedy, no dict/consistency — only the guess_id token is suppressed until N THINK-blocks exist
(a decoding-format constraint on the model's OWN search, not answer info). Eval = clean full-TEST.
"""

from __future__ import annotations

import statistics

import torch

import format_sweep as F  # board, ALLOWED_GEN, THINK, LETTER_SET, tok, CFG, VOCAB
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Game, Status
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.model import WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = F.DEV
tok = F.tok
GID = tok.guess_id
GID_SUB = len(F.ALLOWED_GEN) - 1  # guess_id is the last entry of ALLOWED_GEN


@torch.no_grad()
def forced_play(model, secrets, n_cands, chunk=384):
    """Batched clean play, but a game may not emit <GUESS> until it has produced n_cands THINK-blocks."""
    model.eval()
    games = [Game(s) for s in secrets]
    visible: list[list] = [[] for _ in secrets]
    alive = [True] * len(secrets)
    for _turn in range(6):
        idx = [i for i in range(len(games)) if alive[i] and games[i].status is Status.ONGOING]
        if not idx:
            break
        for cs in range(0, len(idx), chunk):
            sub = idx[cs : cs + chunk]
            seqs = [F.board(visible[i]) for i in sub]
            committed = [False] * len(sub)
            think = [0] * len(sub)
            guess: list[list[int]] = [[] for _ in sub]
            tdone = [False] * len(sub)
            for _step in range(120):  # bigger budget: more candidates take more tokens
                act = [j for j in range(len(sub)) if not tdone[j]]
                if not act:
                    break
                L = max(len(seqs[j]) for j in act)
                ids = torch.full((len(act), L), tok.pad_id, dtype=torch.long, device=DEV)
                last = torch.empty(len(act), dtype=torch.long, device=DEV)
                for a, j in enumerate(act):
                    ids[a, : len(seqs[j])] = torch.tensor(seqs[j], device=DEV)
                    last[a] = len(seqs[j]) - 1
                logits = model.forward(ids)[torch.arange(len(act), device=DEV), last]
                sub_logits = logits[:, F.ALLOWED_GEN].clone()
                for a, j in enumerate(act):  # forbid committing until n_cands candidates exist
                    if not committed[j] and think[j] < n_cands:
                        sub_logits[a, GID_SUB] = float("-inf")
                pick = F.ALLOWED_GEN[torch.argmax(sub_logits, dim=-1)].tolist()
                for a, j in enumerate(act):
                    t = int(pick[a])
                    seqs[j].append(t)
                    if t == F.THINK:
                        think[j] += 1
                    if committed[j]:
                        if t in F.LETTER_SET:
                            guess[j].append(t)
                        if len(guess[j]) >= 5:
                            tdone[j] = True
                    elif t == GID:
                        committed[j] = True
            for a, i in enumerate(sub):
                gl = guess[a]
                word = "".join(tok.id_to_token(t) for t in gl[:5]) if len(gl) >= 5 else "zzzzz"
                turn = games[i].guess(word if len(word) == 5 else "zzzzz")
                if turn.valid:
                    visible[i].append(turn)
                else:
                    alive[i] = False
    return games


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
    return len(wins) / len(games), sum(is_valid(t.guess) for g in games for t in g.turns) / n if n else 0.0, cg / vg if vg else 0.0


def main():
    _, held = split(seed=0)
    TEST = held[96:]
    m = WordleGenerator(F.CFG, F.VOCAB).to(DEV)
    load_checkpoint("runs/validity_max_v4.pt", m)
    m.eval()
    print(f"[cot-width] forcing N candidate-blocks before commit on the REAL 0.338 model, TEST n={len(TEST)}", flush=True)
    print("  N   win        valid   respect", flush=True)
    for N in (1, 3, 6, 10):
        w, v, r = metrics(forced_play(m, list(TEST), N))
        print(f"  {N:>2}  {w:.3f} ({int(round(w*len(TEST)))}/{len(TEST)})  {v:.3f}  {r:.3f}", flush=True)
    print("\n[COT-WIDTH DONE]", flush=True)


if __name__ == "__main__":
    main()
