"""Beam search over the validity trie (diagnostic, not committed): a different inference search.

Per turn: free-gen the think scratchpad until <GUESS>, then **beam-search the 5 guess letters**, masked
to dictionary-valid continuations (the trie), maximizing cumulative letter log-prob. This finds the
model's *most-probable valid word* (sequence-level argmax over real words) — vs greedy-constrained
(token-level argmax, 0.436) and best-of-N (stochastic sample + vote, 0.72). Aided (spelling-only, public
dictionary; no clue-consistency). Eval pure-deduction held-out. Env: BEAM_B (default 8), BEAM_CKPT, SIZE.
"""

from __future__ import annotations

import os
import statistics

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import _valid_trie, load_checkpoint

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
_SIZES = {"tiny": (128, 6, 4, 512, 0.10), "base": (320, 10, 8, 1280, 0.10),
          "large": (512, 16, 8, 2048, 0.10), "xl": (640, 20, 10, 2560, 0.15)}
_d, _l, _h, _ff, _dr = _SIZES[os.environ.get("SIZE", "large")]
CFG = ModelConfig(d_model=_d, n_layers=_l, n_heads=_h, d_ff=_ff, context_len=256, dropout=_dr)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
TRIE = _valid_trie()
B = int(os.environ.get("BEAM_B", "8"))
CKPT = os.environ.get("BEAM_CKPT", "runs/cot_eph_aux_fair.pt")


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
def _free_to_commit(model, seq):
    for _ in range(60):
        nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
        seq.append(nxt)
        if nxt == tok.guess_id:
            return seq
    seq.append(tok.guess_id)
    return seq


@torch.no_grad()
def play_beam(model, secret, beam):
    g = Game(secret)
    while g.status is Status.ONGOING:
        seq = _free_to_commit(model, board_only(g.turns))
        beams = [(0.0, [], TRIE)]  # (cum_logp, letter_token_ids, trie_node)
        for _ in range(5):
            rows = [seq + letters for _, letters, _ in beams]
            L = max(len(r) for r in rows)
            ids = torch.full((len(rows), L), tok.pad_id, dtype=torch.long, device=DEV)
            for i, r in enumerate(rows):
                ids[i, : len(r)] = torch.tensor(r, device=DEV)
            last = torch.tensor([len(r) - 1 for r in rows], device=DEV)
            logits = model.forward(ids)[torch.arange(len(rows), device=DEV), last]
            logp = torch.log_softmax(logits[:, LETTER_IDS], dim=-1)  # [nbeam, 26]
            cand = []
            for bi, (cum, letters, node) in enumerate(beams):
                for idx in node:  # only trie-valid continuations
                    cand.append((cum + float(logp[bi, idx]), letters + [LETTER_LO + idx], node[idx]))
            if not cand:
                break
            beams = sorted(cand, key=lambda x: -x[0])[:beam]
        word = "".join(tok.id_to_token(t) for t in beams[0][1][:5]) if beams else "zzzzz"
        g.guess(word if len(word) == 5 else "zzzzz")
    return g


def metrics(games):
    wins = [x for x in games if x.won]
    n = sum(len(x.turns) for x in games)
    return {"win": len(wins) / len(games), "valid": sum(is_valid(t.guess) for x in games for t in x.turns) / n if n else 0.0,
            "avg": statistics.mean(x.guesses_used for x in wins) if wins else float("nan")}


def main():
    _, held = split(seed=0)
    val, test = tuple(held[:96]), tuple(held[96:])
    print(f"[beam] ckpt={CKPT} B={B}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()
    mv = metrics([play_beam(model, s, B) for s in val])
    mt = metrics([play_beam(model, s, B) for s in test])
    print(f"  beam-{B}  VAL : win {mv['win']:.3f} valid {mv['valid']:.3f}", flush=True)
    print(f"  beam-{B}  TEST: win {mt['win']:.3f} valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print("  [ref] free-gen 0.281 ; constrained-greedy 0.436 ; best-of-128 0.719", flush=True)
    print("\n[BEAM DONE]", flush=True)


if __name__ == "__main__":
    main()
