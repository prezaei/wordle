"""Green-anchored decoding (tests the 'spell from the constraint' fix): force-copy known greens.

The order diagnostic showed free-gen invalid-rate jumps to 52% (and 19.5% green-violations) when a green
sits late in the word — L->R generation commits the start, then can't compose a valid word with the
required late letter. Fix under test: when committing the 5 guess letters, FORCE the known green letters
at their positions and let the model greedily generate only the blanks. The model still generates the word;
we only hold the letters it already earned (basic play, not candidate-ranking). Compare three modes via
ANCHOR env: none (=pure free-gen 0.281 baseline), green (force greens), green+trie (force greens AND mask
blanks to dictionary-valid continuations). Honest: free-gen otherwise, greens are the model's own clues,
eval on disjoint TEST. Env: EVAL_CKPT, ANCHOR=none|green|greentrie.
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
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
CKPT = os.environ.get("EVAL_CKPT", "runs/cot_eph_aux_fair.pt")
ANCHOR = os.environ.get("ANCHOR", "green")  # none | green | greentrie
TRIE = _valid_trie()


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def known_greens(turns):
    g = {}
    for t in turns:
        if t.feedback is None:
            continue
        for i, c in enumerate(t.feedback):
            if c is Color.GREEN:
                g[i] = t.guess[i]
    return g


@torch.no_grad()
def gen_guess(model, turns):
    greens = known_greens(turns) if ANCHOR != "none" else {}
    seq = board_only(turns)
    # free-gen the think until commit
    committed = False
    for _ in range(60):
        nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
        seq.append(nxt)
        if nxt == tok.guess_id:
            committed = True
            break
    if not committed:
        seq.append(tok.guess_id)
    # commit 5 letters: force greens, optionally mask blanks to trie, else greedy
    letters, node = [], TRIE
    for pos in range(5):
        if pos in greens:  # FORCE the known green letter
            ch = greens[pos]
            tid = tok.token_to_id(ch)
        else:
            logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
            if ANCHOR == "greentrie" and node:
                opts = [LETTER_LO + idx for idx in node]
                tid = max(opts, key=lambda t: float(logits[t]))
            else:
                tid = int(LETTER_IDS[int(torch.argmax(logits[torch.tensor(LETTER_IDS, device=DEV)]))])
            ch = tok.id_to_token(tid)
        letters.append(ch)
        seq.append(tid)
        node = node.get((tok.token_to_id(ch) - LETTER_LO), {}) if node else {}
    return "".join(letters)


def metrics(model, secrets):
    games = []
    for s in secrets:
        g = Game(s)
        while g.status is Status.ONGOING:
            w = gen_guess(model, g.turns)
            g.guess(w if len(w) == 5 else "zzzzz")
        games.append(g)
    wins = [g for g in games if g.won]
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": sum(is_valid(t.guess) for g in games for t in g.turns) / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def main():
    _, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    print(f"[anchor] ckpt={CKPT} mode={ANCHOR}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()
    mt, mv = metrics(model, TEST), metrics(model, VAL)
    print(f"  TEST  win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  VAL   win {mv['win']:.3f} valid {mv['valid']:.3f}", flush=True)
    print(f"  [ref] pure free-gen 0.281/0.662 ; constrained-greedy 0.436 ; beam-trie 0.55", flush=True)
    print("\n[ANCHOR DONE]", flush=True)


if __name__ == "__main__":
    main()
