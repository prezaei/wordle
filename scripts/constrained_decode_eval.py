"""Diagnostic (eval-only, not committed): does the stage-1 model KNOW the words but can't spell them
in free-gen? Compare free-generation vs dictionary-constrained decoding on the SAME weights.

Constrained decoding = at each of the 5 committed-guess letters, mask the model's next-letter logits
to the letters that continue to a real word given the prefix-so-far (the valid-guess trie), then
argmax among those. This is SPELLING-ONLY: it forces a real dictionary word but does NOT tell the
model which words are clue-consistent or the answer (the model still deduces WHICH valid word). So
validity is ~1.0 by construction; the interesting number is WIN — if it jumps, the model knew good
words and free-gen drift was costing wins (knowledge present, expression limited).

Loads runs/cot_eph_aux_fair.pt. Honest held-out: greedy, train-only secrets never involved, eval on
disjoint VAL/TEST. Prints free vs constrained win+valid; writes a sample of constrained boards to the
dashboard progress file.
"""

from __future__ import annotations

import os
import statistics

import torch

from viz_progress import append_epoch
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
TRIE = _valid_trie()  # keyed by 0-25 letter index (== token_id - LETTER_LO)
CKPT = os.environ.get("CD_CKPT", "runs/cot_eph_aux_fair.pt")
PROG = "runs/constrained_eval_progress.jsonl"


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
    """Free-generate the think scratchpad until the model emits <GUESS> (or a cap); returns seq."""
    for _ in range(60):
        logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
        nxt = int(ALLOWED_GEN[int(torch.argmax(logits[ALLOWED_GEN]))])
        seq.append(nxt)
        if nxt == tok.guess_id:
            return seq, True
    seq.append(tok.guess_id)  # force a commit if it never did
    return seq, False


@torch.no_grad()
def play_free(model, secret):
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


@torch.no_grad()
def play_constrained(model, secret):
    """Free think, then the 5 guess letters masked to dictionary-valid continuations (trie)."""
    g = Game(secret)
    while g.status is Status.ONGOING:
        seq, _ = _free_to_commit(model, board_only(g.turns))
        node, guess = TRIE, []
        for _ in range(5):
            allowed = list(node.keys())
            if not allowed:  # dead end (shouldn't happen at depth<5 for length-5 words)
                break
            allowed_tok = torch.tensor([LETTER_LO + i for i in allowed], device=DEV)
            logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
            choice = int(allowed_tok[int(torch.argmax(logits[allowed_tok]))])
            guess.append(choice)
            seq.append(choice)
            node = node[choice - LETTER_LO]
        word = "".join(tok.id_to_token(t) for t in guess[:5])
        g.guess(word if len(word) == 5 else "zzzzz")
    return g


def metrics(games):
    wins = [g for g in games if g.won]
    n = sum(len(g.turns) for g in games)
    return {
        "win": len(wins) / len(games),
        "valid": sum(is_valid(t.guess) for g in games for t in g.turns) / n if n else 0.0,
        "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan"),
    }


def main():
    train, held = split(seed=0)
    val, test = tuple(held[:96]), tuple(held[96:])
    if os.path.exists(PROG):
        os.remove(PROG)
    print(f"[cd] loading {CKPT}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT, model)
    model.eval()

    print("[cd] === FREE-GEN (baseline) ===", flush=True)
    f_val = metrics([play_free(model, s) for s in val])
    f_test = metrics([play_free(model, s) for s in test])
    print(f"  free  VAL : win {f_val['win']:.3f} valid {f_val['valid']:.3f}", flush=True)
    print(f"  free  TEST: win {f_test['win']:.3f} valid {f_test['valid']:.3f} avg {f_test['avg']:.2f}", flush=True)

    print("[cd] === DICTIONARY-CONSTRAINED DECODING (spelling-only) ===", flush=True)
    c_val = metrics([play_constrained(model, s) for s in val])
    c_games_test = [play_constrained(model, s) for s in test]
    c_test = metrics(c_games_test)
    print(f"  cdec  VAL : win {c_val['win']:.3f} valid {c_val['valid']:.3f}", flush=True)
    print(f"  cdec  TEST: win {c_test['win']:.3f} valid {c_test['valid']:.3f} avg {c_test['avg']:.2f}", flush=True)

    append_epoch(PROG, 0, c_test, c_games_test[:12], sample=12, kind="constrained")

    print("\n[cd] === SUMMARY (same weights, honest held-out TEST) ===", flush=True)
    print(f"  free-gen      : win {f_test['win']:.3f}  valid {f_test['valid']:.3f}", flush=True)
    print(f"  constrained   : win {c_test['win']:.3f}  valid {c_test['valid']:.3f}", flush=True)
    print(f"  delta (cdec-free): win {c_test['win'] - f_test['win']:+.3f}  valid {c_test['valid'] - f_test['valid']:+.3f}", flush=True)
    print("\n[CD DONE]", flush=True)


if __name__ == "__main__":
    main()
