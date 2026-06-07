"""R2 measurement: quantify FM-1 selection bias on cot_eph_aux.pt / dpo.pt.

Replicates VERBATIM the greedy ephemeral-CoT inference from scripts/cot_ephemeral_aux.py
(board_only, ALLOWED_GEN, play, evaluate, CFG 512x16 vocab35, load_checkpoint). Computes
greedy 6-row win on held[:96] (SELECTED), held[96:] (NEVER-SELECTED ~honest), and full held.
Also instruments play() to capture committed guesses + emitted <think> candidate words for the
FM-2 inference-exploitation check (held-out words emitted that are NOT the current secret).
"""

from __future__ import annotations

import statistics
import sys

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
# Verbatim CFG from cot_ephemeral_aux.py / dpo_commit.py
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)


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
def play(model, secret):
    """VERBATIM from cot_ephemeral_aux.py play()."""
    g = Game(secret)
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)  # NO think from past turns — ephemeral
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
def play_traced(model, secret):
    """play() with capture of every emitted <think> candidate word and committed guesses.
    Parses the same token stream the model autoregressively produces. Returns (game, think_words,
    committed_words). A think candidate = 5 letters following a THINK token; committed guess =
    5 letters following <GUESS>."""
    g = Game(secret)
    all_think: list[str] = []
    all_commit: list[str] = []
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)
        prompt_len = len(seq)
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
        # parse generated suffix for think candidates
        gen = seq[prompt_len:]
        i = 0
        while i < len(gen):
            t = gen[i]
            if t == THINK:
                letts = []
                j = i + 1
                while j < len(gen) and gen[j] in LETTER_SET and len(letts) < 5:
                    letts.append(gen[j])
                    j += 1
                if len(letts) == 5:
                    all_think.append("".join(tok.id_to_token(x) for x in letts))
                i = j
            else:
                i += 1
        committed_word = word if len(word) == 5 else "zzzzz"
        all_commit.append(committed_word)
        g.guess(committed_word)
    return g, all_think, all_commit


def evaluate(model, secrets):
    """VERBATIM from cot_ephemeral_aux.py evaluate()."""
    model.eval()
    games = [play(model, s) for s in secrets]
    wins = [gg for gg in games if gg.won]
    v = sum(is_valid(t.guess) for gg in games for t in gg.turns)
    n = sum(len(gg.turns) for gg in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(gg.guesses_used for gg in wins) if wins else float("nan")}


def win_on(model, secrets):
    model.eval()
    games = [play(model, s) for s in secrets]
    wins = sum(1 for gg in games if gg.won)
    return wins / len(games), wins, len(games)


def main():
    _, held = split(seed=0)
    held = list(held)
    selected = tuple(held[:96])
    never = tuple(held[96:])
    full = tuple(held)
    print(f"[split] full held={len(full)}  selected[:96]={len(selected)}  never[96:]={len(never)}", flush=True)

    which = sys.argv[1] if len(sys.argv) > 1 else "runs/cot_eph_aux.pt"
    print(f"\n===== checkpoint {which} =====", flush=True)
    m = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(which, m)
    m.eval()

    w_sel, ws, ns = win_on(m, selected)
    print(f"  win held[:96]  (SELECTED)      = {w_sel:.4f}  ({ws}/{ns})", flush=True)
    w_nev, wn, nn_ = win_on(m, never)
    print(f"  win held[96:]  (NEVER-SELECTED)= {w_nev:.4f}  ({wn}/{nn_})", flush=True)
    w_full, wf, nf = win_on(m, full)
    print(f"  win full held  (REPORTED slice)= {w_full:.4f}  ({wf}/{nf})", flush=True)
    print(f"  SELECTION-BIAS GAP win([:96])-win([96:]) = {w_sel - w_nev:+.4f}  ({(w_sel - w_nev) * 100:+.1f} pts)", flush=True)

    # FM-2 inference-exploitation: 20 held-out secrets, capture emitted think + committed words
    held_set = set(full)
    print("\n  --- FM-2 inference exploitation (first 20 held secrets) ---", flush=True)
    probe = full[:20]
    tot_think = 0
    think_heldout_notsecret = 0
    tot_commit = 0
    commit_heldout_notsecret = 0
    commit_valid = 0
    for s in probe:
        _, thinks, commits = play_traced(m, s)
        for w in thinks:
            tot_think += 1
            if w in held_set and w != s:
                think_heldout_notsecret += 1
        for w in commits:
            tot_commit += 1
            if is_valid(w):
                commit_valid += 1
            if w in held_set and w != s:
                commit_heldout_notsecret += 1
    print(f"  THINK candidates emitted: {tot_think}; that are held-out words != secret: "
          f"{think_heldout_notsecret} ({think_heldout_notsecret / max(1, tot_think) * 100:.1f}%)", flush=True)
    print(f"  COMMITTED guesses: {tot_commit}; valid-dict: {commit_valid}; "
          f"held-out words != secret: {commit_heldout_notsecret} "
          f"({commit_heldout_notsecret / max(1, tot_commit) * 100:.1f}%)", flush=True)
    print(f"  (baseline: held-out is {len(full)}/2315 = {len(full) / 2315 * 100:.1f}% of all answers)", flush=True)
    print(f"\n[R2 DONE {which}]", flush=True)


if __name__ == "__main__":
    main()
