"""INPUT -> OUTPUT for a CoT inference, showing the <think> (id 34) tokens in the generated stream.

Honest self-context: board-only-style rolling prompt; the model GENERATES <think> words then <GUESS>.
Contrast with infer_show.py (the non-CoT baseline, whose output is just 5 letters, no think tokens).
"""

from __future__ import annotations

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = "mps"
tok = Tokenizer()
THINK = tok.vocab_size  # 34
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def disp(tid):
    if tid == THINK:
        return "💭<think>"
    if tid == tok.guess_id:
        return "🎯<GUESS>"
    return {tok.bos_id: "<BOS>", tok.sep_id: "<SEP>", tok.green_id: "🟩", tok.yellow_id: "🟨", tok.gray_id: "⬜"}.get(tid, tok.id_to_token(tid))


print("[load] runs/cot_50m.pt (a CoT model) …", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_50m.pt", model)
model.eval()

secret = "pride"
print(f"\n================ CoT inference, secret = {secret} ================", flush=True)
g = Game(secret)
seq = [tok.bos_id]
turn_no = 0
while g.status is Status.ONGOING:
    turn_no += 1
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
    out = seq[prompt_len:]
    word = "".join(tok.id_to_token(t) for t in guess[:5])
    g.guess(word if len(word) == 5 else "zzzzz")
    turn = g.turns[-1]
    fb = "❌ INVALID" if turn.feedback is None else "".join(SQ[c] for c in turn.feedback)
    print(f"\n  --- turn {turn_no} ---", flush=True)
    print(f"  INPUT  ids   : {seq[:prompt_len]}", flush=True)
    print(f"  INPUT  tokens: {' '.join(disp(t) for t in seq[:prompt_len])}", flush=True)
    print(f"  OUTPUT ids   : {out}", flush=True)
    print(f"  OUTPUT tokens: {' '.join(disp(t) for t in out)}    <- the 34's are the <think> CoT tokens", flush=True)
    print(f"  committed    : '{word}'  {fb}", flush=True)
    seq += _fb(turn) + [tok.sep_id]
print(f"\n  => {g.status.value} in {g.guesses_used}", flush=True)
print("\n[CIO DONE]", flush=True)
