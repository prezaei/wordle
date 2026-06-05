"""Show INPUT -> OUTPUT for a normal (SFT, free-generation) inference — char-50M+aux, runs/sft_aux.pt.

No RL, no CoT. The standard model: it reads the §5.2 board prompt (past guesses + color feedback,
ending in <GUESS>) and GENERATES exactly 5 letter tokens; the engine then validates the word. We print
the literal input tokens the model sees and the 5 letters it emits, per turn.
"""

from __future__ import annotations

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Game, Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.model.serialization import build_prompt, decode_word
from wordle_slm.rl.rollout import letter_id_tensor
from wordle_slm.sft.train import load_checkpoint

DEV = "mps"
tok = Tokenizer()
LARGE = ModelConfig.preset("large")  # 512x16 ~50M, vocab 34
SQ = {"<green>": "🟩", "<yellow>": "🟨", "<gray>": "⬜"}


def show(tid: int) -> str:
    t = tok.id_to_token(tid)
    return SQ.get(t, t)  # color tokens -> squares, everything else verbatim (<BOS>/<GUESS>/letters)


print("[load] runs/sft_aux.pt (char-50M+aux, the honest best 0.436) …", flush=True)
model = WordleGenerator(LARGE, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_aux.pt", model)
model.eval()
letter_ids = letter_id_tensor(tok, DEV)

_, held = split(seed=0)
for secret in ["pride", "sleek"]:
    print(f"\n================ secret = {secret} ================", flush=True)
    g = Game(secret)
    turn_no = 0
    while g.status is Status.ONGOING:
        turn_no += 1
        prompt = build_prompt(g.turns, tok)  # ends in <GUESS>
        out5 = model.generate(torch.tensor(prompt, device=DEV), letter_ids, sample=False)
        word = decode_word(out5.tolist(), tok)
        g.guess(word)
        turn = g.turns[-1]
        fb = "❌ INVALID (not a word)" if turn.feedback is None else "".join(SQ[f"<{c.name.lower()}>"] for c in turn.feedback)
        print(f"\n  --- turn {turn_no} ---", flush=True)
        print(f"  INPUT  ids   : {prompt}", flush=True)
        print(f"  INPUT  tokens: {' '.join(show(t) for t in prompt)}", flush=True)
        print(f"  OUTPUT ids   : {out5.tolist()}   (restricted to the 26 letters)", flush=True)
        print(f"  OUTPUT word  : '{word}'   valid={is_valid(word)}", flush=True)
        print(f"  feedback     : {fb}", flush=True)
    print(f"\n  => {g.status.value} in {g.guesses_used}", flush=True)
print("\n[INFER DONE]", flush=True)
