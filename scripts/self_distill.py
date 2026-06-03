"""Self-distillation (driver script, not committed): bank the +19-pt spelling gain into the model.

The model already wins 58% with dictionary-constrained (beam+dict) decoding but only 39% greedy --
the gap is gibberish its greedy emits. Fix: generate the model's OWN beam+dict games (always valid
words) and SFT on them, so plain greedy learns to emit those real words. The dictionary touches only
the TRAINING DATA, never inference -> the final model free-generates with no word list at decode.

Mixes beam+dict self-play (spell-the-valid-word target) with InfoMax-on-answers (keep strategy).
Warm-starts from runs/sft_xl.pt. Saves best-by-greedy-held-out to runs/sft_distill.pt.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from random import Random

import torch

logging.disable(logging.CRITICAL)

from wordle_slm.config import ModelConfig, SFTConfig  # noqa: E402
from wordle_slm.data import is_valid, load_valid_guesses, split, train_probe  # noqa: E402
from wordle_slm.engine import Color, Game, Status  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.model.serialization import build_prompt, decode_word  # noqa: E402
from wordle_slm.rl.grpo import eval_win_rate  # noqa: E402
from wordle_slm.rl.rollout import letter_id_tensor, play_game  # noqa: E402
from wordle_slm.sft.train import _batches, load_checkpoint, make_batch, save_checkpoint, sft_loss  # noqa: E402
from wordle_slm.teacher import generate_transcripts  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
XL = ModelConfig(d_model=512, n_layers=8, n_heads=8, d_ff=2048)
train, held = split(seed=0)
curve = tuple(held[:96])
probe = train_probe(seed=0, size=96)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
WIDTH = 8
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
LETTERS = "abcdefghijklmnopqrstuvwxyz"
TRIE: dict = {}
for w in load_valid_guesses():
    node = TRIE
    for ch in w:
        node = node.setdefault(ch, {})


@torch.no_grad()
def beam_dict_word(model, prompt, letter_ids) -> str:
    beams = [(0.0, [], TRIE)]
    for _ in range(5):
        seqs = [prompt + [int(letter_ids[i]) for i in ls] for _, ls, _ in beams]
        logp = torch.log_softmax(model.forward(torch.tensor(seqs, device=DEV))[:, -1, letter_ids], dim=-1)
        cand = []
        for b, (cum, ls, node) in enumerate(beams):
            for ch, child in node.items():
                j = LETTERS.index(ch)
                cand.append((cum + float(logp[b, j]), ls + [j], child))
        cand.sort(key=lambda x: x[0], reverse=True)
        beams = cand[:WIDTH]
    return decode_word([int(letter_ids[i]) for i in beams[0][1]], tok)


def beam_dict_game(model, secret, letter_ids) -> Game:
    g = Game(secret)
    while g.status is Status.ONGOING:
        g.guess(beam_dict_word(model, build_prompt(g.turns, tok), letter_ids))
    return g


def greedy_eval(model, secrets) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [x for x in games if x.won]
    v = sum(is_valid(t.guess) for x in games for t in x.turns)
    n = sum(len(x.turns) for x in games)
    return {"win": len(wins) / len(games), "valid": v / n, "avg": statistics.mean(x.guesses_used for x in wins) if wins else float("nan")}


print("[load] warm-start from runs/sft_xl.pt", flush=True)
model = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_xl.pt", model)
letter_ids = letter_id_tensor(tok, DEV)

print("[gen] beam+dict self-play on 1200 train secrets …", flush=True)
t0 = time.perf_counter()
model.eval()
rng = Random(0)
selfplay_secrets = rng.sample(list(train), 1200)
selfplay = []
for i, s in enumerate(selfplay_secrets):
    selfplay.append(beam_dict_game(model, s, letter_ids))
    if (i + 1) % 200 == 0:
        print(f"      {i + 1}/1200 self-play games  ({time.perf_counter() - t0:.0f}s)", flush=True)
wins = sum(g.won for g in selfplay)
print(f"      {len(selfplay)} self-play games, teacher win-rate {wins / len(selfplay):.3f}, {time.perf_counter() - t0:.0f}s", flush=True)

print("[gen] InfoMax-on-answers (2 passes, for strategy) …", flush=True)
strat = []
for sd in range(2):
    strat += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=sd)]
games = selfplay + strat
print(f"      mix: {len(selfplay)} self-play + {len(strat)} InfoMax = {len(games)} games", flush=True)

print("[sft] warm-start distill (cosine) …", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
EPOCHS, BATCH, EVAL_EVERY = 30, 128, 3
steps_per_epoch = (len(games) + BATCH - 1) // BATCH
total = EPOCHS * steps_per_epoch
rng2 = Random(0)
best = -1.0
step = 0
model.train()
for epoch in range(EPOCHS):
    losses = []
    for idx in _batches(len(games), BATCH, rng2):
        for grp in opt.param_groups:
            grp["lr"] = 1.5e-5 + 0.5 * (1.5e-4 - 1.5e-5) * (1 + math.cos(math.pi * step / total))
        ids, tgt, mask = make_batch([games[i] for i in idx], tok, DEV)
        loss = sft_loss(model, ids, tgt, mask, letter_ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        step += 1
    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        m = greedy_eval(model, curve)
        extra = ""
        if epoch % 9 == 0 or epoch == EPOCHS - 1:
            extra = f"  probe={eval_win_rate(model, tok, probe, device=DEV):.3f}"
        print(f"  epoch {epoch:>2}  loss={statistics.mean(losses):.3f}  GREEDY win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{extra}", flush=True)
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/sft_distill.pt", model, opt, epoch, SFTConfig())

print(f"\n=== distill milestone: GREEDY full held-out ({len(held)}) ===", flush=True)
b = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_distill.pt", b)
f = greedy_eval(b, tuple(held))
print(f"  greedy win   : {f['win']:.3f}  ({int(round(f['win'] * len(held)))}/{len(held)})   [was 0.391 before distill]", flush=True)
print(f"  greedy valid : {f['valid']:.3f}   [was 0.658]", flush=True)
print(f"  greedy avg   : {f['avg']:.2f}", flush=True)
print("\n=== sample greedy games (NO dictionary at inference) ===", flush=True)
b.eval()
for s in held[:12]:
    g = play_game(b, tok, s, sample=False, device=DEV)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[DISTILL DONE]", flush=True)
