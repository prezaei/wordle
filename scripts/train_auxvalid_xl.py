"""Stronger base (driver, not committed): xl (98M) + auxiliary trie-validity loss — stack the two
things that actually moved the base (scale + the aux loss, which gave +3.4 at 50M -> 0.436).

Wall-clock capped + best-by-held-out checkpointing so an MPS hang can't cost the result.
Best -> runs/sft_aux_xl.pt.
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
from wordle_slm.engine import Color  # noqa: E402
from wordle_slm.engine.constraints import is_consistent  # noqa: E402
from wordle_slm.model import Tokenizer, WordleGenerator  # noqa: E402
from wordle_slm.model.serialization import encode_completed_game, guess_letter_target_positions  # noqa: E402
from wordle_slm.rl.grpo import eval_win_rate  # noqa: E402
from wordle_slm.rl.rollout import letter_id_tensor, play_game  # noqa: E402
from wordle_slm.sft.pretrain import pretrain_lm, pretrain_words  # noqa: E402
from wordle_slm.sft.train import _batches, load_checkpoint, pad_and_mask, save_checkpoint  # noqa: E402
from wordle_slm.teacher import generate_transcripts  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
XL = ModelConfig.preset("xl")  # 640x20, ~98M
train, held = split(seed=0)
curve = tuple(held[:80])
probe = train_probe(seed=0, size=80)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
LAMBDA = 0.5
LETTERS = "abcdefghijklmnopqrstuvwxyz"
TRIE: dict = {}
for w in load_valid_guesses():
    node = TRIE
    for ch in w:
        node = node.setdefault(ch, {})


def game_mask(turns) -> torch.Tensor:
    """Dense CPU [L,26] trie-validity mask (built once per game; padded+moved to MPS per batch)."""
    seq = encode_completed_game(turns, tok)
    m = torch.zeros((len(seq), 26))
    positions = guess_letter_target_positions(seq, tok)
    for g0 in range(0, len(positions), 5):
        node = TRIE
        for q in positions[g0 : g0 + 5]:
            for ch in node:
                m[q - 1, LETTERS.index(ch)] = 1.0
            node = node.get(tok.id_to_token(seq[q]), {})
    return m


def evaluate(model, secrets) -> dict:
    model.eval()
    games = [play_game(model, tok, s, sample=False, device=DEV) for s in secrets]
    model.train()
    wins = [g for g in games if g.won]
    valid = total = consistent = vc = 0
    for g in games:
        prior = []
        for turn in g.turns:
            total += 1
            v = is_valid(turn.guess)
            valid += int(v)
            if v:
                vc += 1
                consistent += int(is_consistent(turn.guess, prior))
            prior.append(turn)
    return {"win": len(wins) / len(games), "valid": valid / total,
            "consistent": consistent / vc if vc else 0.0,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


print(f"[1/4] deep spell warm-up ({XL.estimated_params() / 1e6:.0f}M, depth {XL.n_layers}) …", flush=True)
model = WordleGenerator(XL, tok.vocab_size).to(DEV)
pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=12, batch_size=256, device=DEV, seed=0)

print("[2/4] strong teacher (5 passes, 80% InfoMax) …", flush=True)
t0 = time.perf_counter()
games = []
for s in range(5):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
print(f"      {len(games)} games in {time.perf_counter() - t0:.0f}s", flush=True)

print("[3/4] precompute trie-validity masks (dense, CPU) …", flush=True)
aux_mask = [game_mask(g.turns) for g in games]

print(f"[4/4] SFT with aux validity loss (lambda={LAMBDA}), capped + best-checkpointed …", flush=True)
letter_ids = letter_id_tensor(tok, DEV)
opt = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.01)
EPOCHS, BATCH, EVAL_EVERY, CAP_S = 90, 64, 5, 4700  # ~78-min SFT cap (run it long)
steps_per_epoch = (len(games) + BATCH - 1) // BATCH
total_steps, warmup_steps = EPOCHS * steps_per_epoch, 600
PEAK, FLOOR = 2.5e-4, 2e-5


def lr_at(s):
    if s < warmup_steps:
        return PEAK * (s + 1) / warmup_steps
    p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
    return FLOOR + 0.5 * (PEAK - FLOOR) * (1 + math.cos(math.pi * p))


rng = Random(0)
best_win = -1.0
step = 0
t_sft = time.perf_counter()
model.train()
for epoch in range(EPOCHS):
    if time.perf_counter() - t_sft > CAP_S:
        print(f"  [time cap at epoch {epoch}]", flush=True)
        break
    losses, auxes = [], []
    for idx in _batches(len(games), BATCH, rng):
        seqs = [encode_completed_game(games[i].turns, tok) for i in idx]
        input_ids, target_idx, loss_mask = pad_and_mask(seqs, tok, DEV)
        B, L = input_ids.shape
        vmask_cpu = torch.zeros((B, L, 26))  # build on CPU (fast), one transfer to MPS
        for bi, i in enumerate(idx):
            mi = aux_mask[i]
            vmask_cpu[bi, : mi.shape[0]] = mi
        vmask = vmask_cpu.to(DEV)
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        logp = torch.log_softmax(model.forward(input_ids)[:, :, letter_ids], dim=-1)
        nll = -logp.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)
        imit = (nll * loss_mask).sum() / loss_mask.sum()
        valid_mass = (logp.exp() * vmask).sum(-1).clamp_min(1e-9)
        aux = (-(valid_mass.log()) * loss_mask).sum() / loss_mask.sum()
        loss = imit + LAMBDA * aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(imit.detach()))
        auxes.append(float(aux.detach()))
        step += 1
    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve)
        extra = ""
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            pw = eval_win_rate(model, tok, probe, device=DEV)
            extra = f"  probe={pw:.3f} gap={pw - m['win']:+.3f}"
        print(f"  epoch {epoch:>2}  imit={statistics.mean(losses):.3f} aux={statistics.mean(auxes):.3f}  win={m['win']:.3f} valid={m['valid']:.3f} consistent={m['consistent']:.3f} avg={m['avg']:.2f}{extra}", flush=True)
        if m["win"] > best_win:
            best_win = m["win"]
            save_checkpoint("runs/sft_aux_xl.pt", model, opt, epoch, SFTConfig())

print(f"\n=== AUX-XL milestone: full held-out ({len(held)}) — 98M + aux loss ===", flush=True)
b = WordleGenerator(XL, tok.vocab_size).to(DEV)
load_checkpoint("runs/sft_aux_xl.pt", b)
f = evaluate(b, tuple(held))
pw = eval_win_rate(b, tok, train_probe(seed=0, size=128), device=DEV)
print(f"  held-out win : {f['win']:.3f}  ({int(round(f['win'] * len(held)))}/{len(held)})   [50M+aux was 0.436]", flush=True)
print(f"  valid-rate   : {f['valid']:.3f}   consistent : {f['consistent']:.3f}   avg : {f['avg']:.2f}", flush=True)
print(f"  probe(train) : {pw:.3f}   gap : {pw - f['win']:+.3f}", flush=True)
print("\n[AUXXL DONE]", flush=True)
