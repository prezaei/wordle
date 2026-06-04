"""Full recipe (driver, not committed): real-text (TinyStories) pretrain + BPE, then OUR SFT.

Tests the lever we proved decisive: a real-language pretrain. My own implementation (no oreo code):
byte-level BPE on TinyStories -> pretrain a GPT (our transformer backbone) on TinyStories -> SFT on
our InfoMax-teacher Wordle games (oreo-style text transcript, action-masked) -> greedy eval.
~12M params (oreo's scale). Run with: uv run --with datasets --with tokenizers python scripts/oreo_recipe.py
"""

from __future__ import annotations

import logging
import re
import statistics
import time
from random import Random

import torch

logging.disable(logging.CRITICAL)

from datasets import load_dataset  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from wordle_slm.config import ModelConfig, SFTConfig  # noqa: E402
from wordle_slm.data import is_valid, split  # noqa: E402
from wordle_slm.sft.train import load_checkpoint, save_checkpoint  # noqa: E402
from wordle_slm.engine import Color, Game, Status  # noqa: E402
from wordle_slm.model.transformer import WordleGenerator  # noqa: E402
from wordle_slm.teacher import generate_transcripts  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
VOCAB, BLOCK, N_DOCS = 2048, 256, 10000
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
FB = {Color.GREEN: "G", Color.YELLOW: "Y", Color.GRAY: "B"}

# --- 1. TinyStories + byte-level BPE ----------------------------------------------------------
print(f"[1/5] download {N_DOCS} TinyStories docs + train byte-level BPE (vocab {VOCAB}) …", flush=True)
t0 = time.perf_counter()
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
docs = []
for i, ex in enumerate(ds):
    docs.append(ex["text"])
    if i + 1 >= N_DOCS:
        break
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()  # so tok.decode() restores spaces (Ġ -> ' '); fixes guess parsing
tok.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<pad>", "<bos>", "<eos>"]))
PAD, BOS, EOS = (tok.token_to_id(s) for s in ("<pad>", "<bos>", "<eos>"))
VOCAB = tok.get_vocab_size()
print(f"      vocab={VOCAB}; {len(docs)} docs in {time.perf_counter() - t0:.0f}s", flush=True)

# --- 2. Wordle text transcript + offset-based action mask -------------------------------------


def fb_str(feedback) -> str:
    return "".join(FB[c] for c in feedback) if feedback is not None else "BBBBB"


def _e(s: str) -> list[int]:
    return tok.encode(s).ids  # ByteLevel add_prefix_space -> each segment starts with a space token


def serialize(game: Game) -> tuple[list[int], list[bool]]:
    """Per-segment encoding (oreo-style): exact action mask + identical train/inference boundaries."""
    ids, mask = [BOS], [False]
    for turn in game.turns:
        for seg, act in (("guess", False), (turn.guess, True), ("fb " + fb_str(turn.feedback), False)):
            e = _e(seg)
            ids += e
            mask += [act] * len(e)
    o = _e("win" if game.won else "lose")
    ids += o
    mask += [False] * len(o)
    return ids + [EOS], mask + [False]


def prompt_ids(game: Game) -> list[int]:
    ids = [BOS]
    for turn in game.turns:
        ids += _e("guess") + _e(turn.guess) + _e("fb " + fb_str(turn.feedback))
    return ids + _e("guess")  # cue to generate the next guess word


# sanity: the action tokens of a finished game decode to its guess words
_g = Game("crane")
_g.guess("slate")
_g.guess("crane")
_ids, _m = serialize(_g)
_act = tok.decode([i for i, a in zip(_ids, _m) if a]).replace(" ", "")
assert "slate" in _act and "crane" in _act, f"action mask wrong: {_act!r}"
print(f"      action-mask sanity ok: {_act!r}", flush=True)

# --- 3. Tokenize TinyStories for the LM pretrain ----------------------------------------------
stream: list[int] = []
for d in docs:
    stream += tok.encode(d).ids + [EOS]
stream_t = torch.tensor(stream, dtype=torch.long)
print(f"[2/5] pretrain stream: {len(stream) / 1e6:.2f}M tokens", flush=True)

# --- 4. Model (our transformer backbone, ~12M) + pretrain on TinyStories ----------------------
cfg = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=BLOCK, dropout=0.1)  # ~50M
model = WordleGenerator(cfg, VOCAB).to(DEV)
print(f"[3/5] pretrain GPT ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M) on TinyStories …", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
gen = torch.Generator().manual_seed(0)
PRE_STEPS, PBATCH = 2000, 32
model.train()
for step in range(PRE_STEPS):
    starts = torch.randint(0, len(stream) - BLOCK - 1, (PBATCH,), generator=gen)
    batch = torch.stack([stream_t[s : s + BLOCK + 1] for s in starts]).to(DEV)
    logits = model.forward(batch[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 400 == 0:
        print(f"  pretrain step {step}  loss={float(loss.detach()):.3f}", flush=True)

# --- 5. SFT on teacher games (action-masked), then eval ---------------------------------------
train, held = split(seed=0)
curve = tuple(held[:96])
print("[4/5] teacher games (6 passes, 80% InfoMax) + SFT …", flush=True)
t0 = time.perf_counter()
games = []
for s in range(10):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
ser = [serialize(g) for g in games]
print(f"      {len(games)} games serialized in {time.perf_counter() - t0:.0f}s", flush=True)


def pad_batch(rows):
    L = max(len(s[0]) for s in rows)
    ids = torch.full((len(rows), L), PAD, dtype=torch.long)
    m = torch.zeros((len(rows), L))
    for i, (s, msk) in enumerate(rows):
        ids[i, : len(s)] = torch.tensor(s)
        m[i, : len(msk)] = torch.tensor([float(x) for x in msk])
    return ids.to(DEV), m.to(DEV)


def lm_masked_loss(ids, m):
    logits = model.forward(ids)
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt, mm = ids[:, 1:], m[:, 1:]
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (nll * mm).sum() / mm.sum().clamp_min(1)


@torch.no_grad()
def gen_guess(pids: list[int]) -> str:
    model.eval()
    seq = list(pids)
    for _ in range(6):
        nxt = int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1]))
        seq.append(nxt)
        cont = tok.decode(seq[len(pids):])
        mt = re.match(r"\s*([a-z]+)", cont)
        if mt and len(mt.group(1)) >= 5:
            return mt.group(1)[:5]
        if nxt == EOS:
            break
    mt = re.match(r"\s*([a-z]+)", tok.decode(seq[len(pids):]))
    return (mt.group(1)[:5] if mt else "")[:5]


def play(secret) -> Game:
    g = Game(secret)
    while g.status is Status.ONGOING:
        w = gen_guess(prompt_ids(g))
        g.guess(w if len(w) == 5 else "zzzzz")  # short -> invalid
    return g


def evaluate(secrets) -> dict:
    games_ = [play(s) for s in secrets]
    wins = [g for g in games_ if g.won]
    v = sum(is_valid(t.guess) for g in games_ for t in g.turns)
    n = sum(len(g.turns) for g in games_)
    return {"win": len(wins) / len(games_), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


import math  # noqa: E402

opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
EPOCHS, BATCH = 28, 64
steps_per_epoch = (len(ser) + BATCH - 1) // BATCH
total_steps, warmup = EPOCHS * steps_per_epoch, 400
sched_step = 0
best_held = -1.0
rng = Random(0)
model.train()
for epoch in range(EPOCHS):
    order = list(range(len(ser)))
    rng.shuffle(order)
    losses = []
    for i in range(0, len(order), BATCH):
        lr = 4e-4 * (min(1.0, (sched_step + 1) / warmup) if sched_step < warmup
                     else 0.1 + 0.45 * (1 + math.cos(math.pi * (sched_step - warmup) / (total_steps - warmup))))
        for grp in opt.param_groups:
            grp["lr"] = lr
        ids, m = pad_batch([ser[k] for k in order[i : i + BATCH]])
        loss = lm_masked_loss(ids, m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        sched_step += 1
    if epoch % 3 == 0 or epoch == EPOCHS - 1:
        m = evaluate(curve)  # held-out subsample — never trained on
        flag = ""
        if m["win"] > best_held:
            best_held = m["win"]
            save_checkpoint("runs/recipe_50m.pt", model, opt, epoch, SFTConfig())
            flag = "  <- best held-out, saved"
        print(f"  sft epoch {epoch:>2}  loss={statistics.mean(losses):.3f}  win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)
    else:
        print(f"  sft epoch {epoch:>2}  loss={statistics.mean(losses):.3f}", flush=True)
    model.train()

print("\n[5/5] === honest milestone (best-by-held-out checkpoint) ===", flush=True)
load_checkpoint("runs/recipe_50m.pt", model)  # the best held-out checkpoint, not the last
probe = Random(1).sample(list(train), 200)  # answers the model TRAINED on (= oreo's eval regime)
pr = evaluate(tuple(probe))
print(f"  SEEN/train win   : {pr['win']:.3f}   (oreo evals on this regime — no held-out split)", flush=True)
final = evaluate(tuple(held))
print(f"  held-out win     : {final['win']:.3f}  ({int(round(final['win'] * len(held)))}/{len(held)})   [our strict split; char-50M=0.436]", flush=True)
print(f"  => contamination gap = {pr['win'] - final['win']:+.3f}", flush=True)
print(f"  valid-rate   : {final['valid']:.3f}   avg : {final['avg']:.2f}", flush=True)
print("\n=== sample greedy held-out games ===", flush=True)
for s in held[:10]:
    g = play(s)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[RECIPE DONE]", flush=True)
