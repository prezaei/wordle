"""BPE-on-wordlist experiment (driver, not committed): test the tokenization hypothesis in OUR repo.

Replaces char-level with a from-scratch BPE trained on the valid word list (no TinyStories, no
external code). ~12M model. Guesses are generated as BPE subword tokens (real letter-chunks) instead
of 5 independent chars — the hypothesis being that this kills the spelling/validity wall that capped
char-level at ~0.66 valid / 0.44 win. Reuses our transformer backbone, engine, and InfoMax teacher.
Pipeline: pretrain (word-list LM) -> SFT (teacher games, action-masked) -> greedy eval on held-out.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter
from random import Random

import torch

logging.disable(logging.CRITICAL)

from wordle_slm.config import ModelConfig  # noqa: E402
from wordle_slm.data import is_valid, load_valid_guesses, split  # noqa: E402
from wordle_slm.engine import Color, Game, Status  # noqa: E402
from wordle_slm.model.transformer import WordleGenerator  # noqa: E402
from wordle_slm.sft.train import load_checkpoint, save_checkpoint  # noqa: E402
from wordle_slm.config import SFTConfig  # noqa: E402
from wordle_slm.teacher import generate_transcripts  # noqa: E402

DEV = "mps"
torch.manual_seed(0)
NUM_MERGES = 400
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}

# --- BPE on the valid word list ---------------------------------------------------------------


def train_bpe(words: list[str], num_merges: int) -> list[tuple[str, str]]:
    seqs = [list(w) for w in words]
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs: Counter = Counter()
        for s in seqs:
            for i in range(len(s) - 1):
                pairs[(s[i], s[i + 1])] += 1
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b))
        for j, s in enumerate(seqs):
            i, out = 0, []
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[j] = out
    return merges


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    s = list(word)
    for a, b in merges:
        i, out = 0, []
        while i < len(s):
            if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(s[i])
                i += 1
        s = out
    return s


print(f"[bpe] training BPE on the valid word list ({NUM_MERGES} merges) …", flush=True)
VALID = list(load_valid_guesses())
merges = train_bpe(VALID, NUM_MERGES)
# vocab: 7 specials + the BPE letter-chunk tokens that actually occur
PAD, BOS, EOS, GUESS, GREEN, YELLOW, GRAY = range(7)
chunk_set = sorted({t for w in VALID for t in encode_word(w, merges)})
chunk_id = {c: 7 + i for i, c in enumerate(chunk_set)}
id_chunk = {i: c for c, i in chunk_id.items()}
VOCAB = 7 + len(chunk_set)
LETTER_IDS = torch.tensor(list(chunk_id.values()), device=DEV)  # the generatable (non-special) tokens
COLOR_ID = {Color.GREEN: GREEN, Color.YELLOW: YELLOW, Color.GRAY: GRAY}
avg_tok = statistics.mean(len(encode_word(w, merges)) for w in VALID)
print(f"      vocab={VOCAB} (7 specials + {len(chunk_set)} chunks), avg {avg_tok:.2f} tokens/word", flush=True)


def word_token_ids(word: str) -> list[int]:
    return [chunk_id[t] for t in encode_word(word, merges)]


def serialize_game(turns) -> tuple[list[int], list[bool]]:
    """(ids, action_mask) for a finished game; action=True on the guess's BPE tokens."""
    ids, mask = [BOS], [False]
    for turn in turns:
        ids.append(GUESS)
        mask.append(False)
        wt = word_token_ids(turn.guess)
        ids += wt
        mask += [True] * len(wt)
        fb = turn.feedback if turn.feedback is not None else [Color.GRAY] * 5
        ids += [COLOR_ID[c] for c in fb]
        mask += [False] * 5
    ids.append(EOS)
    mask.append(False)
    return ids, mask


def prompt_ids(turns) -> list[int]:
    ids = [BOS]
    for turn in turns:
        ids.append(GUESS)
        ids += word_token_ids(turn.guess)
        fb = turn.feedback if turn.feedback is not None else [Color.GRAY] * 5
        ids += [COLOR_ID[c] for c in fb]
    ids.append(GUESS)  # cue to generate the next guess
    return ids


# --- batching + masked LM loss ----------------------------------------------------------------


def pad_batch(seqs: list[list[int]], masks: list[list[bool]]):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    tmask = torch.zeros((len(seqs), L))
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, : len(s)] = torch.tensor(s)
        tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
    return ids.to(DEV), tmask.to(DEV)


def lm_loss(model, ids, tmask):
    logits = model.forward(ids)  # [B,L,V]
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    m = tmask[:, 1:]
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (nll * m).sum() / m.sum().clamp_min(1)


@torch.no_grad()
def bpe_generate(model, prompt: list[int], sample=False, gen=None) -> str:
    model.eval()
    seq = list(prompt)
    letters = ""
    for _ in range(6):  # a 5-letter word is <=5 chunks
        logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
        ll = logits[LETTER_IDS]
        if sample:
            j = int(torch.multinomial(torch.softmax(ll, 0).cpu(), 1, generator=gen))
        else:
            j = int(torch.argmax(ll))
        tid = int(LETTER_IDS[j])
        seq.append(tid)
        letters += id_chunk[tid]
        if len(letters) >= 5:
            break
    return letters[:5]


def play(model, secret, sample=False, gen=None) -> Game:
    g = Game(secret)
    while g.status is Status.ONGOING:
        g.guess(bpe_generate(model, prompt_ids(g.turns), sample, gen))
    return g


def evaluate(model, secrets) -> dict:
    model.eval()
    games = [play(model, s) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


# --- sanity: serialization round-trips to a valid game string --------------------------------
_g = Game("crane")
_g.guess("slate")
_g.guess("crane")
_dec = "".join(id_chunk[i] for i in word_token_ids("slate"))
assert _dec == "slate", f"bpe round-trip failed: {_dec}"
print(f"      round-trip ok: slate -> {encode_word('slate', merges)} -> {_dec}", flush=True)

train, held = split(seed=0)
cfg = ModelConfig.preset("large")  # ~50M (512x16) — match our best char model's capacity
model = WordleGenerator(cfg, VOCAB).to(DEV)
print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, vocab {VOCAB}", flush=True)

# --- pretrain: LM over the valid word list (learn valid BPE-token sequences) -------------------
print("[pretrain] word-list LM …", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
rng = Random(0)
pre_seqs = [([BOS] + word_token_ids(w) + [EOS]) for w in VALID]
pre_masks = [[True] * len(s) for s in pre_seqs]
model.train()
for epoch in range(8):
    order = list(range(len(pre_seqs)))
    rng.shuffle(order)
    losses = []
    for i in range(0, len(order), 512):
        idx = order[i : i + 512]
        ids, tmask = pad_batch([pre_seqs[k] for k in idx], [pre_masks[k] for k in idx])
        loss = lm_loss(model, ids, tmask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    if epoch % 2 == 0:
        print(f"  pretrain epoch {epoch}  loss={statistics.mean(losses):.3f}", flush=True)

# --- SFT on teacher games (action-masked) -----------------------------------------------------
print("[sft] teacher transcripts (5 passes, 80% InfoMax) …", flush=True)
t0 = time.perf_counter()
games = []
for s in range(5):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
ser = [serialize_game(g.turns) for g in games]
print(f"      {len(games)} games in {time.perf_counter() - t0:.0f}s", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
EPOCHS, BATCH = 60, 96
curve = tuple(held[:96])
best = -1.0
model.train()
for epoch in range(EPOCHS):
    order = list(range(len(ser)))
    rng.shuffle(order)
    losses = []
    for i in range(0, len(order), BATCH):
        idx = order[i : i + BATCH]
        ids, tmask = pad_batch([ser[k][0] for k in idx], [ser[k][1] for k in idx])
        loss = lm_loss(model, ids, tmask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve)
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/bpe_50m.pt", model, opt, epoch, SFTConfig())
            flag = "  <- best, saved"
        print(f"  sft epoch {epoch:>2}  loss={statistics.mean(losses):.3f}  win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)

print(f"\n=== BPE-50M milestone: full held-out ({len(held)}) — best checkpoint ===", flush=True)
model = WordleGenerator(cfg, VOCAB).to(DEV)
load_checkpoint("runs/bpe_50m.pt", model)
final = evaluate(model, tuple(held))
print(f"  held-out win : {final['win']:.3f}  ({int(round(final['win'] * len(held)))}/{len(held)})   [char-level was 0.436]", flush=True)
print(f"  valid-rate   : {final['valid']:.3f}   [char-level was 0.66 — KEY]   avg : {final['avg']:.2f}", flush=True)
print("\n=== sample greedy held-out games ===", flush=True)
for s in held[:10]:
    g = play(model, s)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[BPE DONE]", flush=True)
