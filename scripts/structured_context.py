"""Cheap diagnostic (driver, not committed): does explicit game-state management help held-out?

Clean A/B from the SAME pretrained init: SFT one model on the RAW board (our current format) and one
on a board + an explicit DERIVED-STATE block (greens-by-position / present / absent) inserted before
each guess. Same size/epochs/seed/teacher/aux-loss — the only difference is the state block. If the
state model wins more held-out, context management is a real lever; if not, the wall is enumeration.
"""

from __future__ import annotations

import copy
import statistics
from random import Random

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.model.serialization import decode_word, encode_completed_game
from wordle_slm.rl.rollout import letter_id_tensor, play_game
from wordle_slm.sft import pad_and_mask, pretrain_lm, pretrain_words, sft_loss, valid_continuation_mask
from wordle_slm.sft.train import _batches
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
CFG = ModelConfig(d_model=384, n_layers=8, n_heads=6, d_ff=1536, context_len=256, dropout=0.1)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}


def _fb_ids(turn) -> list[int]:
    if turn.feedback is None:
        return [tok.gray_id] * 5
    return [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def derive_state(turns) -> tuple[dict[int, str], set[str], set[str]]:
    greens: dict[int, str] = {}
    present: set[str] = set()
    grays: set[str] = set()
    for turn in turns:
        if not turn.valid or turn.feedback is None:
            continue
        for i, c in enumerate(turn.feedback):
            ch = turn.guess[i]
            if c is Color.GREEN:
                greens[i] = ch
                present.add(ch)
            elif c is Color.YELLOW:
                present.add(ch)
            else:
                grays.add(ch)
    return greens, present, grays - present  # absent = gray and never present/green


def state_block(turns_so_far) -> list[int]:
    greens, present, absent = derive_state(turns_so_far)
    ids = [tok.sep_id, tok.token_to_id("<green>")]
    for i in range(5):
        ids.append(tok.token_to_id(greens[i]) if i in greens else tok.pad_id)  # blank slot = PAD
    ids.append(tok.token_to_id("<yellow>"))
    ids += [tok.token_to_id(ch) for ch in sorted(present)]
    ids.append(tok.token_to_id("<gray>"))
    ids += [tok.token_to_id(ch) for ch in sorted(absent)]
    ids.append(tok.sep_id)
    return ids


def encode_game_state(turns) -> list[int]:
    ids = [tok.bos_id]
    for k, turn in enumerate(turns):
        ids += state_block(turns[:k])
        ids += [tok.guess_id, *tok.encode_letters(turn.guess), *_fb_ids(turn), tok.sep_id]
    return ids + [tok.eos_id]


def prompt_state(turns) -> list[int]:
    ids = [tok.bos_id]
    for k, turn in enumerate(turns):
        ids += state_block(turns[:k])
        ids += [tok.guess_id, *tok.encode_letters(turn.guess), *_fb_ids(turn), tok.sep_id]
    return ids + state_block(turns) + [tok.guess_id]


def play_state(model, secret) -> Game:
    g = Game(secret)
    while g.status is Status.ONGOING:
        prompt = torch.tensor(prompt_state(g.turns), device=DEV)
        g.guess(decode_word(model.generate(prompt, LETTER_IDS, sample=False).tolist(), tok))
    return g


def evaluate(model, secrets, with_state: bool) -> dict:
    model.eval()
    games = [(play_state(model, s) if with_state else play_game(model, tok, s, sample=False, device=DEV)) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


LETTER_IDS = letter_id_tensor(tok, DEV)
train, held = split(seed=0)
eval_secrets = tuple(held[:200])

print(f"[pretrain] shared spell warm-up ({CFG.estimated_params() / 1e6:.0f}M) …", flush=True)
base_init = WordleGenerator(CFG, tok.vocab_size).to(DEV)
pretrain_lm(base_init, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=8, batch_size=256, device=DEV, seed=0)

print("[data] teacher games (5 passes, 80% InfoMax) …", flush=True)
games = []
for s in range(5):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
raw_seqs = [encode_completed_game(g.turns, tok) for g in games]
state_seqs = [encode_game_state(g.turns) for g in games]
print(f"      {len(games)} games; raw len~{len(raw_seqs[0])}, state len~{len(state_seqs[0])}", flush=True)


def train_one(model, seqs, label, with_state) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    rng = Random(0)
    EPOCHS, BATCH = 25, 96
    model.train()
    for epoch in range(EPOCHS):
        for idx in _batches(len(seqs), BATCH, rng):
            bs = [seqs[i] for i in idx]
            ids, tgt, mask = pad_and_mask(bs, tok, DEV)
            vmask = valid_continuation_mask(bs, tok, DEV)
            loss = sft_loss(model, ids, tgt, mask, LETTER_IDS, valid_mask=vmask, aux_lambda=0.5)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            m = evaluate(model, eval_secrets[:96], with_state)
            print(f"  [{label}] epoch {epoch:>2}  win={m['win']:.3f} valid={m['valid']:.3f}", flush=True)
        model.train()
    return evaluate(model, eval_secrets, with_state)


print("\n[A] BASELINE: raw board (current format) …", flush=True)
base = train_one(copy.deepcopy(base_init), raw_seqs, "raw", with_state=False)
print("\n[B] +STATE: explicit derived-constraint block …", flush=True)
st = train_one(copy.deepcopy(base_init), state_seqs, "state", with_state=True)

print("\n=== STRUCTURED-CONTEXT A/B (held-out 200) ===", flush=True)
print(f"  raw board      : win {base['win']:.3f}  valid {base['valid']:.3f}  avg {base['avg']:.2f}", flush=True)
print(f"  + state block  : win {st['win']:.3f}  valid {st['valid']:.3f}  avg {st['avg']:.2f}", flush=True)
print(f"  => delta (state - raw) = {st['win'] - base['win']:+.3f}", flush=True)
print("\n[CTX DONE]", flush=True)
