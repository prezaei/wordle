"""CoT prototype (driver, not committed): does reasoning surface the latent pass@10=0.79 capability?

HONEST — no inference-time rules. The model generates a short reasoning trace (a few consistent
candidate words) then commits a guess, learned from teacher traces. At inference it reasons entirely
in its own weights (no consistency filter, no candidate list). Clean A/B from the same pretrained
init: no-CoT (board->guess) vs CoT (board-> <think> cands <guess> word). Held-out eval (no
memorization: train on train answers, eval on held). The teacher uses the consistency filter only to
build TRAINING traces (same legitimacy as our InfoMax-teacher SFT); the model never sees it at test.
"""

from __future__ import annotations

import copy
import statistics
from random import Random

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_answers, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import consistent_candidates
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft import pretrain_lm, pretrain_words
from wordle_slm.sft.train import _batches
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size  # new token id (34); model vocab = 35
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=384, n_layers=8, n_heads=6, d_ff=1536, context_len=256, dropout=0.1)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)  # CoT may emit think/guess
ANSWERS = load_answers()
K_CANDS = 3


def _letters(word: str) -> list[int]:
    return [tok.token_to_id(c) for c in word]


def _fb(turn) -> list[int]:
    if turn.feedback is None:
        return [tok.gray_id] * 5
    return [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def pick_cands(history, guess: str, rng: Random) -> list[str]:
    cons = list(consistent_candidates(history, ANSWERS))
    others = [w for w in cons if w != guess]
    rng.shuffle(others)
    cands = [guess] + others[: K_CANDS - 1]
    rng.shuffle(cands)
    return cands


def serialize(game: Game, cot: bool, rng: Random) -> tuple[list[int], list[bool]]:
    ids, mask = [tok.bos_id], [False]
    for k, turn in enumerate(game.turns):
        if cot:
            for c in pick_cands(game.turns[:k], turn.guess, rng):
                ids.append(THINK)
                mask.append(True)  # learn to emit the think marker + the candidate
                for t in _letters(c):
                    ids.append(t)
                    mask.append(True)
            ids.append(tok.guess_id)
            mask.append(True)  # in CoT the model decides when to commit
        else:
            ids.append(tok.guess_id)
            mask.append(False)  # baseline: <GUESS> is the env cue
        for t in _letters(turn.guess):
            ids.append(t)
            mask.append(True)
        for t in _fb(turn):
            ids.append(t)
            mask.append(False)
        ids.append(tok.sep_id)
        mask.append(False)
    ids.append(tok.eos_id)
    mask.append(False)
    return ids, mask


def cot_prompt(turns, rng: Random) -> list[int]:
    ids = [tok.bos_id]
    for k, turn in enumerate(turns):
        for c in pick_cands(turns[:k], turn.guess, rng):
            ids += [THINK, *_letters(c)]
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids  # model generates from here: <think> ... <GUESS> word


@torch.no_grad()
def cot_play(model, secret) -> Game:
    g = Game(secret)
    rng = Random(0)
    while g.status is Status.ONGOING:
        seq = cot_prompt(g.turns, rng)
        guess, committed = [], False
        for _ in range(40):
            logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
            nxt = int(ALLOWED_GEN[int(torch.argmax(logits[ALLOWED_GEN]))])
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
def plain_play(model, secret) -> Game:
    from wordle_slm.model.serialization import build_prompt, decode_word
    lids = torch.tensor(LETTER_IDS, device=DEV)
    g = Game(secret)
    while g.status is Status.ONGOING:
        prompt = torch.tensor(build_prompt(g.turns, tok), device=DEV)
        g.guess(decode_word(model.generate(prompt, lids, sample=False).tolist(), tok))
    return g


def evaluate(model, secrets, cot: bool) -> dict:
    model.eval()
    games = [(cot_play(model, s) if cot else plain_play(model, s)) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


train, held = split(seed=0)
eval_secrets = tuple(held[:200])
print(f"[pretrain] shared spell warm-up ({CFG.estimated_params(VOCAB) / 1e6:.0f}M, vocab {VOCAB}) …", flush=True)
base_init = WordleGenerator(CFG, VOCAB).to(DEV)
pretrain_lm(base_init, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=8, batch_size=256, device=DEV, seed=0)

print("[data] teacher games + serialize (no-CoT and CoT) …", flush=True)
games = []
for s in range(5):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
rng = Random(0)
plain_seqs = [serialize(g, cot=False, rng=rng) for g in games]
cot_seqs = [serialize(g, cot=True, rng=rng) for g in games]
print(f"      {len(games)} games; plain len~{len(plain_seqs[0][0])}, cot len~{len(cot_seqs[0][0])}", flush=True)


def train_one(model, seqs, label, cot) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    rng2 = Random(0)
    EPOCHS, BATCH = 25, 96
    model.train()
    for epoch in range(EPOCHS):
        for idx in _batches(len(seqs), BATCH, rng2):
            bs = [seqs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            ids, tmask = ids.to(DEV), tmask.to(DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            loss = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            m = evaluate(model, eval_secrets[:96], cot)
            print(f"  [{label}] epoch {epoch:>2}  loss={float(loss):.3f}  win={m['win']:.3f} valid={m['valid']:.3f}", flush=True)
        model.train()
    return evaluate(model, eval_secrets, cot)


print("\n[A] no-CoT baseline …", flush=True)
base = train_one(copy.deepcopy(base_init), plain_seqs, "plain", cot=False)
print("\n[B] CoT …", flush=True)
ct = train_one(copy.deepcopy(base_init), cot_seqs, "cot", cot=True)

print("\n=== CoT A/B (held-out 200) ===", flush=True)
print(f"  no-CoT : win {base['win']:.3f}  valid {base['valid']:.3f}  avg {base['avg']:.2f}", flush=True)
print(f"  CoT    : win {ct['win']:.3f}  valid {ct['valid']:.3f}  avg {ct['avg']:.2f}", flush=True)
print(f"  => delta (CoT - no-CoT) = {ct['win'] - base['win']:+.3f}", flush=True)
print("\n[COT DONE]", flush=True)
