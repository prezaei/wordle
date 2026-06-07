"""Info-gain expert iteration (Design B: constraint is a TRAINING WHEEL; inference is pure free-gen).

The remaining wall is deduction quality. This trains it with a DENSE info-gain signal, without any
inference crutch:
  1. roll the model out UNDER constrained decoding on TRAIN secrets (the training wheel — makes
     rollouts valid and the winning trajectories reachable),
  2. score each turn by REALIZED info-gain = how much that guess shrank the consistent-secret set
     (computed over TRAIN answers only — honest; answer-hood stays train-only),
  3. keep the high-info-gain (>= halved the set) or winning turns, and SFT the FREE-GEN model to
     reproduce them (+ base teacher for anti-forgetting + the trie-validity aux),
  4. EVAL IS PURE FREE-GEN (no constraint) — the honest held-out number. Best-by win+valid. Iterate.

The constraint never touches inference. Loads runs/cot_eph_aux_fair.pt -> runs/cot_eph_aux_infogain.pt.
Env: IG_IN/IG_OUT/IG_ROUNDS/IG_CHUNK/IG_EPOCHS/IG_SEEDS. Writes runs/infogain_progress.jsonl.
"""

from __future__ import annotations

import math
import os
import statistics
import traceback
from random import Random

import torch

from viz_progress import append_epoch
from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_valid_guesses, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.engine.scoring import score
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
VALID = load_valid_guesses()
TRIE = _valid_trie()  # keyed by 0-25 letter index (== token_id - LETTER_LO)
K_CANDS = 3
AUX_LAMBDA = 1.0
KEEP_IG = math.log(2)  # keep a turn if its guess at least HALVED the candidate set (or the game won)

CKPT_IN = os.environ.get("IG_IN", "runs/cot_eph_aux_fair.pt")
CKPT_OUT = os.environ.get("IG_OUT", "runs/cot_eph_aux_infogain.pt")
PROG = "runs/infogain_progress.jsonl"
ROUNDS = int(os.environ.get("IG_ROUNDS", "5"))
CHUNK = int(os.environ.get("IG_CHUNK", "300"))
EPOCHS = int(os.environ.get("IG_EPOCHS", "2"))
SEEDS = int(os.environ.get("IG_SEEDS", "2"))
ROLLOUT = os.environ.get("IG_ROLLOUT", "constrained")  # "constrained" (Design B wheel) | "free" (pure STaR)
BATCH = 128


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def pick_cands(history, guess, rng):
    cands = [guess]
    for _ in range(300):
        if len(cands) >= K_CANDS:
            break
        w = rng.choice(VALID)
        if w != guess and w not in cands and is_consistent(w, history):
            cands.append(w)
    rng.shuffle(cands)
    return cands


def turn_example(history, guess, rng):
    ids = board_only(history)
    mask = [False] * len(ids)
    for c in pick_cands(history, guess, rng):
        ids.append(THINK)
        mask.append(True)
        ids += _letters(c)
        mask += [True] * 5
    ids.append(tok.guess_id)
    mask.append(True)
    ids += _letters(guess)
    mask += [True] * 5
    ids.append(tok.eos_id)
    mask.append(False)
    return ids, mask


def game_examples(game, rng):
    return [turn_example(game.turns[:k], turn.guess, rng) for k, turn in enumerate(game.turns)]


def cot_valid_mask(seqs):
    L = max(len(s) for s in seqs)
    vmask = torch.zeros((len(seqs), L, 26))
    for i, seq in enumerate(seqs):
        for t, tokid in enumerate(seq):
            if tokid not in WORD_STARTS or t + 5 >= len(seq):
                continue
            node = TRIE
            for j in range(5):
                for child in node:
                    vmask[i, t + j, child] = 1.0
                node = node.get(seq[t + 1 + j] - LETTER_LO, {})
    return vmask


@torch.no_grad()
def play(model, secret):
    """PURE free-gen greedy rollout — the honest inference path (no constraint)."""
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
    """TRAINING-ONLY rollout: free think, then 5 guess letters masked to dictionary-valid words."""
    g = Game(secret)
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)
        committed = False
        for _ in range(60):
            if committed:
                break
            nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
            seq.append(nxt)
            if nxt == tok.guess_id:
                committed = True
        if not committed:
            seq.append(tok.guess_id)
        node, guess = TRIE, []
        for _ in range(5):
            allowed = list(node.keys())
            if not allowed:
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


def evaluate(model, secrets):
    model.eval()
    return metrics([play(model, s) for s in secrets])  # PURE FREE-GEN (Design B honest number)


def infogain_examples(model, secrets, rng, pool):
    """Constrained rollouts -> keep turns that shrank the train-answer set a lot (or won). Dense
    info-gain selection. Returns (examples, kept, total_turns, mean_infogain)."""
    model.eval()
    roll = play_constrained if ROLLOUT == "constrained" else play  # Design B wheel vs pure free-gen
    exs, kept, total, ig_sum = [], 0, 0, 0.0
    for s in secrets:
        g = roll(model, s)
        cands = [w for w in pool]  # possible secrets given no clues yet (honest: train answers)
        for k, turn in enumerate(g.turns):
            total += 1
            before = len(cands)
            if turn.valid and turn.feedback is not None:
                cands = [w for w in cands if score(turn.guess, w) == turn.feedback]
            ig = math.log(before) - math.log(max(1, len(cands))) if before > 1 else 0.0
            ig_sum += ig
            if g.won or ig >= KEEP_IG:  # high-info-gain or winning-trajectory turns
                exs.append(turn_example(g.turns[:k], turn.guess, rng))
                kept += 1
    return exs, kept, total, (ig_sum / total if total else 0.0)


def train_on(model, opt, exs, epochs, rng):
    model.train()
    last = float("nan")
    for _ in range(epochs):
        for idx in _batches(len(exs), BATCH, rng):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            vmask = cot_valid_mask([s for s, _ in bs]).to(DEV)
            ids, tmask = ids.to(DEV), tmask.to(DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            imit = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
            vm = vmask[:, :-1]
            aux_pos = (vm.sum(-1) > 0).float() * tmask[:, 1:]
            valid_mass = (logp_let.exp() * vm).sum(-1).clamp_min(1e-9)
            aux = (-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
            loss = imit + AUX_LAMBDA * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.detach())
    return last


def main():
    train, held = split(seed=0)
    safe_openers = tuple(o for o in OPENERS if o not in set(held))
    val, test, viz = tuple(held[:96]), tuple(held[96:]), tuple(held[:24])
    pool = tuple(train)  # info-gain candidate universe = TRAIN answers (honest; answer-hood train-only)
    if os.path.exists(PROG):
        os.remove(PROG)
    print(f"[ig] in={CKPT_IN} out={CKPT_OUT} rounds={ROUNDS} chunk={CHUNK} epochs={EPOCHS} seeds={SEEDS}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT_IN, model)

    games0 = []
    for s in range(SEEDS):
        games0 += [
            tr.game
            for tr in generate_transcripts(
                tuple(train), weak_frac=0.5, openers=safe_openers, seed=s, valid_pool=VALID, answer_pool=tuple(train)
            )
        ]
    rng = Random(0)
    base_exs = [e for g in games0 for e in game_examples(g, rng)]
    print(f"[ig] base teacher examples: {len(base_exs)} (from {len(games0)} games)", flush=True)

    b = evaluate(model, viz)
    print(f"[ig] stage-1 free-gen baseline: viz_win={b['win']:.3f} valid={b['valid']:.3f}", flush=True)
    append_epoch(PROG, 0, b, [play(model, s) for s in viz], sample=12, kind="infogain")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    kept_all: list = []
    bm = evaluate(model, val)
    best = bm["win"] + bm["valid"]
    save_checkpoint(CKPT_OUT, model, opt, 0, SFTConfig())
    print(f"[ig] baseline free-gen VAL win={bm['win']:.3f} valid={bm['valid']:.3f} (saved {CKPT_OUT})", flush=True)

    for rnd in range(1, ROUNDS + 1):
        try:
            secs = [train[(rnd * CHUNK + i) % len(train)] for i in range(CHUNK)]
            new_exs, kept, total, mean_ig = infogain_examples(model, secs, rng, pool)
            kept_all += new_exs
            loss = train_on(model, opt, base_exs + kept_all, EPOCHS, rng)
            vg = [play(model, s) for s in viz]
            vm = metrics(vg)
            append_epoch(PROG, rnd, vm, vg, sample=12, kind="infogain")
            valw = evaluate(model, val)
            flag = ""
            if valw["win"] + valw["valid"] > best:
                best = valw["win"] + valw["valid"]
                save_checkpoint(CKPT_OUT, model, opt, rnd, SFTConfig())
                flag = "  <- best, saved"
            print(
                f"[round {rnd}] kept {kept}/{total} turns (mean_ig={mean_ig:.2f} nats, total kept {len(kept_all)})  "
                f"loss={loss:.3f}  free viz_win={vm['win']:.3f} viz_valid={vm['valid']:.3f}  "
                f"VAL_win={valw['win']:.3f} VAL_valid={valw['valid']:.3f}{flag}",
                flush=True,
            )
        except Exception:
            print(f"[round {rnd}] ERROR:\n{traceback.format_exc()}", flush=True)

    print("\n=== Info-gain XIT final (best by free-gen win+valid), PURE FREE-GEN held-out ===", flush=True)
    final = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT_OUT, final)
    ft, fv = evaluate(final, test), evaluate(final, val)
    print(f"  TEST  free-gen (HONEST) : win {ft['win']:.3f} valid {ft['valid']:.3f} avg {ft['avg']:.2f}", flush=True)
    print(f"  VAL   free-gen          : win {fv['win']:.3f} valid {fv['valid']:.3f}", flush=True)
    print("  [ref] stage-1 free-gen TEST 0.281/0.662", flush=True)
    print("\n[IG DONE]", flush=True)


if __name__ == "__main__":
    main()
