"""On-policy self-distillation from the constrained decoder (driver, not committed).

The diagnostic showed the stage-1 model KNOWS the words (dictionary-constrained decode: validity 1.0,
TEST win 0.436 vs free-gen 0.281) — free-gen *spelling drift* was the only gap. This run closes it
WITHOUT any inference crutch: roll the model out with **dictionary-constrained decoding** (every guess
a real word, chosen by the model's own preference + deduction), then **SFT the free-gen model to
imitate its own constrained guesses** (+ the per-letter trie-validity aux). The constrained rollout is
a strictly-better version of the model (same deduction, valid spelling), so imitating it pulls
free-gen toward valid-word emission — on-policy distillation. Iterate. The trie is a *training-time*
teacher only; inference stays unaided free-gen.

Honest: train-only secrets, dictionary pools, eval held-out greedy (free-gen) on disjoint VAL/TEST.
Loads runs/cot_eph_aux_fair.pt -> runs/cot_eph_aux_distill.pt. Env: DISTILL_ROUNDS/CHUNK/EPOCHS/
SEEDS/OVERSAMPLE. Writes runs/distill_progress.jsonl for the dashboard.
"""

from __future__ import annotations

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

CKPT_IN = os.environ.get("DISTILL_IN", "runs/cot_eph_aux_fair.pt")
CKPT_OUT = os.environ.get("DISTILL_OUT", "runs/cot_eph_aux_distill.pt")
PROG = "runs/distill_progress.jsonl"
ROUNDS = int(os.environ.get("DISTILL_ROUNDS", "5"))
CHUNK = int(os.environ.get("DISTILL_CHUNK", "300"))
EPOCHS = int(os.environ.get("DISTILL_EPOCHS", "2"))
SEEDS = int(os.environ.get("DISTILL_SEEDS", "2"))
OVERSAMPLE = int(os.environ.get("DISTILL_OVERSAMPLE", "3"))
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
    """Free-gen greedy rollout (the honest inference path)."""
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
    """Free think, then 5 guess letters masked to dictionary-valid continuations (the teacher)."""
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
    return metrics([play(model, s) for s in secrets])  # FREE-GEN eval (the honest number)


def distill_examples(model, secrets, rng):
    """Constrained-decode rollouts -> per-turn SFT examples (every guess is a real word the model chose)."""
    model.eval()
    exs, n_turns = [], 0
    for s in secrets:
        g = play_constrained(model, s)
        for k, turn in enumerate(g.turns):
            if turn.valid:  # constrained guesses are valid by construction; skip any rare dead-end
                exs.append(turn_example(g.turns[:k], turn.guess, rng))
                n_turns += 1
    return exs, n_turns


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
    if os.path.exists(PROG):
        os.remove(PROG)
    print(f"[distill] in={CKPT_IN} out={CKPT_OUT} rounds={ROUNDS} chunk={CHUNK} epochs={EPOCHS} "
          f"seeds={SEEDS} oversample={OVERSAMPLE}", flush=True)
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
    print(f"[distill] base teacher examples: {len(base_exs)} (from {len(games0)} games)", flush=True)

    b = evaluate(model, viz)
    print(f"[distill] stage-1 free-gen baseline: viz_win={b['win']:.3f} valid={b['valid']:.3f}", flush=True)
    append_epoch(PROG, 0, b, [play(model, s) for s in viz], sample=12, kind="distill")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    distilled: list = []
    # select on win + validity (not win alone) so the validity gains actually get banked.
    bm = evaluate(model, val)
    best = bm["win"] + bm["valid"]
    save_checkpoint(CKPT_OUT, model, opt, 0, SFTConfig())
    print(f"[distill] baseline VAL win={bm['win']:.3f} valid={bm['valid']:.3f} score={best:.3f} (saved {CKPT_OUT})", flush=True)

    for rnd in range(1, ROUNDS + 1):
        try:
            secs = [train[(rnd * CHUNK + i) % len(train)] for i in range(CHUNK)]
            new_exs, n = distill_examples(model, secs, rng)
            distilled += new_exs
            loss = train_on(model, opt, base_exs + distilled * OVERSAMPLE, EPOCHS, rng)
            vg = [play(model, s) for s in viz]
            vm = metrics(vg)
            append_epoch(PROG, rnd, vm, vg, sample=12, kind="distill")
            valw = evaluate(model, val)
            flag = ""
            if valw["win"] + valw["valid"] > best:  # combined: bank win+validity gains
                best = valw["win"] + valw["valid"]
                save_checkpoint(CKPT_OUT, model, opt, rnd, SFTConfig())
                flag = "  <- best, saved"
            print(
                f"[round {rnd}] +{len(new_exs)} distill exs (total {len(distilled)})  loss={loss:.3f}  "
                f"free viz_win={vm['win']:.3f} viz_valid={vm['valid']:.3f}  "
                f"VAL_win={valw['win']:.3f} VAL_valid={valw['valid']:.3f}{flag}",
                flush=True,
            )
        except Exception:
            print(f"[round {rnd}] ERROR:\n{traceback.format_exc()}", flush=True)

    print("\n=== Distill final (best by VAL free-gen win), FREE-GEN held-out ===", flush=True)
    final = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT_OUT, final)
    ft, fv = evaluate(final, test), evaluate(final, val)
    print(f"  TEST  free-gen (HONEST) : win {ft['win']:.3f} valid {ft['valid']:.3f} avg {ft['avg']:.2f}", flush=True)
    print(f"  VAL   free-gen          : win {fv['win']:.3f} valid {fv['valid']:.3f}", flush=True)
    print("  [ref] stage-1 free-gen TEST 0.281/0.662 ; constrained-decode TEST 0.436/1.000", flush=True)
    print("\n[DISTILL DONE]", flush=True)


if __name__ == "__main__":
    main()
