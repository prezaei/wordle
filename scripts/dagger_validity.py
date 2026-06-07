"""Validity-targeted DAgger on the fair SFT model (driver, not committed).

Stage 2 of the "make free-gen only emit real words" push. Loads the stage-1 checkpoint (cranked
pretrain + aux), then iteratively:
  1. roll the model out greedily on TRAIN secrets,
  2. find every turn where it emitted an INVALID word,
  3. add a corrective per-turn example whose target is a real dictionary word CONSISTENT with that
     board (rejection-sampled from the full 14,855-word valid-guess list),
  4. re-train (aux-SFT) on the expert teacher data AGGREGATED with the growing corrective set
     (proper DAgger: D <- D ∪ corrected-mistakes).
This directly attacks the conditional invalid-word failure (the model inventing non-words late-game).
Honest: train-only secrets, eval held-out greedy, dictionary pools (knows spelling, not answer-hood).

Env knobs (defaults = full run): DAGGER_IN, DAGGER_OUT, DAGGER_ROUNDS, DAGGER_CHUNK, DAGGER_EPOCHS,
DAGGER_SEEDS. Writes runs/dagger_validity_progress.jsonl for the live dashboard. Functions are
module-level (importable for testing); the run is under main().
"""

from __future__ import annotations

import os
import statistics
import traceback
from random import Random

import torch

from viz_progress import append_epoch  # scripts/ helper: per-round boards -> live dashboard
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
K_CANDS = 3
AUX_LAMBDA = 1.0

CKPT_IN = os.environ.get("DAGGER_IN", "runs/cot_eph_aux_fair.pt")
CKPT_OUT = os.environ.get("DAGGER_OUT", "runs/cot_eph_aux_dagger.pt")
PROG = "runs/dagger_validity_progress.jsonl"
ROUNDS = int(os.environ.get("DAGGER_ROUNDS", "6"))
CHUNK = int(os.environ.get("DAGGER_CHUNK", "400"))
EPOCHS = int(os.environ.get("DAGGER_EPOCHS", "2"))
SEEDS = int(os.environ.get("DAGGER_SEEDS", "3"))
OVERSAMPLE = int(os.environ.get("DAGGER_OVERSAMPLE", "1"))  # replicate corrective exs so they aren't
BATCH = 128                                                 # drowned out by the (much larger) base set


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
    """One per-turn example: board-only history (no loss) + think-cands + committed guess (loss)."""
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
    trie = _valid_trie()
    L = max(len(s) for s in seqs)
    vmask = torch.zeros((len(seqs), L, 26))
    for i, seq in enumerate(seqs):
        for t, tokid in enumerate(seq):
            if tokid not in WORD_STARTS or t + 5 >= len(seq):
                continue
            node = trie
            for j in range(5):
                for child in node:
                    vmask[i, t + j, child] = 1.0
                node = node.get(seq[t + 1 + j] - LETTER_LO, {})
    return vmask


@torch.no_grad()
def play(model, secret):
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
    return metrics([play(model, s) for s in secrets])


def consistent_valid(history, rng):
    """A real dictionary word consistent with the clues so far (rejection sample); None if none found."""
    for _ in range(400):
        w = rng.choice(VALID)
        if is_consistent(w, history):
            return w
    return None


def corrective_examples(model, secrets, rng):
    """Roll the model out; for each INVALID guess, an example targeting a valid consistent word."""
    model.eval()
    exs, n_invalid, n_turns = [], 0, 0
    for s in secrets:
        g = play(model, s)
        for k, turn in enumerate(g.turns):
            n_turns += 1
            if is_valid(turn.guess):
                continue
            n_invalid += 1
            w = consistent_valid(g.turns[:k], rng)
            if w is not None:
                exs.append(turn_example(g.turns[:k], w, rng))
    return exs, n_invalid, n_turns


def train_on(model, opt, exs, epochs, rng):
    """Aux-SFT (imitation + trie-validity) on per-turn examples; identical loss to the fair driver."""
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

    print(f"[dagger] in={CKPT_IN} out={CKPT_OUT} rounds={ROUNDS} chunk={CHUNK} epochs={EPOCHS} seeds={SEEDS}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT_IN, model)

    # Base expert teacher data (same recipe as the fair SFT) — DAgger aggregates corrections onto it.
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
    print(f"[dagger] base teacher examples: {len(base_exs)} (from {len(games0)} games)", flush=True)

    b = evaluate(model, viz)
    print(f"[dagger] stage-1 baseline: viz_win={b['win']:.3f} valid={b['valid']:.3f}", flush=True)
    append_epoch(PROG, 0, b, [play(model, s) for s in viz], sample=12, kind="dagger")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    corrective: list = []
    best = evaluate(model, val)["win"]
    save_checkpoint(CKPT_OUT, model, opt, 0, SFTConfig())  # seed the output with stage-1 weights
    print(f"[dagger] stage-1 VAL win baseline = {best:.3f} (saved as {CKPT_OUT})", flush=True)

    for rnd in range(1, ROUNDS + 1):
        try:
            secs = [train[(rnd * CHUNK + i) % len(train)] for i in range(CHUNK)]
            new_corr, n_inv, n_turns = corrective_examples(model, secs, rng)
            corrective += new_corr
            loss = train_on(model, opt, base_exs + corrective * OVERSAMPLE, EPOCHS, rng)
            vg = [play(model, s) for s in viz]
            vm = metrics(vg)
            append_epoch(PROG, rnd, vm, vg, sample=12, kind="dagger")
            valw = evaluate(model, val)
            flag = ""
            if valw["win"] > best:
                best = valw["win"]
                save_checkpoint(CKPT_OUT, model, opt, rnd, SFTConfig())
                flag = "  <- best, saved"
            print(
                f"[round {rnd}] invalid {n_inv}/{n_turns} ({n_inv / max(1, n_turns):.2f}) +{len(new_corr)} corrective "
                f"(total {len(corrective)})  loss={loss:.3f}  viz_win={vm['win']:.3f} viz_valid={vm['valid']:.3f}  "
                f"VAL_win={valw['win']:.3f} VAL_valid={valw['valid']:.3f}{flag}",
                flush=True,
            )
        except Exception:  # one bad round must not kill the unattended job — log loudly, continue
            print(f"[round {rnd}] ERROR:\n{traceback.format_exc()}", flush=True)

    print("\n=== DAgger-validity final (best by VAL), eval on disjoint held-out ===", flush=True)
    final = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(CKPT_OUT, final)
    ft, fv, ftr = evaluate(final, test), evaluate(final, val), evaluate(final, tuple(train[:200]))
    print(f"  TEST  held[96:] (HONEST) : win {ft['win']:.3f} valid {ft['valid']:.3f} avg {ft['avg']:.2f}", flush=True)
    print(f"  VAL   held[:96]          : win {fv['win']:.3f} valid {fv['valid']:.3f}", flush=True)
    print(f"  TRAIN[:200]              : win {ftr['win']:.3f} valid {ftr['valid']:.3f}", flush=True)
    print("\n[DAGGER DONE]", flush=True)


if __name__ == "__main__":
    main()
