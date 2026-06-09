"""Iterative Refine: a learned EDIT operator that does cross-pass computation (draft -> fix -> commit).

The lesson from Reasoning-CoT: verbalizing a deterministic function of the input adds no info, so the
model routes around it. Extra compute only helps when it does work that's HARD in one pass. Verification
(is this concrete word consistent?) and repair (fix the violation) are exactly that: checking/fixing a
given candidate is easier than generating a consistent one from scratch. So: condition on raw history +
a DRAFT word, output a better word. At inference, draft = the model's own previous guess, fed back K times
(full-word lookahead each pass), then commit. Honest: model-driven edits only — NO engine, NO dict, NO
consistency filter; the draft is the model's own output; eval = held[96:], non-word counted not fed back.

Lean vocab (36) so spelling stays strong (the big-vocab reason model spelled at only 0.74).
Seq per (turn, draft sample): board_only(history) [DRAFT] d0..d4 [GUESS] g0..g4 [EOS]   (loss on g + aux)
Drafts at train: BLANK*5 (initial-gen) | random valid | near-miss (target w/ letters flipped) | identity.
At play: pass0 draft=BLANK*5 -> g0 ; pass i draft=g(i-1) -> g(i) ; commit g_K.
Env: IR_PRETRAIN(25) IR_EPOCHS(40) IR_SECRETS(1852) IR_TEACHER(3) IR_AUX(1.0) IR_LR(3e-4) IR_PASSES(3)
     IR_DRAFTS(3 samples/turn) IR_OUT.
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_valid_guesses, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.game import Turn
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(int(os.environ.get("IR_SEED", "0")))
tok = Tokenizer()
B = tok.vocab_size       # 34
DRAFT = B                # 34: marks the draft slot
BLANK = B + 1            # 35: an empty draft letter (initial pass)
VOCAB = B + 2            # 36

LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
AUX_LAMBDA = float(os.environ.get("IR_AUX", "1.0"))
PASSES = int(os.environ.get("IR_PASSES", "3"))
DRAFTS = int(os.environ.get("IR_DRAFTS", "3"))
BATCH = int(os.environ.get("IR_BATCH", "128"))
_VALID = load_valid_guesses()


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def _draft_tokens(word_or_none):
    """5 tokens: BLANK*5 for the initial (no-draft) pass, else the draft word's letters."""
    if word_or_none is None:
        return [BLANK] * 5
    return [tok.token_to_id(c) for c in word_or_none]


def sample_drafts(target, rng):
    """Draft distribution the edit operator must handle: initial(BLANK), random valid, near-miss, identity."""
    out = [None]  # always include the initial (BLANK) pass -> teaches first-guess generation
    for _ in range(DRAFTS - 1):
        r = rng.random()
        if r < 0.45:  # near-miss: target with 1-2 positions flipped (the core 'fix me' signal)
            w = list(target)
            for _ in range(rng.choice([1, 2])):
                p = rng.randrange(5)
                w[p] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            out.append("".join(w))
        elif r < 0.85:  # arbitrary valid word
            out.append(rng.choice(_VALID))
        else:  # identity: if the draft is already right, keep it
            out.append(target)
    return out


def build_example(game, rng):
    exs = []
    for k, turn in enumerate(game.turns):
        history = board_only(game.turns[:k])
        for draft in sample_drafts(turn.guess, rng):
            ids = history + [DRAFT] + _draft_tokens(draft) + [tok.guess_id]
            mask = [False] * len(ids)
            for ch in turn.guess:
                ids.append(tok.token_to_id(ch))
                mask.append(True)
            ids.append(tok.eos_id)
            mask.append(False)
            exs.append((ids, mask))
    return exs


def guess_aux_mask(seqs):
    """[B,L,26] trie-valid mask at the 5 letters after each <GUESS> (gated by loss mask downstream)."""
    trie = _valid_trie()
    L = max(len(s) for s in seqs)
    vmask = torch.zeros((len(seqs), L, 26))
    for i, seq in enumerate(seqs):
        for t, tokid in enumerate(seq):
            if tokid != tok.guess_id or t + 5 >= len(seq):
                continue
            node = trie
            for j in range(5):
                for child in node:
                    vmask[i, t + j, child] = 1.0
                nxt = seq[t + 1 + j] - LETTER_LO
                node = node.get(nxt, {}) if 0 <= nxt < 26 else {}
    return vmask


@torch.no_grad()
def _gen5(model, prefix):
    seq = list(prefix)
    letters = []
    for _ in range(5):
        tid = int(LIDS[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][LIDS]))])
        letters.append(tok.id_to_token(tid))
        seq.append(tid)
    return "".join(letters)


@torch.no_grad()
def play_refine(model, secret, passes=PASSES):
    """Clean protocol. pass0 draft=BLANK*5 -> g0 ; then feed each guess back as the next draft; commit g_K."""
    g = Game(secret)
    visible: list[Turn] = []
    while g.status is Status.ONGOING:
        history = board_only(visible)
        draft = None  # initial = BLANK
        word = ""
        for _ in range(passes + 1):
            word = _gen5(model, history + [DRAFT] + _draft_tokens(draft) + [tok.guess_id])
            if word == draft:  # fixed point -> stop early
                break
            draft = word
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break
    return g


def evaluate(model, secrets, passes=PASSES):
    model.eval()
    games = [play_refine(model, s, passes) for s in secrets]
    wins = [x for x in games if x.won]
    n = sum(len(x.turns) for x in games)
    return {"win": len(wins) / len(games),
            "valid": sum(is_valid(t.guess) for x in games for t in x.turns) / n if n else 0.0,
            "avg": statistics.mean(x.guesses_used for x in wins) if wins else float("nan")}


def _run_epochs(model, exs, opt, sched, epochs, with_aux, label, eval_fn=None, out=None, eval_every=5):
    rng2 = Random(0)
    best, saved = -1.0, False
    model.train()
    for epoch in range(epochs):
        for idx in _batches(len(exs), 256 if not with_aux else BATCH, rng2):
            bs = [exs[i] for i in idx]
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
            if with_aux:
                vmask = guess_aux_mask([s for s, _ in bs]).to(DEV)[:, :-1]
                logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
                aux_pos = (vmask.sum(-1) > 0).float() * tmask[:, 1:]
                valid_mass = (logp_let.exp() * vmask).sum(-1).clamp_min(1e-9)
                loss = loss + AUX_LAMBDA * ((-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
        if eval_fn is None:
            if epoch % eval_every == 0 or epoch == epochs - 1:
                print(f"  [{label}] epoch {epoch:>2} loss {float(loss.detach()):.3f}", flush=True)
        elif epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            m = eval_fn(model)
            flag = ""
            if m["win"] > best:
                best, saved = m["win"], True
                save_checkpoint(out, model, opt, epoch, SFTConfig())
                flag = "  <- best, saved"
            print(f"  epoch {epoch:>2}  win {m['win']:.3f} / valid {m['valid']:.3f} / avg {m['avg']:.2f}{flag}", flush=True)
            model.train()
    return saved


def warmup(model, epochs):
    """Spell warm-up: bos [DRAFT] BLANK*5 [GUESS] w0..w4 EOS over the full dict (initial-pass spelling)."""
    base = [tok.bos_id, DRAFT] + [BLANK] * 5 + [tok.guess_id]
    exs = [(base + _letters(w) + [tok.eos_id], [False] * len(base) + [True] * 5 + [False]) for w in _VALID]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    _run_epochs(model, exs, opt, None, epochs, with_aux=False, label="warmup")


def main():
    train, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    safe = tuple(o for o in ("salet", "crane", "slate", "trace", "stare", "raise", "crate") if o not in set(held))
    cap = int(os.environ.get("IR_SECRETS", str(len(train))))
    secrets = tuple(train[:cap])
    OUT = os.environ.get("IR_OUT", "runs/iter_refine.pt")
    EPOCHS = int(os.environ.get("IR_EPOCHS", "40"))
    LR = float(os.environ.get("IR_LR", "3e-4"))
    PRE = int(os.environ.get("IR_PRETRAIN", "25"))
    TEACHER = int(os.environ.get("IR_TEACHER", "3"))
    print(f"[refine] VOCAB={VOCAB} aux={AUX_LAMBDA} epochs={EPOCHS} pretrain={PRE} passes={PASSES} "
          f"drafts/turn={DRAFTS} |secrets|={len(secrets)}", flush=True)

    CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    print(f"[refine] spell warm-up ({PRE} ep)", flush=True)
    warmup(model, PRE)

    rng = Random(0)
    games = []
    for s in range(TEACHER):
        games += [tr.game for tr in generate_transcripts(
            secrets, weak_frac=0.5, openers=safe, seed=300 + s, valid_pool=_VALID, answer_pool=secrets)]
    exs = [e for game in games for e in build_example(game, rng)]
    rng.shuffle(exs)
    print(f"[refine] teacher games={len(games)} -> examples={len(exs)} (~{DRAFTS}x drafts)", flush=True)

    # Lean during-training eval (batch-1 autoregressive eval is latency-bound on MPS): few games, 1 pass,
    # sparse cadence — just to track learning + pick best. Full pass-ablation + TEST happen at the end.
    EVAL_N = int(os.environ.get("IR_EVAL_N", "48"))
    EVAL_PASSES = int(os.environ.get("IR_EVAL_PASSES", "1"))
    EVAL_EVERY = int(os.environ.get("IR_EVAL_EVERY", "8"))
    VAL_EVAL = VAL[:EVAL_N]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    print(f"[refine] epoch  VAL[:{EVAL_N}] win / valid / avg  (eval passes={EVAL_PASSES}, every {EVAL_EVERY})", flush=True)
    saved = _run_epochs(model, exs, opt, sched, EPOCHS, with_aux=True, label="sft",
                        eval_fn=lambda m: evaluate(m, VAL_EVAL, passes=EVAL_PASSES), out=OUT, eval_every=EVAL_EVERY)

    print("\n=== ITER-REFINE: honest clean-protocol TEST (draft->refine, no dict) ===", flush=True)
    b = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(OUT if saved else OUT, b)
    # pass-count ablation on VAL: does refinement actually help vs single-pass?
    for p in (0, 1, 3):
        mv = evaluate(b, VAL, passes=p)
        print(f"  VAL passes={p}: win {mv['win']:.3f} valid {mv['valid']:.3f}", flush=True)
    mt = evaluate(b, TEST)
    print(f"  TEST win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  [bar] validity-max v4 clean 0.338 ; dense 0.065 ; GRPO 0.332 ; reason-CoT (collapsed)", flush=True)
    print("\n[REFINE DONE]", flush=True)


if __name__ == "__main__":
    main()
