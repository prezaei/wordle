"""Validity-max (composition lever, not committed): push in-weights free-gen validity past 0.80.

Diagnosis from the lost games (mango/dried/hocky): the model DEDUCES the answer's letters/positions but
fails to COMPOSE them into a valid 5-letter word — it emits near-words (manim, hocky, bried). Perfect
spelling is worth +15pts: constrained-decode (mask to real words at inference, validity 1.0) = 0.436 vs
greedy free-gen 0.281. The closest prior, distill_constrained, raised in-weights free-gen validity
0.62→0.80 but win stayed flat. The UNTRIED question: if in-weights validity reaches ~0.95, does free-gen
WIN climb toward 0.436, or stay flat (proving deduction-generalization, not spelling, is the real wall)?

This run answers it: warm-start stage-1, fine-tune on teacher games (preserve deduction) + the model's own
CONSTRAINED-DECODE games (always-valid composition targets) + a CRANKED aux-validity loss (λ default 3).
Track free-gen win AND validity EVERY epoch so we see whether win follows validity. Honest: teacher/secret
pools are train-only; eval = free-gen greedy (ZERO rules) on disjoint TEST. -> runs/validity_max.pt

Env: VM_AUX (aux lambda, default 3.0), VM_EPOCHS (default 16), VM_LR (default 1.5e-4), VM_SECRETS (cap,
default 800), VM_CONSTRAINED (include constrained self-games, default 1), VM_BASE/VM_OUT (ckpts).
"""

from __future__ import annotations

import os
import statistics
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
import os as _os
torch.manual_seed(int(_os.environ.get("VM_SEED","0")))  # train seed (split stays seed=0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=float(os.environ.get("VM_DROPOUT","0.1")))  # ~50M
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
VALID = load_valid_guesses()
TRIE = _valid_trie()
K_CANDS = 3
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}

AUX_LAMBDA = float(os.environ.get("VM_AUX", "3.0"))  # cranked from stage-1's 1.0
EPOCHS = int(os.environ.get("VM_EPOCHS", "16"))
LR = float(os.environ.get("VM_LR", "1.5e-4"))
CAP = int(os.environ.get("VM_SECRETS", "800"))
USE_CONSTRAINED = os.environ.get("VM_CONSTRAINED", "1") == "1"
BASE = os.environ.get("VM_BASE", "runs/cot_eph_aux_fair.pt")
OUT = os.environ.get("VM_OUT", "runs/validity_max.pt")


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


def game_examples(game, rng):
    exs = []
    for k, turn in enumerate(game.turns):
        ids = board_only(game.turns[:k])
        mask = [False] * len(ids)
        for c in pick_cands(game.turns[:k], turn.guess, rng):
            ids.append(THINK)
            mask.append(True)
            ids += _letters(c)
            mask += [True] * 5
        ids.append(tok.guess_id)
        mask.append(True)
        ids += _letters(turn.guess)
        mask += [True] * 5
        ids.append(tok.eos_id)
        mask.append(False)
        exs.append((ids, mask))
    return exs


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
    """Greedy ephemeral-CoT FREE-gen play — the honest eval path (no rules, no mask)."""
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
    """Greedy think, then commit 5 letters MASKED to the dictionary trie — always emits a real word.
    Used only to MINE always-valid composition targets on TRAIN secrets (a training wheel, never eval)."""
    g = Game(secret)
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)
        for _ in range(60):  # free-gen the think until it commits
            nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
            seq.append(nxt)
            if nxt == tok.guess_id:
                break
        node, letters = TRIE, []
        for _ in range(5):  # mask each letter to trie-valid continuations
            logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
            opts = list(node.keys())
            if not opts:
                break
            best = max(opts, key=lambda idx: float(logits[LETTER_LO + idx]))
            letters.append(LETTER_LO + best)
            seq.append(LETTER_LO + best)
            node = node[best]
        word = "".join(tok.id_to_token(t) for t in letters[:5])
        g.guess(word if len(word) == 5 else "zzzzz")
    return g


def evaluate(model, secrets):
    model.eval()
    games = [play(model, s) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def main():
    train, held = split(seed=0)
    safe_openers = tuple(o for o in OPENERS if o not in set(held))
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    secrets = tuple(train[:CAP])
    PROG = "runs/validity_max_progress.jsonl"
    if os.path.exists(PROG):
        os.remove(PROG)
    print(f"[vm] aux={AUX_LAMBDA} epochs={EPOCHS} lr={LR} |secrets|={len(secrets)} constrained={USE_CONSTRAINED} "
          f"base={BASE}", flush=True)

    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(BASE, model)
    base = evaluate(model, VAL)
    print(f"[vm] warm-start VAL win={base['win']:.3f} valid={base['valid']:.3f}  "
          f"(stage-1 ref: free-gen TEST 0.281/0.662 ; constrained-mask 0.436/1.0)", flush=True)

    _SEED = int(os.environ.get("VM_SEED", "0"))
    rng = Random(_SEED)
    games = []
    for s in range(3):  # teacher games preserve deduction
        games += [tr.game for tr in generate_transcripts(
            secrets, weak_frac=0.5, openers=safe_openers, seed=100 + s + 1000 * _SEED, valid_pool=VALID, answer_pool=secrets)]
    n_teacher = len(games)
    if USE_CONSTRAINED:  # the model's OWN always-valid composition targets (winning constrained games)
        print(f"[vm] mining constrained-decode games on {len(secrets)} train secrets …", flush=True)
        model.eval()
        n_cwin = 0
        for i, s in enumerate(secrets):
            gc = play_constrained(model, s)
            if gc.won:
                games.append(gc)
                n_cwin += 1
            if (i + 1) % 200 == 0:
                print(f"      {i + 1}/{len(secrets)}  constrained_wins={n_cwin}", flush=True)
        print(f"[vm] constrained win_rate={n_cwin / len(secrets):.3f}  added {n_cwin} valid games", flush=True)
    exs = [e for g in games for e in game_examples(g, rng)]
    rng.shuffle(exs)
    print(f"[vm] examples={len(exs)} (teacher_games={n_teacher} + constrained_games={len(games) - n_teacher})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    rng2 = Random(_SEED)
    best = base["win"]
    saved = False
    VIZ = tuple(held[:24])
    model.train()
    print("[vm] epoch  free-gen VAL: win / valid  (does win follow validity past 0.80?)", flush=True)
    for epoch in range(EPOCHS):
        for idx in _batches(len(exs), 128, rng2):
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
        sched.step()
        model.eval()
        vgames = [play(model, s) for s in VIZ]
        vwins = [x for x in vgames if x.won]
        vn = sum(len(x.turns) for x in vgames)
        vmet = {"win": len(vwins) / len(vgames),
                "valid": sum(is_valid(t.guess) for x in vgames for t in x.turns) / vn if vn else 0.0,
                "avg": statistics.mean(x.guesses_used for x in vwins) if vwins else float("nan")}
        append_epoch(PROG, epoch, vmet, vgames, sample=12, kind="sft")
        m = evaluate(model, VAL)
        sel = m["valid"] if os.environ.get("VM_SELECT", "win") == "valid" else m["win"]  # validity is robust
        flag = ""
        if sel > best:
            best = sel
            save_checkpoint(OUT, model, opt, epoch, SFTConfig())
            saved = True
            flag = "  <- best, saved"
        print(f"  epoch {epoch:>2}  win {m['win']:.3f} / valid {m['valid']:.3f}{flag}", flush=True)
        model.train()

    print("\n=== validity-max: honest TEST (held[96:], free-gen greedy, zero rules) ===", flush=True)
    if not saved:
        print("  [null-on-win] never beat stage-1 VAL — re-eval validity trajectory above for the relationship.", flush=True)
        b = model
    else:
        b = WordleGenerator(CFG, VOCAB).to(DEV)
        load_checkpoint(OUT, b)
    ft, fv = evaluate(b, TEST), evaluate(b, VAL)
    print(f"  TEST  win {ft['win']:.3f} ({int(round(ft['win'] * len(TEST)))}/{len(TEST)}) valid {ft['valid']:.3f} avg {ft['avg']:.2f}", flush=True)
    print(f"  VAL   win {fv['win']:.3f} valid {fv['valid']:.3f}", flush=True)
    print(f"  [verdict] stage-1 free-gen 0.281/0.662 ; constrained-mask 0.436/1.0 — did validity rise? did win follow?", flush=True)
    print("\n[VM DONE]", flush=True)


if __name__ == "__main__":
    main()
