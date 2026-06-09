"""Reasoning-CoT: make the model EXECUTE the deduction, not amortize it.

The wall (0.34) is single-pass amortization: we ask one forward pass to compress deduce+search+verify,
learned from ~1852 games. Humans don't memorize clue->answer; they run an algorithm (maintain constraints
-> search vocab -> verify). This teaches the model to do the first, generalizing step EXPLICITLY: condition
on the RAW board history (the grounded input that generalizes — NOT the dense constraint state, which was
handed over for free and got memorized), then DERIVE and state the constraint state token-by-token as its
reasoning (greens-by-pos, yellows-excluded-where, grays-with-counts), THEN guess conditioned on its own
derived constraints. Deriving constraints from history is answer-agnostic (pure clue logic) so it should
GENERALIZE, and the explicit derivation is a reasoning scaffold the guess can respect.

Honest: teacher computes the true constraints at TRAIN time (supervision, allowed); at inference the model
emits its OWN reasoning + guess via free/structured decode (no engine, no dict, no consistency filter); the
reasoning is ephemeral (regenerated each turn, discarded); non-word counted but not fed back; eval = held[96:].

Seq per turn: board_only(history)  [THINK]  <5 green-row slots> [SEP] <yellow facts><gray facts>  [GUESS] l0..l4 [EOS]
Loss on THINK + reasoning + GUESS + 5 letters (+ aux validity on letters). NOT on board history / EOS.
Env: RC_PRETRAIN(30) RC_EPOCHS(40) RC_SECRETS(1852) RC_TEACHER(3) RC_AUX(1.0) RC_LR(3e-4) RC_OUT RC_DIAG(0).
"""

from __future__ import annotations

import os
import statistics
from collections import defaultdict
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
torch.manual_seed(int(os.environ.get("RC_SEED", "0")))
tok = Tokenizer()
B = tok.vocab_size            # 34 base (pad/bos/eos/sep/guess/green/yellow/gray + a-z @ 8..33)
GREEN0 = B                    # 26 green-row tokens: GREEN0 + letter_idx  (a letter known at this slot)
YELLOW0 = B + 26             # 130 yellow facts: YELLOW0 + letter_idx*5 + excluded_pos
GRAY0 = B + 26 + 130        # 78 gray facts: GRAY0 + letter_idx*3 + count(0..2)
BLANK = B + 26 + 130 + 78   # unknown green slot
THINK = BLANK + 1            # marks start of reasoning
VOCAB = THINK + 1            # 270

LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
# decode action spaces (honest: format/grammar only — no answer info, no dict)
GREENROW_ALLOWED = torch.tensor(list(range(GREEN0, GREEN0 + 26)) + [BLANK], device=DEV)
FACT_ALLOWED = torch.tensor(list(range(YELLOW0, YELLOW0 + 130)) + list(range(GRAY0, GRAY0 + 78)) + [tok.guess_id], device=DEV)

AUX_LAMBDA = float(os.environ.get("RC_AUX", "1.0"))


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    """Raw history: bos + (guess g0..g4 fb0..fb4 sep)*  — the grounded input (only VALID turns at play)."""
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def clue_state(turns):
    """Pure clue logic from valid turns: greens by pos, yellow (letter,pos) pairs, gray exact-counts."""
    greens: dict[int, str] = {}
    yellows: set[tuple[str, int]] = set()
    ever_gray: set[str] = set()
    for t in turns:
        if t.feedback is None:
            continue
        for i, c in enumerate(t.feedback):
            ltr = t.guess[i]
            if c is Color.GREEN:
                greens[i] = ltr
            elif c is Color.YELLOW:
                yellows.add((ltr, i))
            else:
                ever_gray.add(ltr)
    gcount: dict[str, int] = defaultdict(int)
    for ltr in greens.values():
        gcount[ltr] += 1
    for ltr, _ in yellows:
        gcount[ltr] += 1
    gray_counts = {ltr: min(gcount[ltr], 2) for ltr in ever_gray}
    return greens, yellows, gray_counts


def reasoning_tokens(greens, yellows, gray_counts):
    """The derived constraint state, as a token sequence (the model's 'show your work'):
    5 green-row slots + SEP + sorted yellow facts + sorted gray facts."""
    ids = []
    for pos in range(5):
        ids.append(GREEN0 + (ord(greens[pos]) - 97) if pos in greens else BLANK)
    ids.append(tok.sep_id)
    for ltr, pos in sorted(yellows):
        ids.append(YELLOW0 + (ord(ltr) - 97) * 5 + pos)
    for ltr in sorted(gray_counts):
        ids.append(GRAY0 + (ord(ltr) - 97) * 3 + gray_counts[ltr])
    return ids


def build_example(game):
    """Per-turn: raw history -> [THINK] derived-constraints [GUESS] 5 letters. Loss on reasoning + guess."""
    exs = []
    for k, turn in enumerate(game.turns):
        ids = board_only(game.turns[:k])
        mask = [False] * len(ids)
        ids.append(THINK)
        mask.append(True)
        rcon = reasoning_tokens(*clue_state(game.turns[:k]))
        ids += rcon
        mask += [True] * len(rcon)
        ids.append(tok.guess_id)
        mask.append(True)
        for ch in turn.guess:
            ids.append(tok.token_to_id(ch))
            mask.append(True)
        ids.append(tok.eos_id)
        mask.append(False)
        exs.append((ids, mask))
    return exs


def guess_aux_mask(seqs):
    """[B,L,26] trie-valid next-letter mask at the 5 letters after each <GUESS>. (Gated by the loss mask
    downstream, so only the FINAL guess letters — mask=True — actually receive aux.)"""
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
def _argmax_over(model, seq, allowed):
    logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
    return int(allowed[int(torch.argmax(logits[allowed]))])


@torch.no_grad()
def play_reason(model, secret):
    """Clean protocol. Structured decode of the model's OWN reasoning (honest format-enforcement only):
    THINK -> 5 green-row slots -> SEP -> facts until GUESS -> 5 free letters. Reasoning ephemeral."""
    g = Game(secret)
    visible: list[Turn] = []
    while g.status is Status.ONGOING:
        seq = board_only(visible)
        seq.append(THINK)
        for _ in range(5):  # the model derives each green slot (letter or BLANK)
            seq.append(_argmax_over(model, seq, GREENROW_ALLOWED))
        seq.append(tok.sep_id)
        for _ in range(26):  # facts until the model commits (cap = backstop)
            nxt = _argmax_over(model, seq, FACT_ALLOWED)
            seq.append(nxt)
            if nxt == tok.guess_id:
                break
        if seq[-1] != tok.guess_id:
            seq.append(tok.guess_id)
        letters = []
        for _ in range(5):  # free char-gen, masked to letters only (no dict)
            tid = int(LIDS[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][LIDS]))])
            letters.append(tok.id_to_token(tid))
            seq.append(tid)
        word = "".join(letters)
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break
    return g


def evaluate(model, secrets):
    model.eval()
    games = [play_reason(model, s) for s in secrets]
    wins = [x for x in games if x.won]
    n = sum(len(x.turns) for x in games)
    return {"win": len(wins) / len(games),
            "valid": sum(is_valid(t.guess) for x in games for t in x.turns) / n if n else 0.0,
            "avg": statistics.mean(x.guesses_used for x in wins) if wins else float("nan")}


def _train_batches(model, exs, opt, epochs, lr, with_aux, label):
    opt = opt or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = Random(1)
    model.train()
    for epoch in range(epochs):
        last = 0.0
        for idx in _batches(len(exs), 256 if not with_aux else 128, rng):
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
            last = float(loss.detach())
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  [{label}] epoch {epoch:>2} loss {last:.3f}", flush=True)
    return opt


def reason_warmup(model, epochs):
    """Spell warm-up at the right position: bos THINK <5 BLANK> SEP GUESS w0..w4 EOS, over the FULL dict.
    Honest: empty reasoning carries no answer-hood; just public spelling after a GUESS."""
    base = [tok.bos_id, THINK] + reasoning_tokens({}, set(), {}) + [tok.guess_id]
    exs = [(base + _letters(w) + [tok.eos_id], [False] * len(base) + [True] * 5 + [False]) for w in load_valid_guesses()]
    _train_batches(model, exs, None, epochs, 1e-3, with_aux=False, label="warmup")


@torch.no_grad()
def diag_derivation(model, secrets):
    """Diagnostic: does the model DERIVE the constraint state correctly? (the generalizing skill)."""
    ok = tot = 0
    for s in secrets:
        g = Game(s)
        # one teacher-ish turn: open with salet, then check derived greens vs true
        g.guess("salet")
        if g.status is not Status.ONGOING:
            continue
        seq = board_only(g.turns)
        seq.append(THINK)
        derived = [_argmax_over(model, seq, GREENROW_ALLOWED) for _ in range(5)]
        true_g, _, _ = clue_state(g.turns)
        for pos in range(5):
            want = (GREEN0 + ord(true_g[pos]) - 97) if pos in true_g else BLANK
            ok += int(derived[pos] == want)
            tot += 1
    return ok / tot if tot else float("nan")


def main():
    train, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    safe = tuple(o for o in ("salet", "crane", "slate", "trace", "stare", "raise", "crate") if o not in set(held))
    cap = int(os.environ.get("RC_SECRETS", str(len(train))))
    secrets = tuple(train[:cap])
    OUT = os.environ.get("RC_OUT", "runs/reason_cot.pt")
    EPOCHS = int(os.environ.get("RC_EPOCHS", "40"))
    LR = float(os.environ.get("RC_LR", "3e-4"))
    PRE = int(os.environ.get("RC_PRETRAIN", "30"))
    TEACHER = int(os.environ.get("RC_TEACHER", "3"))
    print(f"[reason] VOCAB={VOCAB} aux={AUX_LAMBDA} epochs={EPOCHS} pretrain={PRE} |secrets|={len(secrets)}", flush=True)

    CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    print(f"[reason] reasoning-format spell warm-up ({PRE} ep)", flush=True)
    reason_warmup(model, PRE)

    games = []
    for s in range(TEACHER):
        games += [tr.game for tr in generate_transcripts(
            secrets, weak_frac=0.5, openers=safe, seed=300 + s, valid_pool=load_valid_guesses(), answer_pool=secrets)]
    exs = [e for game in games for e in build_example(game)]
    Random(0).shuffle(exs)
    print(f"[reason] teacher games={len(games)} -> examples={len(exs)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    rng2 = Random(0)
    best, saved = -1.0, False
    model.train()
    print("[reason] epoch  clean-VAL win / valid / derive-acc", flush=True)
    for epoch in range(EPOCHS):
        for idx in _batches(len(exs), 128, rng2):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            vmask = guess_aux_mask([s for s, _ in bs]).to(DEV)[:, :-1]
            ids, tmask = ids.to(DEV), tmask.to(DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            imit = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
            aux_pos = (vmask.sum(-1) > 0).float() * tmask[:, 1:]
            valid_mass = (logp_let.exp() * vmask).sum(-1).clamp_min(1e-9)
            aux = (-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
            loss = imit + AUX_LAMBDA * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            m = evaluate(model, VAL)
            dacc = diag_derivation(model, VAL[:48])
            flag = ""
            if m["win"] > best:
                best, saved = m["win"], True
                save_checkpoint(OUT, model, opt, epoch, SFTConfig())
                flag = "  <- best, saved"
            print(f"  epoch {epoch:>2}  win {m['win']:.3f} / valid {m['valid']:.3f} / derive {dacc:.3f}{flag}", flush=True)
            model.train()

    print("\n=== REASON-COT: honest clean-protocol TEST (free reasoning + guess, no dict) ===", flush=True)
    b = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(OUT, b)
    mt = evaluate(b, TEST)
    print(f"  TEST win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  derive-acc(TEST[:48]) {diag_derivation(b, TEST[:48]):.3f}", flush=True)
    print(f"  [bar] validity-max v4 clean 0.338 ; dense-encode 0.065 ; GRPO 0.332", flush=True)
    print("\n[REASON DONE]", flush=True)


if __name__ == "__main__":
    main()
