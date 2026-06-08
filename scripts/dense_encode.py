"""Dense constraint-state encoding (honest port of the colleague's V2 idea).

Learning from rynowak/mm V2: REPLACE the raw game history with a clean, pre-digested constraint-state
summary, and encode it richly — greens by position, yellows with their excluded position, grays with
EXACT counts (duplicate-safe). This is the cleaner version of my failed infill (which *appended* a
template to the history; here the constraint state is the ONLY input). The model then generates the
5-letter word char-by-char, FREELY (letter-mask only — NO dictionary, NO answer-trie, NO consistency
filter), so it must learn to deduce + spell from the constraints. Honest: train-only secrets, clean
protocol (non-words counted, not fed back), eval on disjoint TEST. Bar to beat: validity-max ~0.337.

Sequence: [bos] g0..g4 (green row: <L>-green or <blank>) [sep] <facts: L-yellow-pos, L-gray-count> <guess> l0..l4 [eos]
Env: DE_PRETRAIN (30), DE_EPOCHS (40), DE_SECRETS (1852), DE_TEACHER (3), DE_AUX (1.0), DE_LR (3e-4), DE_OUT.
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
from wordle_slm.engine.scoring import score
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft import pretrain_lm, pretrain_words
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
B = tok.vocab_size  # 34 base tokens (pad/bos/eos/sep/guess/green/yellow/gray + 26 letters @ 8..33)
GREEN0 = B            # 26 green-row tokens: GREEN0 + letter_idx
YELLOW0 = B + 26      # 130 yellow tokens: YELLOW0 + letter_idx*5 + pos
GRAY0 = B + 26 + 130  # 78 gray tokens: GRAY0 + letter_idx*3 + count(0..2)
BLANK = B + 26 + 130 + 78  # unknown green slot
VOCAB = BLANK + 1     # 269
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
AUX_LAMBDA = float(os.environ.get("DE_AUX", "1.0"))


def clue_state(turns):
    """Wordle clue logic from valid turns: greens by pos, yellow (letter,pos) pairs, gray exact-counts."""
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
    gray_counts = {ltr: min(gcount[ltr], 2) for ltr in ever_gray}  # exactly this many (0 = absent)
    return greens, yellows, gray_counts


def encode_constraint(greens, yellows, gray_counts):
    """The dense prompt: bos + green-row(5) + sep + facts."""
    ids = [tok.bos_id]
    for pos in range(5):
        ids.append(GREEN0 + (ord(greens[pos]) - 97) if pos in greens else BLANK)
    ids.append(tok.sep_id)
    for ltr, pos in sorted(yellows):
        ids.append(YELLOW0 + (ord(ltr) - 97) * 5 + pos)
    for ltr in sorted(gray_counts):
        ids.append(GRAY0 + (ord(ltr) - 97) * 3 + gray_counts[ltr])
    return ids


def build_example(game):
    """Per-turn: constraint state -> <guess> -> 5 letters. Loss + aux on the 5 letters only."""
    exs = []
    for k, turn in enumerate(game.turns):
        greens, yellows, gray_counts = clue_state(game.turns[:k])
        ids = encode_constraint(greens, yellows, gray_counts)
        mask = [False] * len(ids)
        ids.append(tok.guess_id)
        mask.append(False)
        for ch in turn.guess:  # the 5 committed letters (learned)
            ids.append(tok.token_to_id(ch))
            mask.append(True)
        ids.append(tok.eos_id)
        mask.append(False)
        exs.append((ids, mask))
    return exs


def guess_aux_mask(seqs):
    """[B,L,26] trie-valid next-letter mask at the 5 guess-letter predict positions (the word after <GUESS>)."""
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
def play_dense(model, secret):
    """Clean protocol: encode constraint state from VALID turns, generate 5 letters free (letter-mask),
    non-word counted but not fed back."""
    g = Game(secret)
    visible: list[Turn] = []
    while g.status is Status.ONGOING:
        greens, yellows, gray_counts = clue_state(visible)
        seq = encode_constraint(greens, yellows, gray_counts) + [tok.guess_id]
        letters = []
        for _ in range(5):
            logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
            tid = int(LETTER_IDS[int(torch.argmax(logits[LIDS]))])  # free char-gen, masked to letters only
            letters.append(tok.id_to_token(tid))
            seq.append(tid)
        word = "".join(letters)
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break  # non-word: greedy would repeat from unchanged state -> game lost
    return g


def evaluate(model, secrets):
    model.eval()
    games = [play_dense(model, s) for s in secrets]
    wins = [x for x in games if x.won]
    n = sum(len(x.turns) for x in games)
    return {"win": len(wins) / len(games), "valid": sum(is_valid(t.guess) for x in games for t in x.turns) / n if n else 0.0,
            "avg": statistics.mean(x.guesses_used for x in wins) if wins else float("nan"),
            "nonword": sum(1 for x in games if not x.won and x.turns and not x.turns[-1].valid)}


def dense_warmup(model, epochs, lr=1e-3):
    """Spell warm-up IN THE DENSE FORMAT: empty-constraint -> spell a valid word, over the FULL dictionary.
    Teaches full-vocab spelling at the task's positions/format. Honest: empty constraint carries NO
    answer-hood — it's just 'spell this valid word', which is public for every word (incl. held-out)."""
    base = encode_constraint({}, set(), {}) + [tok.guess_id]  # [bos, _,_,_,_,_, sep, guess]
    exs = []
    for w in load_valid_guesses():
        ids = base + [tok.token_to_id(c) for c in w] + [tok.eos_id]
        mask = [False] * len(base) + [True] * 5 + [False]
        exs.append((ids, mask))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = Random(1)
    model.train()
    for epoch in range(epochs):
        last = 0.0
        for idx in _batches(len(exs), 256, rng):
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
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.detach())
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  [warmup] epoch {epoch:>2} loss {last:.3f}", flush=True)


def main():
    train, held = split(seed=0)
    safe = tuple(o for o in ("salet", "crane", "slate", "trace", "stare", "raise", "crate") if o not in set(held))
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    cap = int(os.environ.get("DE_SECRETS", str(len(train))))
    secrets = tuple(train[:cap])
    OUT = os.environ.get("DE_OUT", "runs/dense_encode.pt")
    EPOCHS = int(os.environ.get("DE_EPOCHS", "40"))
    LR = float(os.environ.get("DE_LR", "3e-4"))
    PRE = int(os.environ.get("DE_PRETRAIN", "30"))
    TEACHER = int(os.environ.get("DE_TEACHER", "3"))
    print(f"[dense] VOCAB={VOCAB} aux={AUX_LAMBDA} epochs={EPOCHS} pretrain={PRE} |secrets|={len(secrets)}", flush=True)

    model = WordleGenerator(CFG := ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1), VOCAB).to(DEV)
    print(f"[dense] DENSE-format spell warm-up ({PRE} ep) — empty-constraint -> word, full dict (right positions)", flush=True)
    dense_warmup(model, PRE)

    games = []
    for s in range(TEACHER):
        games += [tr.game for tr in generate_transcripts(secrets, weak_frac=0.5, openers=safe, seed=300 + s, valid_pool=load_valid_guesses(), answer_pool=secrets)]
    exs = [e for game in games for e in build_example(game)]
    Random(0).shuffle(exs)
    print(f"[dense] teacher games={len(games)} -> examples={len(exs)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    rng2 = Random(0)
    best, saved = -1.0, False
    model.train()
    print("[dense] epoch  clean-VAL win / valid / nonword-losses", flush=True)
    for epoch in range(EPOCHS):
        for idx in _batches(len(exs), 128, rng2):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            vmask = guess_aux_mask([s for s, _ in bs]).to(DEV)
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
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            m = evaluate(model, VAL)
            flag = ""
            if m["win"] > best:
                best = m["win"]
                save_checkpoint(OUT, model, opt, epoch, SFTConfig())
                saved = True
                flag = "  <- best, saved"
            print(f"  epoch {epoch:>2}  loss={float(loss.detach()):.3f}  win {m['win']:.3f} / valid {m['valid']:.3f} / nonword {m['nonword']}{flag}", flush=True)
            model.train()

    print("\n=== DENSE-ENCODE: honest clean-protocol TEST (free char-gen, no dict) ===", flush=True)
    b = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(OUT if saved else OUT, b)
    mt = evaluate(b, TEST)
    print(f"  TEST win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  [bar] validity-max v7 clean 0.338 ; stage-1 0.243", flush=True)
    print("\n[DENSE DONE]", flush=True)


if __name__ == "__main__":
    main()
