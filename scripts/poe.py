"""Product-of-Experts: fuse a common-word GENERATOR with a clue-consistency EXPERT at the decode logits.

The route-around finding (reason-CoT, iter-refine both collapsed): any auxiliary channel the model can
ignore becomes a no-op; bottlenecking instead memorizes (dense). PoE escapes BOTH: two SEPARATE networks
whose letter-logits MULTIPLY — neither can be ignored (both must assign mass to a letter for it to win),
and the consistency net learns the RULE (answer-agnostic, full-dict) rather than an answer lookup.

  G (generator): P_G(letter | board history)  — the existing best model (validity_max_v4, FROZEN), with its
                 load-bearing ephemeral CoT. Provides COMMON-WORD generation (answers are common words).
  C (consistency expert): P_C(letter | constraint-state, partial word) — a NEW net trained answer-AGNOSTICALLY
                 on (clue-state -> consistent valid word) pairs over the FULL 14,855-word dict (NOT the answer
                 set). Provides CLUE-RESPECT + validity. Learns the rule; can't memorize a 2,315-answer lookup.
  Decode:  letter = argmax_over_26[ logP_G + beta * logP_C ]   (mask to the 26 letters only — the usual action
                 space; NO dictionary, NO trie, NO consistency-engine, NO answer set at inference).

Honesty: both experts are weights evaluated by forward pass (same category as the accepted aux-validity loss,
just stronger). C trained ONLY on the public valid-guess dict, answer-agnostic, never on held-out answers.
beta=0 in the sweep == G-alone baseline (built-in ablation). TRAIN-vs-HELD audit: if PoE only helps on TRAIN
(memorization), it's rejected — same audit that caught dense-encode and the colleague's contamination.

Env: POE_CSTATES(20000) POE_CEPOCHS(30) POE_CPRE(20) POE_CAUX(1.0) POE_LR(3e-4) POE_BETAS("0,0.5,1,2")
     POE_G(runs/validity_max_v4.pt) POE_OUT(runs/poe_c.pt).
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
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.engine.game import Turn
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint

DEV = "mps"
torch.manual_seed(int(os.environ.get("POE_SEED", "0")))
tok = Tokenizer()
B = tok.vocab_size  # 34 base
# ---- G (generator) vocab: base + THINK (matches validity_max) ----
THINK = B            # 34
VOCAB_G = B + 1      # 35
# ---- C (consistency expert) vocab: base + constraint tokens (matches dense_encode) ----
GREEN0 = B           # 26
YELLOW0 = B + 26     # 130
GRAY0 = B + 26 + 130  # 78
BLANK = B + 26 + 130 + 78  # 268
VOCAB_C = BLANK + 1  # 269

LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)  # G's CoT action space
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
CAUX = float(os.environ.get("POE_CAUX", "1.0"))
VALID = load_valid_guesses()


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):  # G's input format
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def clue_state(turns):  # C's deduction logic
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


def encode_constraint(greens, yellows, gray_counts):  # C's input
    ids = [tok.bos_id]
    for pos in range(5):
        ids.append(GREEN0 + (ord(greens[pos]) - 97) if pos in greens else BLANK)
    ids.append(tok.sep_id)
    for ltr, pos in sorted(yellows):
        ids.append(YELLOW0 + (ord(ltr) - 97) * 5 + pos)
    for ltr in sorted(gray_counts):
        ids.append(GRAY0 + (ord(ltr) - 97) * 3 + gray_counts[ltr])
    return ids


# ----------------------------- C: answer-agnostic consistency data -----------------------------
def consistency_data(n_states, rng):
    """(clue-state -> consistent valid word) pairs over the FULL dict. Answer-AGNOSTIC: the hypothetical
    'secret' is drawn from the full 14,855-word valid list (NOT the answer set), and targets are ANY valid
    words consistent with the clue-state — so C learns the consistency RULE, not an answer lookup."""
    exs = []
    for _ in range(n_states):
        secret = rng.choice(VALID)
        g = Game(secret)
        turns: list[Turn] = []
        for _ in range(rng.randint(0, 3)):
            if g.status is not Status.ONGOING:
                break
            t = g.guess(rng.choice(VALID))
            if t.valid:
                turns.append(t)
        cons = [secret] if is_consistent(secret, turns) else []
        for _ in range(60):
            w = rng.choice(VALID)
            if w not in cons and is_consistent(w, turns):
                cons.append(w)
            if len(cons) >= 4:
                break
        pre = encode_constraint(*clue_state(turns)) + [tok.guess_id]
        for w in cons:
            exs.append((pre + _letters(w) + [tok.eos_id], [False] * len(pre) + [True] * 5 + [False]))
    return exs


def guess_aux_mask(seqs):
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


def _train(model, exs, epochs, lr, with_aux, label, batch=128):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
    rng = Random(0)
    model.train()
    for epoch in range(epochs):
        last = 0.0
        for idx in _batches(len(exs), batch, rng):
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
                loss = loss + CAUX * ((-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.detach())
        if sched is not None:
            sched.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  [{label}] epoch {epoch:>2} loss {last:.3f}", flush=True)


def warmup_C(model, epochs):
    """Spell warm-up for C: empty-constraint -> valid word (full dict). Honest (public spelling)."""
    pre = encode_constraint({}, set(), {}) + [tok.guess_id]
    exs = [(pre + _letters(w) + [tok.eos_id], [False] * len(pre) + [True] * 5 + [False]) for w in VALID]
    _train(model, exs, epochs, 1e-3, with_aux=False, label="C-warmup", batch=256)


# ----------------------------- PoE decode -----------------------------
@torch.no_grad()
def _g_commit_prefix(G, visible):
    """Run G's ephemeral CoT to the commit point; return seq ending at <GUESS>, ready for 5 letters."""
    seq = board_only(visible)
    for _ in range(60):
        nxt = int(ALLOWED_GEN[int(torch.argmax(G.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
        seq.append(nxt)
        if nxt == tok.guess_id:
            return seq
    seq.append(tok.guess_id)
    return seq


@torch.no_grad()
def play_poe(G, C, secret, beta):
    """Clean protocol. At each of the 5 letter positions: argmax over 26 letters of logP_G + beta*logP_C.
    beta=0 -> pure G (baseline). NO dict/trie/engine/answer-set — only the 26-letter action mask."""
    g = Game(secret)
    visible: list[Turn] = []
    while g.status is Status.ONGOING:
        seqG = _g_commit_prefix(G, visible)
        seqC = encode_constraint(*clue_state(visible)) + [tok.guess_id]
        letters = []
        for _ in range(5):
            lpG = torch.log_softmax(G.forward(torch.tensor([seqG], device=DEV))[0, -1][LIDS], dim=-1)
            if beta != 0.0:
                lpC = torch.log_softmax(C.forward(torch.tensor([seqC], device=DEV))[0, -1][LIDS], dim=-1)
                combined = lpG + beta * lpC
            else:
                combined = lpG
            tid = int(LIDS[int(torch.argmax(combined))])
            letters.append(tok.id_to_token(tid))
            seqG.append(tid)
            seqC.append(tid)
        word = "".join(letters)
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break
    return g


def evaluate(G, C, secrets, beta):
    games = [play_poe(G, C, s, beta) for s in secrets]
    wins = [x for x in games if x.won]
    n = sum(len(x.turns) for x in games)
    return {"win": len(wins) / len(games),
            "valid": sum(is_valid(t.guess) for x in games for t in x.turns) / n if n else 0.0,
            "avg": statistics.mean(x.guesses_used for x in wins) if wins else float("nan")}


def main():
    train, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    EVAL_N = int(os.environ.get("POE_EVAL_N", "48"))  # lean sweep/audit (PoE decode is dual-model, heavy)
    VAL_SWEEP = VAL[:EVAL_N]
    TRAIN_PROBE = tuple(train[:EVAL_N])  # for the memorization audit
    G_PATH = os.environ.get("POE_G", "runs/validity_max_v4.pt")
    OUT = os.environ.get("POE_OUT", "runs/poe_c.pt")
    N_STATES = int(os.environ.get("POE_CSTATES", "20000"))
    CEPOCHS = int(os.environ.get("POE_CEPOCHS", "30"))
    CPRE = int(os.environ.get("POE_CPRE", "20"))
    LR = float(os.environ.get("POE_LR", "3e-4"))
    BETAS = [float(b) for b in os.environ.get("POE_BETAS", "0,0.5,1,2").split(",")]
    print(f"[poe] G={G_PATH} C-states={N_STATES} C-epochs={CEPOCHS} pre={CPRE} betas={BETAS}", flush=True)

    G = WordleGenerator(CFG, VOCAB_G).to(DEV)
    load_checkpoint(G_PATH, G)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    C = WordleGenerator(CFG, VOCAB_C).to(DEV)
    print(f"[poe] C spell warm-up ({CPRE} ep)", flush=True)
    warmup_C(C, CPRE)
    print(f"[poe] building answer-agnostic consistency data ({N_STATES} states) …", flush=True)
    exs = consistency_data(N_STATES, Random(1))
    Random(0).shuffle(exs)
    print(f"[poe] C training examples={len(exs)}", flush=True)
    _train(C, exs, CEPOCHS, LR, with_aux=True, label="C-sft")
    save_checkpoint(OUT, C, torch.optim.AdamW(C.parameters(), lr=LR), CEPOCHS, SFTConfig())
    C.eval()

    print("\n[poe] beta sweep on VAL (beta=0 == G-alone baseline):", flush=True)
    best_beta, best_win = 0.0, -1.0
    for beta in BETAS:
        m = evaluate(G, C, VAL_SWEEP, beta)
        flag = ""
        if m["win"] > best_win:
            best_win, best_beta = m["win"], beta
            flag = "  <- best"
        print(f"  beta={beta:>4}: VAL win {m['win']:.3f} valid {m['valid']:.3f} avg {m['avg']:.2f}{flag}", flush=True)

    print(f"\n[poe] === memorization audit @ best beta={best_beta} (TRAIN vs HELD) ===", flush=True)
    tr = evaluate(G, C, TRAIN_PROBE, best_beta)
    print(f"  TRAIN-probe win {tr['win']:.3f}  |  (if TRAIN >> HELD -> C memorized, REJECT)", flush=True)
    mt = evaluate(G, C, TEST, best_beta)
    g_only = evaluate(G, C, TEST, 0.0)
    print(f"\n=== PoE: honest clean-protocol TEST (G x C^{best_beta}, no dict) ===", flush=True)
    print(f"  TEST  win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  G-only TEST (beta=0) win {g_only['win']:.3f} valid {g_only['valid']:.3f}", flush=True)
    print(f"  delta vs G-only: {mt['win'] - g_only['win']:+.3f}   |   train-probe {tr['win']:.3f}", flush=True)
    print(f"  [bar] validity-max v4 clean 0.338", flush=True)
    print("\n[POE DONE]", flush=True)


if __name__ == "__main__":
    main()
