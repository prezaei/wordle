"""Ephemeral-CoT + aux-validity (driver, not committed): the two honest levers, combined, run long.

= cot_ephemeral.py (honest throwaway-scratchpad CoT: per-turn examples, board-only history, think
regenerated+discarded each turn, NO consistency filter at inference) PLUS the aux-validity loss
(-log P(next letter is dictionary-valid) via a trie) on the CURRENT-turn word-letter positions only.
CoT supplies strategy/search; aux supplies spelling — orthogonal. Trie is TRAINING-only; inference is
free-gen with ZERO rules. Longer schedule (the plain ephemeral run hadn't converged): 50 epochs,
cosine LR, 5 teacher passes. Held-out discipline. -> runs/cot_eph_aux.pt.

The aux mask is GATED by the loss mask: past guesses live in the board-only history (context, no loss),
so aux must apply ONLY where the model is generating (current-turn think candidates + committed guess).
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

from viz_progress import append_epoch  # scripts/ helper: per-epoch boards -> live dashboard
from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_answers, load_valid_guesses, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import consistent_candidates, is_consistent
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft import pretrain_lm, pretrain_words
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
SIZE = os.environ.get("SIZE", "large")  # tiny | base | large | xl — param-scaling sweep
_SIZES = {  # (d_model, n_layers, n_heads, d_ff, dropout, batch, sft_epochs, pretrain_epochs)
    "tiny": (128, 6, 4, 512, 0.10, 256, 50, 30),
    "base": (320, 10, 8, 1280, 0.10, 192, 50, 30),
    "large": (512, 16, 8, 2048, 0.10, 128, 50, 30),
    "xl": (640, 20, 10, 2560, 0.15, 64, 40, 20),  # smaller batch + fewer epochs for MPS time
}
_d, _l, _h, _ff, _dr, _BS, _EP, _PT = _SIZES[SIZE]
CFG = ModelConfig(d_model=_d, n_layers=_l, n_heads=_h, d_ff=_ff, context_len=256, dropout=_dr)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
ANSWERS = load_answers()
VALID = load_valid_guesses()  # the PUBLIC dictionary (14,855 valid guesses) — what a real solver knows
K_CANDS = 3
AUX_LAMBDA = 1.0  # cranked from 0.5: stronger in-the-weights validity pressure (push free-gen spelling)
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


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
    # FAIR fix: candidates are consistent words from the DICTIONARY (valid list), not the answer set.
    # Held-out words appear here only as valid words (legit dictionary knowledge), never as 'answers'.
    # Rejection-sample for speed (the consistent fraction of 14,855 is high enough).
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
    """Per-turn example: board-only history (no loss) + current think+guess (loss)."""
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
    """[B, L, 26] trie-valid next-letter mask at every word-letter predict position (gated to the
    loss region later, so past-board guesses don't contribute)."""
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
        seq = board_only(g.turns)  # NO think from past turns — ephemeral
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


def evaluate(model, secrets):
    model.eval()
    games = [play(model, s) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


train, held = split(seed=0)
SAFE_OPENERS = tuple(o for o in OPENERS if o not in set(held))  # drop held-out openers (e.g. 'trace')
VAL = tuple(held[:96])  # FM-1 fix: select best-checkpoint on VAL ...
TEST = tuple(held[96:])  # ... report the headline ONLY on TEST (never used for selection) — disjoint
curve = VAL
VIZ = tuple(held[:24])  # cheap fixed held-out subset evaluated EVERY epoch for the live dashboard
CKPT = f"runs/cot_eph_aux_{SIZE}.pt"
PROG = f"runs/scale_{SIZE}_progress.jsonl"  # per-epoch boards+metrics -> scripts/live_viz.py
if os.path.exists(PROG):
    os.remove(PROG)  # start each run clean (don't mix epochs across runs)
print(f"[scale {SIZE}] CFG d={_d} L={_l} H={_h} ff={_ff} dropout={_dr} | batch={_BS} sft_ep={_EP} pre_ep={_PT}", flush=True)
nparams = sum(p.numel() for p in WordleGenerator(CFG, VOCAB).parameters())
print(f"[scale {SIZE}] params={nparams / 1e6:.1f}M |VAL|={len(VAL)} |TEST|={len(TEST)} cand-pool=DICTIONARY({len(VALID)}) -> {CKPT}", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=_PT, batch_size=256, device=DEV, seed=0)

print("[data] teacher games -> per-turn ephemeral examples …", flush=True)
games = []
for s in range(5):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.5, openers=SAFE_OPENERS, seed=s, valid_pool=VALID, answer_pool=tuple(train))]  # FAIR: 50% Consistent-over-DICTIONARY (teaches dictionary-wide play incl. held-out valid words) + 50% InfoMax-over-TRAIN (strong strategy, no held-out answer-hood); secrets train-only
rng = Random(0)
exs = [e for g in games for e in game_examples(g, rng)]
print(f"      {len(games)} games -> {len(exs)} per-turn examples; ex len~{len(exs[0][0])}; aux_lambda={AUX_LAMBDA}", flush=True)

EPOCHS, BATCH = _EP, _BS
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=4e-5)
rng2 = Random(0)
best = -1.0
model.train()
for epoch in range(EPOCHS):
    for idx in _batches(len(exs), BATCH, rng2):
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
        # aux: -log(valid letter-mass) at supervised (current-turn) word-letter positions only
        logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)  # [B, L-1, 26]
        vm = vmask[:, :-1]
        aux_pos = (vm.sum(-1) > 0).float() * tmask[:, 1:]  # GATE by the loss mask
        valid_mass = (logp_let.exp() * vm).sum(-1).clamp_min(1e-9)
        aux = (-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
        loss = imit + AUX_LAMBDA * aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()
    # EVERY epoch: greedy eval on a fixed held-out subset -> the live dashboard (every inference shown).
    model.eval()
    vgames = [play(model, s) for s in VIZ]
    vwins = [g for g in vgames if g.won]
    vn = sum(len(g.turns) for g in vgames)
    vm = {
        "win": len(vwins) / len(vgames),
        "valid": sum(is_valid(t.guess) for g in vgames for t in g.turns) / vn if vn else 0.0,
        "avg": statistics.mean(g.guesses_used for g in vwins) if vwins else float("nan"),
    }
    append_epoch(PROG, epoch, vm, vgames, sample=12, kind="sft")
    print(f"  epoch {epoch:>2}  lr={sched.get_last_lr()[0]:.1e}  loss={float(loss.detach()):.3f}  viz_win={vm['win']:.3f} valid={vm['valid']:.3f}", flush=True)
    # EVERY 5 epochs: full VAL eval drives best-checkpoint selection (selection rigor unchanged).
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve)
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint(CKPT, model, opt, epoch, SFTConfig())
            flag = "  <- best, saved"
        print(f"  epoch {epoch:>2}  VAL win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)
    model.train()

print("\n=== FAIR ephemeral-CoT + aux: dictionary-known, deduce; honest TEST (held[96:]) ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint(CKPT, b)
ft = evaluate(b, TEST)
fv = evaluate(b, VAL)
ftr = evaluate(b, tuple(train[:200]))
print(f"  TEST  held[96:] (HONEST headline) : win {ft['win']:.3f}  ({int(round(ft['win'] * len(TEST)))}/{len(TEST)})  valid {ft['valid']:.3f} avg {ft['avg']:.2f}", flush=True)
print(f"  VAL   held[:96] (selected-on)     : win {fv['win']:.3f}", flush=True)
print(f"  TRAIN[:200] (memorization ref)    : win {ftr['win']:.3f}   [gen gap = TRAIN - TEST]", flush=True)
print(f"  [context] over-hobbled(train-pool)=0.17 ; answer-set-leak(contaminated)=0.62 ; this=fair-honest", flush=True)
print("\n=== sample games (board-only prompt, ephemeral think discarded) ===", flush=True)
b.eval()
for s in held[:8]:
    g = play(b, s)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[COTEPHAUX DONE]", flush=True)
