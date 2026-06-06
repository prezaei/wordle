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

import statistics
from random import Random

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_answers, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import consistent_candidates
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft import pretrain_lm, pretrain_words
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)  # ~50M
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
ANSWERS = load_answers()
K_CANDS = 3
AUX_LAMBDA = 0.5
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
    cons = [w for w in consistent_candidates(history, ANSWERS) if w != guess]
    rng.shuffle(cons)
    cands = [guess] + cons[: K_CANDS - 1]
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
curve = tuple(held[:96])
print(f"[pretrain] 50M (vocab {VOCAB}) spell warm-up …", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=10, batch_size=256, device=DEV, seed=0)

print("[data] teacher games -> per-turn ephemeral examples …", flush=True)
games = []
for s in range(5):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
rng = Random(0)
exs = [e for g in games for e in game_examples(g, rng)]
print(f"      {len(games)} games -> {len(exs)} per-turn examples; ex len~{len(exs[0][0])}; aux_lambda={AUX_LAMBDA}", flush=True)

EPOCHS, BATCH = 50, 128
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
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve)
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/cot_eph_aux.pt", model, opt, epoch, SFTConfig())
            flag = "  <- best, saved"
        print(f"  epoch {epoch:>2}  lr={sched.get_last_lr()[0]:.1e}  loss={float(loss.detach()):.3f} (imit={float(imit.detach()):.3f} aux={float(aux.detach()):.3f})  win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)
    model.train()

print(f"\n=== HONEST ephemeral-CoT + aux: full held-out ({len(held)}), best checkpoint ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_aux.pt", b)
f = evaluate(b, tuple(held))
print(f"  held-out win : {f['win']:.3f}  ({int(round(f['win'] * len(held)))}/{len(held)})   [ephemeral-CoT plain = 0.430; char-50M+aux = 0.436]", flush=True)
print(f"  valid-rate   : {f['valid']:.3f}   avg : {f['avg']:.2f}", flush=True)
print("\n=== sample games (board-only prompt, ephemeral think discarded) ===", flush=True)
b.eval()
for s in held[:8]:
    g = play(b, s)
    out = [f"  secret={g.secret} [{g.status.value} in {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[COTEPHAUX DONE]", flush=True)
