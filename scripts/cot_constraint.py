"""Constraint-aware aux (driver, not committed): train the model to KEEP GREENS + REUSE YELLOWS.

The hard-tail failure (joist: HOIST locks _oist, then toist/loist/moint/foigt — invalid, and moint/foigt
even abandon the locked greens). Fix = bake hard-mode discipline into the WEIGHTS (training signal only;
inference stays PURE greedy, no rules — same legitimacy as aux-validity):

  - GREEN term:  at each generated letter position that is a known-green (from the board's own feedback),
                 push prob onto the green letter  -> -log P(green letter).
  - YELLOW term: on the committed guess, push up P(each known-yellow letter APPEARS across the 5 positions)
                 = 1 - Π(1 - P(L at pos))  -> -log P(appears).  Differentiable "reuse the yellows".
  - REPEAT term: push DOWN P(committed guess == any past guess) -> -log(1 - Σ_w Π_j P(w[j]@j)). The
                 train-it-in analog of a negative reward for spelling the same word twice.

All derived only from the model's own clues/history (no dictionary, no answer, no candidate list).
Fine-tunes runs/cot_eph_aux.pt (0.616) -> runs/cot_eph_constraint.pt.
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
ANSWERS = load_answers()
K_CANDS = 3
LAM_V, LAM_G, LAM_Y, LAM_R = 0.5, 0.5, 0.3, 0.3
MAX_PAST = 6
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def _lidx(ch):
    return tok.token_to_id(ch) - LETTER_LO


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def derive(turns):
    """greens {pos:char}, needed-yellow letter indices (present letters not green-placed)."""
    greens, present = {}, set()
    for t in turns:
        if not t.valid or t.feedback is None:
            continue
        for i, c in enumerate(t.feedback):
            if c is Color.GREEN:
                greens[i] = t.guess[i]
                present.add(t.guess[i])
            elif c is Color.YELLOW:
                present.add(t.guess[i])
    needed = [_lidx(ch) for ch in present if ch not in set(greens.values())]
    return greens, needed


def pick_cands(history, guess, rng):
    cons = [w for w in consistent_candidates(history, ANSWERS) if w != guess]
    rng.shuffle(cons)
    cands = [guess] + cons[: K_CANDS - 1]
    rng.shuffle(cands)
    return cands


def game_examples(game, rng):
    """Per-turn example + green_target (per token) + guess predict-positions + needed-yellow idxs."""
    exs = []
    for k, turn in enumerate(game.turns):
        greens, needed = derive(game.turns[:k])
        ids = board_only(game.turns[:k])
        mask = [False] * len(ids)
        gtar = [-1] * len(ids)
        for c in pick_cands(game.turns[:k], turn.guess, rng):
            ids.append(THINK)
            mask.append(True)
            gtar.append(-1)
            for j, ch in enumerate(c):
                ids.append(tok.token_to_id(ch))
                mask.append(True)
                gtar.append(_lidx(greens[j]) if j in greens else -1)
        ids.append(tok.guess_id)
        mask.append(True)
        gtar.append(-1)
        guess_pred = []
        for j, ch in enumerate(turn.guess):
            guess_pred.append(len(ids) - 1)  # predict-position for this guess letter (logp_let index)
            ids.append(tok.token_to_id(ch))
            mask.append(True)
            gtar.append(_lidx(greens[j]) if j in greens else -1)
        ids.append(tok.eos_id)
        mask.append(False)
        gtar.append(-1)
        past = [[_lidx(ch) for ch in pt.guess] for pt in game.turns[:k]][-MAX_PAST:]
        exs.append((ids, mask, gtar, guess_pred, needed, past))
    return exs


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
def play(model, secret, rows=6):
    g = Game(secret, max_guesses=rows)
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)
        guess, committed = [], False
        for _ in range(48):
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


def green_break_rate(games):
    """Fraction of guesses that violate a known green (the bug we're fixing)."""
    bad = tot = 0
    for g in games:
        for k, t in enumerate(g.turns):
            greens, _ = derive(g.turns[:k])
            if not greens:
                continue
            tot += 1
            if any(t.guess[i] != ch for i, ch in greens.items()):
                bad += 1
    return bad / tot if tot else 0.0


def repeat_rate(games):
    """Fraction of games that re-emit a previous guess (the bug we're fixing)."""
    rep = sum(len({t.guess for t in g.turns}) != len(g.turns) for g in games)
    return rep / len(games) if games else 0.0


def evaluate(model, secrets):
    model.eval()
    games = [play(model, s) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n, "gbreak": green_break_rate(games),
            "repeat": repeat_rate(games),
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


train, held = split(seed=0)
curve = tuple(held[:96])
print(f"[load] fine-tune runs/cot_eph_aux.pt (0.616)  λ_v={LAM_V} λ_g={LAM_G} λ_y={LAM_Y}", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_aux.pt", model)

print("[data] teacher games -> per-turn examples + constraints …", flush=True)
games = []
for s in range(3):
    games += [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=s)]
rng = Random(0)
exs = [e for g in games for e in game_examples(g, rng)]
print(f"      {len(games)} games -> {len(exs)} examples", flush=True)

b = evaluate(model, curve)
print(f"[base] win={b['win']:.3f} valid={b['valid']:.3f} green-break={b['gbreak']:.3f} repeat={b['repeat']:.3f}", flush=True)

EPOCHS, BATCH = 20, 128
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
rng2 = Random(0)
best = b["win"]
save_checkpoint("runs/cot_eph_constraint.pt", model, opt, 0, SFTConfig())
for epoch in range(EPOCHS):
    model.train()
    for idx in _batches(len(exs), BATCH, rng2):
        bs = [exs[i] for i in idx]
        L = max(len(s) for s, *_ in bs)
        ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
        tmask = torch.zeros((len(bs), L))
        gtar = torch.full((len(bs), L), -1, dtype=torch.long)
        gpred = torch.zeros((len(bs), 5), dtype=torch.long)
        ymask = torch.zeros((len(bs), 26))
        past_idx = torch.zeros((len(bs), MAX_PAST, 5), dtype=torch.long)
        past_valid = torch.zeros((len(bs), MAX_PAST))
        for i, (s, m, gt, gp, ned, past) in enumerate(bs):
            ids[i, : len(s)] = torch.tensor(s)
            tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            gtar[i, : len(gt)] = torch.tensor(gt)
            gpred[i] = torch.tensor(gp)
            for li in ned:
                ymask[i, li] = 1.0
            for p, w in enumerate(past):
                past_idx[i, p] = torch.tensor(w)
                past_valid[i, p] = 1.0
        vmask = cot_valid_mask([s for s, *_ in bs]).to(DEV)
        ids, tmask, gtar, gpred, ymask = ids.to(DEV), tmask.to(DEV), gtar.to(DEV), gpred.to(DEV), ymask.to(DEV)
        past_idx, past_valid = past_idx.to(DEV), past_valid.to(DEV)
        logits = model.forward(ids)
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        imit = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
        logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)  # [B, L-1, 26]
        m1 = tmask[:, 1:]
        # aux-validity (dictionary)
        vm = vmask[:, :-1]
        vpos = (vm.sum(-1) > 0).float() * m1
        aux_v = (-(logp_let.exp() * vm).sum(-1).clamp_min(1e-9).log() * vpos).sum() / vpos.sum().clamp_min(1.0)
        # GREEN: push P(green letter) at known-green generated positions
        gt = gtar[:, 1:]
        gpos = (gt >= 0).float() * m1
        glogp = logp_let.gather(-1, gt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        aux_g = (-glogp * gpos).sum() / gpos.sum().clamp_min(1.0)
        # YELLOW: push P(needed-yellow letter appears across the 5 committed-guess positions)
        probs = logp_let.exp()  # [B, L-1, 26]
        gprobs = probs.gather(1, gpred.clamp(min=0).unsqueeze(-1).expand(-1, -1, 26))  # [B,5,26]
        p_app = (1.0 - (1.0 - gprobs).clamp(1e-6, 1.0).prod(dim=1)).clamp_min(1e-6)  # [B,26]
        aux_y = (-(p_app.log()) * ymask).sum() / ymask.sum().clamp_min(1.0)
        # REPEAT: push down P(committed guess == any past guess)
        p_rep = torch.zeros(len(bs), device=DEV)
        for p in range(MAX_PAST):
            pw = gprobs.gather(-1, past_idx[:, p, :].unsqueeze(-1)).squeeze(-1)  # [B,5] P(w[j]@j)
            p_rep = p_rep + pw.prod(dim=1) * past_valid[:, p]
        has_past = (past_valid.sum(-1) > 0).float()
        aux_r = (-(1.0 - p_rep.clamp(0, 1 - 1e-6)).log() * has_past).sum() / has_past.sum().clamp_min(1.0)
        loss = imit + LAM_V * aux_v + LAM_G * aux_g + LAM_Y * aux_y + LAM_R * aux_r
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()
    if epoch % 3 == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve)
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/cot_eph_constraint.pt", model, opt, epoch, SFTConfig())
            flag = "  <- best, saved"
        else:
            load_checkpoint("runs/cot_eph_constraint.pt", model)
            flag = "  (reverted)"
        print(f"  epoch {epoch:>2}  lr={sched.get_last_lr()[0]:.1e}  loss={float(loss.detach()):.3f} "
              f"(g={float(aux_g.detach()):.3f} y={float(aux_y.detach()):.3f} r={float(aux_r.detach()):.3f})  "
              f"win={m['win']:.3f} valid={m['valid']:.3f} green-break={m['gbreak']:.3f} repeat={m['repeat']:.3f} "
              f"avg={m['avg']:.2f}{flag}", flush=True)

print(f"\n=== constraint-aux: full held-out ({len(held)}, 6-row), best ckpt ===", flush=True)
bm = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_constraint.pt", bm)
f = evaluate(bm, tuple(held))
print(f"  held-out @6-row : win {f['win']:.3f}  ({int(round(f['win'] * len(held)))}/{len(held)})  "
      f"valid {f['valid']:.3f}  green-break {f['gbreak']:.3f}  repeat {f['repeat']:.3f}  avg {f['avg']:.2f}   "
      f"[base 0.616, DPO 0.631]", flush=True)
print("\n=== sample games ===", flush=True)
bm.eval()
for s in held[:8]:
    g = play(bm, s)
    out = [f"  {g.secret} [{g.status.value} {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[CONSTRAINT DONE]", flush=True)
