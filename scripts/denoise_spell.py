"""Idea #2 — denoising / contrastive spelling-sharpening mid-training (honest).

Warm-start the best model (validity_max_v4) and continue training with the SAME game-SFT + trie-aux
recipe PLUS a contrastive spelling-margin term: at each board state, push the autoregressive
log-prob of the REAL committed word above that of a Hamming-1 corruption that is NOT a real word.

Why: the failure analysis (eval_analyze2.py) found 226/228 non-word deaths are ONE letter from a real
word and 78% respect the clues -- the model sits on the real-word manifold edge and falls off at the
commit. The contrastive term sharpens exactly that edge so greedy/sampled decoding snaps to the real
word instead of the off-by-one chimera. Should compound with anti-repeat sampling.

Honesty: positives are train-game guesses (never held-out answers); negatives are synthetic non-words;
no clue->held-out-answer mapping, no dictionary/filter at inference. Spelling exposure to the full dict
(validity) matches the existing aux-trie standard.

Env: DN_EPOCHS(6), DN_LC(1.0 contrast weight), DN_MARGIN(2.0), DN_LR(1e-4), DN_TEACHER(3),
     DN_BASE(runs/validity_max_v4.pt), DN_OUT(runs/denoise_spell.pt), VM_SECRETS(all train).
Then eval honestly:  uv run python scripts/eval_db.py --checkpoint runs/denoise_spell.pt ...
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

import validity_max as vm
from wordle_slm.config import SFTConfig
from wordle_slm.teacher import generate_transcripts

DEV = vm.DEV
tok = vm.tok
LETTER_LO = tok.token_to_id("a")
LIDS = vm.LIDS
VALID_SET = set(vm.VALID)
VALID5 = [w for w in vm.VALID if len(w) == 5]  # full dict, for UNCONDITIONAL spelling sharpening
ALPHA = "abcdefghijklmnopqrstuvwxyz"

EPOCHS = int(os.environ.get("DN_EPOCHS", "4"))
LC = float(os.environ.get("DN_LC", "1.0"))
MARGIN = float(os.environ.get("DN_MARGIN", "3.0"))
PER_GROUP = int(os.environ.get("DN_PER_GROUP", "12"))  # constrained contrastive per k-group
PER_DICT = int(os.environ.get("DN_PER_DICT", "48"))  # unconditional full-dict contrastive
LR = float(os.environ.get("DN_LR", "1e-4"))
BASE = os.environ.get("DN_BASE", "runs/validity_max_v4.pt")
OUT = os.environ.get("DN_OUT", "runs/denoise_spell.pt")


def corrupt(word: str, rng: Random) -> str | None:
    """Hamming-1 change that is NOT a real word (a near-miss non-word). None if none found."""
    for _ in range(12):
        pos = rng.randrange(5)
        new = rng.choice(ALPHA)
        if new == word[pos]:
            continue
        cand = word[:pos] + new + word[pos + 1 :]
        if cand not in VALID_SET:
            return cand
    return None


def commit_logp(model, seqs: torch.Tensor, commit_start: int) -> torch.Tensor:
    """Sum of autoregressive log-probs of the 5 commit letters (26-letter restricted softmax)."""
    logits = model.forward(seqs)  # [B, L, V]
    pred = logits[:, commit_start - 1 : commit_start + 4]  # predicts the 5 commit letters
    lp = torch.log_softmax(pred[:, :, LIDS], dim=-1)  # [B, 5, 26]
    true = seqs[:, commit_start : commit_start + 5] - LETTER_LO  # letter -> 0..25
    return lp.gather(-1, true.unsqueeze(-1)).squeeze(-1).sum(-1)  # [B]


def build_contrast(games, rng):
    """(num_prior_turns k) -> list of (context_ids, real5_ids, corrupt5_ids). Grouped by k so each
    group is equal-length (board_only(prefix)+<GUESS> = 2+12k) -> batch with no padding."""
    by_k: dict[int, list] = {}
    for g in games:
        for k, turn in enumerate(g.turns):
            if not turn.valid:
                continue
            cw = corrupt(turn.guess, rng)
            if cw is None:
                continue
            ctx = vm.board_only(g.turns[:k]) + [tok.guess_id]
            by_k.setdefault(k, []).append(
                (ctx, [tok.token_to_id(c) for c in turn.guess], [tok.token_to_id(c) for c in cw])
            )
    return by_k


def _margin(model, sample, commit_start):
    pos = torch.tensor([ctx + real for ctx, real, _ in sample], device=DEV)
    neg = torch.tensor([ctx + bad for ctx, _, bad in sample], device=DEV)
    return torch.relu(MARGIN - (commit_logp(model, pos, commit_start) - commit_logp(model, neg, commit_start))).mean()


def contrast_loss(model, by_k, dict5, rng, per_group=PER_GROUP, per_dict=PER_DICT):
    """Margin loss = constrained (train-game k-groups) + UNCONDITIONAL full-dict denoising (empty board)."""
    losses = []
    for k, items in by_k.items():  # constrained: at a real board, real word > corruption
        if items:
            sample = [items[i] for i in rng.sample(range(len(items)), min(per_group, len(items)))]
            losses.append(_margin(model, sample, 2 + 12 * k))
    # unconditional: empty board (commit_start=2), ANY dict word (incl held-out as a valid word -> the
    # aux-trie 'knows English' standard; no clue->answer signal) vs a Hamming-1 non-word.
    ctx0 = [tok.bos_id, tok.guess_id]
    uncond = []
    for w in rng.sample(dict5, min(per_dict, len(dict5))):
        cw = corrupt(w, rng)
        if cw is not None:
            uncond.append((ctx0, [tok.token_to_id(c) for c in w], [tok.token_to_id(c) for c in cw]))
    if uncond:
        losses.append(_margin(model, uncond, 2))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=DEV)


def main():
    train, held = vm.split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    cap = int(os.environ.get("VM_SECRETS", str(len(train))))
    secrets = tuple(train[:cap])
    safe_openers = tuple(o for o in vm.OPENERS if o not in set(held))
    print(f"[dn] base={BASE} epochs={EPOCHS} LC={LC} margin={MARGIN} lr={LR} |secrets|={len(secrets)}", flush=True)

    model = vm.WordleGenerator(vm.CFG, vm.VOCAB).to(DEV)
    vm.load_checkpoint(BASE, model)
    base = vm.evaluate(model, VAL)
    print(f"[dn] warm-start VAL win={base['win']:.3f} valid={base['valid']:.3f}", flush=True)

    rng = Random(0)
    games = []
    for s in range(int(os.environ.get("DN_TEACHER", "2"))):
        games += [tr.game for tr in generate_transcripts(
            secrets, weak_frac=0.5, openers=safe_openers, seed=100 + s, valid_pool=vm.VALID, answer_pool=secrets)]
    exs = [e for g in games for e in vm.game_examples(g, rng)]
    by_k = build_contrast(games, rng)
    n_contrast = sum(len(v) for v in by_k.values())
    print(f"[dn] game-examples={len(exs)}  contrastive-pairs={n_contrast} (k-groups {sorted(by_k)})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    rng2 = Random(0)
    best, saved = base["valid"], False
    model.train()
    for epoch in range(EPOCHS):
        tot_im = tot_c = nb = 0.0
        for idx in vm._batches(len(exs), 128, rng2):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            vmask = vm.cot_valid_mask([s for s, _ in bs]).to(DEV)
            ids, tmask = ids.to(DEV), tmask.to(DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            imit = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
            vmk = vmask[:, :-1]
            aux_pos = (vmk.sum(-1) > 0).float() * tmask[:, 1:]
            valid_mass = (logp_let.exp() * vmk).sum(-1).clamp_min(1e-9)
            aux = (-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
            closs = contrast_loss(model, by_k, VALID5, rng2)
            loss = imit + vm.AUX_LAMBDA * aux + LC * closs
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_im += float(imit.item())
            tot_c += float(closs.item())
            nb += 1
        sched.step()
        model.eval()
        m = vm.evaluate(model, VAL)
        flag = ""
        if m["valid"] > best:
            best, saved = m["valid"], True
            vm.save_checkpoint(OUT, model, opt, epoch, SFTConfig())
            flag = "  <- best, saved"
        print(f"  epoch {epoch:>2}  VAL win {m['win']:.3f} / valid {m['valid']:.3f} | "
              f"imit {tot_im / nb:.3f} contrast {tot_c / nb:.3f}{flag}", flush=True)
        model.train()

    if not saved:
        vm.save_checkpoint(OUT, model, opt, EPOCHS - 1, SFTConfig())
        print("[dn] never beat warm-start VAL-valid; saved final anyway", flush=True)
    b = vm.WordleGenerator(vm.CFG, vm.VOCAB).to(DEV)
    vm.load_checkpoint(OUT, b)
    ft = vm.evaluate(b, TEST)
    print(f"\n[dn DONE] greedy TEST win {ft['win']:.3f} ({int(round(ft['win'] * len(TEST)))}/{len(TEST)}) "
          f"valid {ft['valid']:.3f}  (vs v4 greedy 0.332) -> now run eval_db for the versioned greedy+AR numbers", flush=True)


if __name__ == "__main__":
    main()
