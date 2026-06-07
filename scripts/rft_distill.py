"""RFT / on-policy self-distillation (stage-2 driver, not committed): make greedy match its own search.

Hypothesis (questioned hard before building): plain STaR/RFT on TRAIN is redundant — the near-optimal
teacher already wins ~all train secrets, so adding the model's wins on the SAME secrets adds no coverage
(this is likely why info-gain-XIT was null). The ONE thing that is different and might transfer: the
teacher's winning trajectories are OFF-policy (the teacher's argmax word is not the model's argmax, so
greedy can't always reproduce them), whereas the model's OWN sampled winning games use words the model
already assigns high probability to (ON-policy → greedy-reproducible). best-of-N hits ~0.70 on held-out
TEST that greedy gets 0.281 on — the capability generalizes; greedy just doesn't surface it. So we
distill the model's own per-turn best-of-N VALID-VOTE winning games (on-policy, valid, consensus) back
into the greedy policy, anchored by a slice of teacher games to avoid forgetting.

Honesty: rollouts use the model's own samples + public spelling validity + majority vote (NO peek at the
secret). "Winning" is judged against TRAIN secrets only (train answers are usable). Eval = free-gen greedy
on the disjoint TEST split with ZERO inference rules. Warm-start from stage-1; report best-on-VAL TEST.

Env: RFT_N (best-of-N per turn, default 16), RFT_SECRETS (cap train secrets, default all),
RFT_EPOCHS (default 15), RFT_LR (default 1e-4), RFT_TEMP (default 1.0), RFT_TEACHER (anchor passes, default 2).
-> runs/rft_stage2.pt
"""

from __future__ import annotations

import os
import statistics
from collections import Counter
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
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)  # ~50M, == stage-1
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
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}
STAGE1 = "runs/cot_eph_aux_fair.pt"

N = int(os.environ.get("RFT_N", "16"))
TEMP = float(os.environ.get("RFT_TEMP", "1.0"))
EPOCHS = int(os.environ.get("RFT_EPOCHS", "15"))
LR = float(os.environ.get("RFT_LR", "1e-4"))
TEACHER_PASSES = int(os.environ.get("RFT_TEACHER", "2"))


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
    """Greedy ephemeral-CoT free-gen play (the HONEST eval path — no rules)."""
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
def sample_guesses(model, prompt, n, temp=TEMP, max_new=48):
    """Batched: sample n free-gen guesses (ephemeral think then commit). Returns 5-letter words."""
    maxL = len(prompt) + max_new
    ids = torch.full((n, maxL), tok.pad_id, dtype=torch.long, device=DEV)
    ids[:, : len(prompt)] = torch.tensor(prompt, device=DEV)
    cur = torch.full((n,), len(prompt), device=DEV)
    committed, collected, done = [False] * n, [[] for _ in range(n)], [False] * n
    ar = torch.arange(n, device=DEV)
    for _ in range(max_new):
        if all(done):
            break
        logits = model.forward(ids)[ar, cur - 1]
        probs = torch.softmax(logits[:, ALLOWED_GEN] / temp, dim=-1)
        choice = torch.multinomial(probs, 1).squeeze(1)
        for i in range(n):
            if done[i]:
                continue
            t = int(ALLOWED_GEN[int(choice[i])])
            ids[i, int(cur[i])] = t
            cur[i] = int(cur[i]) + 1
            if committed[i]:
                if t in LETTER_SET:
                    collected[i].append(t)
                if len(collected[i]) >= 5:
                    done[i] = True
            elif t == tok.guess_id:
                committed[i] = True
            if int(cur[i]) >= maxL:
                done[i] = True
    return ["".join(tok.id_to_token(t) for t in c[:5]) for c in collected if len(c) >= 5]


@torch.no_grad()
def play_bon(model, secret, n):
    """Per-turn best-of-N valid-vote game (on-policy, honest: vote never sees the secret)."""
    g = Game(secret)
    while g.status is Status.ONGOING:
        words = sample_guesses(model, board_only(g.turns), n)
        pool = [w for w in words if is_valid(w)]
        if pool:
            guess = Counter(pool).most_common(1)[0][0]
        elif words:
            guess = Counter(words).most_common(1)[0][0]
        else:
            guess = "zzzzz"
        g.guess(guess)
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
    cap = int(os.environ.get("RFT_SECRETS", str(len(train))))
    secrets = tuple(train[:cap])
    PROG = "runs/rft_stage2_progress.jsonl"
    if os.path.exists(PROG):
        os.remove(PROG)
    print(f"[rft] N={N} temp={TEMP} epochs={EPOCHS} lr={LR} teacher_passes={TEACHER_PASSES} "
          f"|secrets|={len(secrets)} |VAL|={len(VAL)} |TEST|={len(TEST)}", flush=True)

    model = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(STAGE1, model)  # warm-start from stage-1
    base = evaluate(model, VAL)
    print(f"[rft] stage-1 warm-start VAL win={base['win']:.3f} valid={base['valid']:.3f}", flush=True)

    # 1) ON-POLICY rollouts: keep the model's own best-of-N winning games on TRAIN secrets.
    print(f"[rft] rolling out best-of-{N} on {len(secrets)} train secrets …", flush=True)
    model.eval()
    win_games, n_bon_win, n_greedy_win = [], 0, 0
    for i, s in enumerate(secrets):
        gb = play_bon(model, s, N)
        if gb.won:
            win_games.append(gb)
            n_bon_win += 1
        gg = play(model, s)  # also bank greedy wins (already-reproducible, reinforces them)
        if gg.won:
            win_games.append(gg)
            n_greedy_win += 1
        if (i + 1) % 200 == 0:
            print(f"      {i + 1}/{len(secrets)}  bon_wins={n_bon_win} greedy_wins={n_greedy_win} "
                  f"games_kept={len(win_games)}", flush=True)
    print(f"[rft] rollout done: bon_win_rate={n_bon_win / len(secrets):.3f} "
          f"greedy_win_rate={n_greedy_win / len(secrets):.3f}  winning_games_kept={len(win_games)}", flush=True)

    # 2) anchor with a slice of teacher games (same recipe as stage-1) to avoid forgetting.
    rng = Random(0)
    anchor = []
    for s in range(TEACHER_PASSES):
        anchor += [tr.game for tr in generate_transcripts(
            secrets, weak_frac=0.5, openers=safe_openers, seed=100 + s, valid_pool=VALID, answer_pool=secrets)]
    print(f"[rft] anchor teacher games={len(anchor)} (passes={TEACHER_PASSES})", flush=True)

    exs = [e for g in (win_games + anchor) for e in game_examples(g, rng)]
    rng.shuffle(exs)
    print(f"[rft] total examples={len(exs)} (rft_games={len(win_games)} + teacher_games={len(anchor)})", flush=True)

    # 3) fine-tune from stage-1 on the on-policy + anchor mix.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    rng2 = Random(0)
    best = base["win"]  # revert-on-regress: never ship worse than stage-1's selected VAL
    saved = False
    VIZ = tuple(held[:24])
    model.train()
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
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/rft_stage2.pt", model, opt, epoch, SFTConfig())
            saved = True
            flag = "  <- best, saved"
        print(f"  epoch {epoch:>2}  loss={float(loss.detach()):.3f}  viz_win={vmet['win']:.3f} "
              f"VAL win={m['win']:.3f} valid={m['valid']:.3f}{flag}", flush=True)
        model.train()

    print("\n=== RFT stage-2: honest TEST (held[96:], free-gen greedy, zero rules) ===", flush=True)
    if not saved:
        print("  [null] never beat stage-1 VAL — on-policy distillation did not help; stage-1 stands.", flush=True)
        print(f"  stage-1 baseline: TEST 0.281 / valid 0.662", flush=True)
        print("\n[RFT DONE]", flush=True)
        return
    b = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint("runs/rft_stage2.pt", b)
    ft, fv, ftr = evaluate(b, TEST), evaluate(b, VAL), evaluate(b, tuple(train[:200]))
    print(f"  TEST  (HONEST headline) : win {ft['win']:.3f} ({int(round(ft['win'] * len(TEST)))}/{len(TEST)}) "
          f"valid {ft['valid']:.3f} avg {ft['avg']:.2f}", flush=True)
    print(f"  VAL   (selected-on)     : win {fv['win']:.3f}", flush=True)
    print(f"  TRAIN[:200] (mem ref)   : win {ftr['win']:.3f}", flush=True)
    print(f"  [ref] stage-1 TEST 0.281/0.662 ; best-of-128 aided 0.719", flush=True)
    print("\n[RFT DONE]", flush=True)


if __name__ == "__main__":
    main()
