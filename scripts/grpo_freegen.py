"""Honest free-gen GRPO on the 50M validity-max base (the one RL experiment worth running).

Why this and not the prior RL runs: every earlier RL attempt was 5M, contaminated-lineage, and treated the
action as "pick/spell a word" — pure sharpening, null. But we just proved the ephemeral CoT is LOAD-BEARING
(no-CoT commit: win 0.073 vs CoT 0.333 at equal validity). So the policy's action is "REASON (free CoT),
then commit 5 letters", and on-policy RL can shape the REASONING, not just the spelling. This rolls out the
full free generation (CoT + <guess> + 5 letters), scores the OUTCOME (win-dominance + non-word penalty),
and does group-relative GRPO (mean-centered advantage, no /std; zero-variance filter; clipped surrogate;
k3 KL to a frozen ref). Honest: reward is engine-outcome only (no solver/dict in the reward), train-only
secrets, eval = greedy CoT free-gen (zero rules) on disjoint TEST. Base + ref = validity_max_v4 (0.333).

Env: GR_BASE (runs/validity_max_v4.pt), GR_OUT (runs/grpo_freegen.pt), GR_ITERS (40), GR_GROUP (8),
GR_BATCH (32 secrets/iter), GR_LR (2e-6), GR_KL (0.04), GR_CLIP (0.2), GR_TEMP (1.0), GR_INNER (2),
GR_SECRETS (cap of train answers used as the secret pool, default all), GR_EVAL (VAL size, 96).
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

import validity_max as V  # reuse board_only, _letters, _fb, play (eval), CFG, VOCAB, tok, LIDS, ALLOWED_GEN
from wordle_slm.config import SFTConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Game, Status
from wordle_slm.model import WordleGenerator
from wordle_slm.sft.train import load_checkpoint, save_checkpoint

DEV = V.DEV
tok = V.tok
LETTER_SET = V.LETTER_SET
ALLOWED = V.ALLOWED_GEN  # [26 letters + THINK + guess_id] -> the policy action space (free CoT + commit)
ALLOWED_CPU = ALLOWED.cpu()

BASE = os.environ.get("GR_BASE", "runs/validity_max_v4.pt")
OUT = os.environ.get("GR_OUT", "runs/grpo_freegen.pt")
ITERS = int(os.environ.get("GR_ITERS", "40"))
GROUP = int(os.environ.get("GR_GROUP", "8"))
BATCH = int(os.environ.get("GR_BATCH", "32"))
LR = float(os.environ.get("GR_LR", "2e-6"))
KL_BETA = float(os.environ.get("GR_KL", "0.04"))
CLIP = float(os.environ.get("GR_CLIP", "0.2"))
TEMP = float(os.environ.get("GR_TEMP", "1.0"))
INNER = int(os.environ.get("GR_INNER", "2"))
VBONUS = float(os.environ.get("GR_VBONUS", "0.0"))  # per-valid-turn partial credit (densifies the signal)
MAXTOK = 40  # generation budget per turn (CoT + commit + letters)


def reward(game: Game) -> float:
    """Outcome reward with the §6.4 dominance ordering: non-word loss < legal loss < win (faster win = more).
    Honest: pure engine outcome, no solver/consistency term. VBONUS adds partial credit per valid guess
    (distinguishes '5 valid then lost' from '1 non-word then lost') — group-relative, so a per-game-varying
    term shifts the advantage; a constant would cancel."""
    vb = VBONUS * sum(1 for t in game.turns if t.valid)
    if game.won:
        return 1.0 + 0.1 * (6 - game.guesses_used) + vb  # 1.1 .. 1.6, faster is better
    if game.turns and not game.turns[-1].valid:
        return -0.5 + vb  # ended on a non-word (the spelling-failure loss we most want to kill)
    return 0.0 + vb  # legal loss (all valid, never solved)


@torch.no_grad()
def _gen_turn(model, contexts, gen):
    """Batched lockstep free generation for one turn over many games. contexts: list[list[int]] (each a
    board_only(visible) prefix). Returns (generated_tokens per game, guess_words per game)."""
    n = len(contexts)
    seqs = [list(c) for c in contexts]
    start = [len(c) for c in contexts]
    committed = [False] * n
    guess: list[list[int]] = [[] for _ in range(n)]
    done = [False] * n
    for _ in range(MAXTOK):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        L = max(len(seqs[i]) for i in active)
        ids = torch.full((len(active), L), tok.pad_id, dtype=torch.long, device=DEV)
        last = []
        for j, i in enumerate(active):
            ids[j, : len(seqs[i])] = torch.tensor(seqs[i], device=DEV)
            last.append(len(seqs[i]) - 1)
        logits = model.forward(ids)[torch.arange(len(active), device=DEV), torch.tensor(last, device=DEV)]
        al = torch.softmax(logits[:, ALLOWED] / TEMP, dim=-1).cpu()  # multinomial on CPU (MPS lesson)
        pick = torch.multinomial(al, 1, generator=gen).squeeze(1)  # [n_active]
        chosen = ALLOWED_CPU[pick].tolist()
        for j, i in enumerate(active):
            t = int(chosen[j])
            seqs[i].append(t)
            if committed[i]:
                if t in LETTER_SET:
                    guess[i].append(t)
                if len(guess[i]) >= 5:
                    done[i] = True
            elif t == tok.guess_id:
                committed[i] = True
    generated = [seqs[i][start[i]:] for i in range(n)]
    words = ["".join(tok.id_to_token(t) for t in guess[i][:5]) if len(guess[i]) >= 5 else "" for i in range(n)]
    return generated, words


@torch.no_grad()
def rollout(model, secrets, gen):
    """For each secret, GROUP sampled trajectories. Returns rows: (full_seq, ctx_len, adv-placeholder, game)
    grouped; advantage filled after. A trajectory contributes one row PER TURN (context+generated)."""
    games = [Game(s) for s in secrets for _ in range(GROUP)]  # len = N*GROUP, grouped contiguously
    visibles: list[list] = [[] for _ in games]
    alive = [True] * len(games)  # a non-word stops feeding (clean protocol), even if turns remain
    turn_rows: list[list[tuple[list[int], int]]] = [[] for _ in games]  # per game: list of (full_seq, ctx_len)
    for _turn in range(6):
        idx = [i for i in range(len(games)) if alive[i] and games[i].status is Status.ONGOING]
        if not idx:
            break
        contexts = [V.board_only(visibles[i]) for i in idx]
        generated, words = _gen_turn(model, contexts, gen)
        for k, i in enumerate(idx):
            ctx = contexts[k]
            turn_rows[i].append((ctx + generated[k], len(ctx)))
            w = words[k]
            t = games[i].guess(w if len(w) == 5 else "zzzzz")
            if t.valid:
                visibles[i].append(t)
            else:
                alive[i] = False  # non-word: like play(), greedy would repeat from unchanged context -> stop
    return games, turn_rows


def _logp_rows(model, rows, grad):
    """Per-row sum of log p over the GENERATED positions (>= ctx_len). rows: list[(seq, ctx_len)].
    Returns a flat tensor of per-token logp and a row-index per token (for grouping), plus token ids list.
    grad=False wraps in no_grad. Chunked to bound memory."""
    out_logp, out_row, out_tok = [], [], []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        order = sorted(range(len(rows)), key=lambda r: len(rows[r][0]))
        for s in range(0, len(order), 64):
            chunk = order[s : s + 64]
            L = max(len(rows[r][0]) for r in chunk)
            ids = torch.full((len(chunk), L), tok.pad_id, dtype=torch.long, device=DEV)
            for j, r in enumerate(chunk):
                ids[j, : len(rows[r][0])] = torch.tensor(rows[r][0], device=DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [chunk, L-1]
            for j, r in enumerate(chunk):
                seq, cl = rows[r]
                # target token at pos p (1..len-1) generated iff p >= cl
                for p in range(max(cl, 1), len(seq)):
                    out_logp.append(tgt[j, p - 1])
                    out_row.append(r)
                    out_tok.append(seq[p])
    return out_logp, out_row, out_tok


def evaluate(model, secrets):
    model.eval()
    games = [V.play(model, s) for s in secrets]  # the REAL eval: greedy CoT free-gen, clean protocol, no rules
    wins = [g for g in games if g.won]
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": sum(is_valid(t.guess) for g in games for t in g.turns) / n if n else 0.0,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def main():
    train, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    cap = int(os.environ.get("GR_SECRETS", str(len(train))))
    pool = list(train[:cap])
    print(f"[grpo] base={BASE} iters={ITERS} group={GROUP} batch={BATCH} lr={LR} kl={KL_BETA} clip={CLIP} "
          f"temp={TEMP} inner={INNER} |pool|={len(pool)}", flush=True)

    model = WordleGenerator(V.CFG, V.VOCAB).to(DEV)
    load_checkpoint(BASE, model)
    ref = WordleGenerator(V.CFG, V.VOCAB).to(DEV)
    load_checkpoint(BASE, ref)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    base = evaluate(model, VAL)
    best = base["win"]
    print(f"[grpo] base VAL win={base['win']:.3f} valid={base['valid']:.3f} avg={base['avg']:.2f}", flush=True)
    save_checkpoint(OUT, model, opt, -1, SFTConfig())  # ensure OUT exists even if RL never beats base

    rng = Random(0)
    cpu_gen = torch.Generator()  # CPU generator for multinomial
    cpu_gen.manual_seed(0)

    print("[grpo] iter  reward(mean) win% nonword% | VAL win/valid", flush=True)
    for it in range(ITERS):
        secrets = rng.sample(pool, BATCH)
        model.eval()
        games, turn_rows = rollout(model, secrets, cpu_gen)
        # rewards + group-relative advantage (mean-center, NO /std), zero-variance filter
        rewards = [reward(g) for g in games]
        adv = [0.0] * len(games)
        kept_rows, kept_adv = [], []
        for gi in range(0, len(games), GROUP):
            grp = rewards[gi : gi + GROUP]
            mean = sum(grp) / len(grp)
            if max(grp) - min(grp) < 1e-6:
                continue  # zero variance -> no learning signal
            for j in range(GROUP):
                a = grp[j] - mean
                adv[gi + j] = a
                for (seq, cl) in turn_rows[gi + j]:
                    kept_rows.append((seq, cl))
                    kept_adv.append(a)
        won = sum(g.won for g in games)
        nonword = sum(1 for g in games if not g.won and g.turns and not g.turns[-1].valid)
        rmean = sum(rewards) / len(rewards)
        if not kept_rows:
            print(f"  {it:>3}  r={rmean:+.3f} win {won / len(games):.2f} nw {nonword / len(games):.2f} | (all-zero-variance, skip)", flush=True)
            continue

        # old + ref per-token logp (detached)
        old_lp, rows_idx, _ = _logp_rows(model, kept_rows, grad=False)
        ref_lp, _, _ = _logp_rows(ref, kept_rows, grad=False)
        old_lp = torch.stack(old_lp).detach()
        ref_lp = torch.stack(ref_lp).detach()
        tok_adv = torch.tensor([kept_adv[r] for r in rows_idx], device=DEV)

        # NOTE: stay in eval() (dropout OFF) for the update so log-probs are consistent with old_lp/ref_lp
        # — gradients still flow via enable_grad in _logp_rows. (Train-mode dropout corrupts the ratio.)
        for _inner in range(INNER):
            new_lp_list, ridx, _ = _logp_rows(model, kept_rows, grad=True)
            new_lp = torch.stack(new_lp_list)
            ratio = torch.exp(new_lp - old_lp)
            pg = -torch.minimum(ratio * tok_adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * tok_adv)
            d = ref_lp - new_lp
            kl = torch.exp(d) - d - 1.0  # k3 KL(policy||ref) estimator, per token
            loss = (pg + KL_BETA * kl).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if it % 4 == 0 or it == ITERS - 1:
            m = evaluate(model, VAL)
            flag = ""
            if m["win"] > best:
                best = m["win"]
                save_checkpoint(OUT, model, opt, it, SFTConfig())
                flag = "  <- best, saved"
            print(f"  {it:>3}  r={rmean:+.3f} win {won / len(games):.2f} nw {nonword / len(games):.2f} | "
                  f"VAL win {m['win']:.3f} valid {m['valid']:.3f}{flag}", flush=True)

    print("\n=== GRPO free-gen: honest TEST (held[96:], greedy CoT free-gen, zero rules) ===", flush=True)
    b = WordleGenerator(V.CFG, V.VOCAB).to(DEV)
    load_checkpoint(OUT, b)
    mt, mv = evaluate(b, TEST), evaluate(b, VAL)
    print(f"  TEST win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  VAL  win {mv['win']:.3f} valid {mv['valid']:.3f}", flush=True)
    print(f"  [bar] base validity_max_v4 = VAL {base['win']:.3f} ; clean TEST 0.338", flush=True)
    print("\n[GRPO DONE]", flush=True)


if __name__ == "__main__":
    main()
