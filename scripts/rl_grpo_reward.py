"""GRPO with the FIXED shaped reward, on the best model (driver, not committed).

Same stabilized token-level GRPO as rl_grpo_polish (eval-mode forward, eps=0.2, k3 KL beta=0.05, lr
5e-6, group-relative advantage no /std, zero-variance filter), BUT the reward is now the full shaped
`compute_reward` — including the NEW repeat + drop-present (yellow-reuse) penalties — instead of the
ad-hoc win+speed-invalid. Tests whether GRPO, now that its rollouts' repeats/yellow-drops/invalids are
penalised, can shape those behaviours on top of the best model. Honest: train TRAIN answers, eval
held-out greedy, no rules. Starts from runs/dpo.pt (0.631) -> runs/rl_grpo_reward.pt.
"""

from __future__ import annotations

import copy
import os
import statistics
import time
from random import Random

import torch

from viz_progress import append_epoch  # scripts/ helper: per-update rollouts + grades -> dashboard
from wordle_slm.config import ModelConfig, RewardConfig, SFTConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.rl.reward import compute_reward
from wordle_slm.sft.train import load_checkpoint, save_checkpoint

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
ROWS = 10
TEMP = 1.0
EPS, BETA_KL = 0.2, 0.05  # stronger KL anchor to the good ref (the unstable run used 0.01)
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


@torch.no_grad()
def batched_generate(model, prompts, max_new=44, temp=TEMP):
    B = len(prompts)
    lens0 = [len(p) for p in prompts]
    maxL = max(lens0) + max_new
    ids = torch.full((B, maxL), tok.pad_id, dtype=torch.long, device=DEV)
    for i, p in enumerate(prompts):
        ids[i, : lens0[i]] = torch.tensor(p, device=DEV)
    cur = torch.tensor(lens0, device=DEV)
    committed, collected, gen, done = [False] * B, [[] for _ in range(B)], [[] for _ in range(B)], [False] * B
    ar = torch.arange(B, device=DEV)
    for _ in range(max_new):
        if all(done):
            break
        logits = model.forward(ids)[ar, cur - 1]
        probs = torch.softmax(logits[:, ALLOWED_GEN] / temp, dim=-1).cpu()
        choice = torch.multinomial(probs, 1).squeeze(1)
        for i in range(B):
            if done[i]:
                continue
            t = int(ALLOWED_GEN[int(choice[i])])
            ids[i, int(cur[i])] = t
            cur[i] = int(cur[i]) + 1
            gen[i].append(t)
            if committed[i]:
                if t in LETTER_SET:
                    collected[i].append(t)
                if len(collected[i]) >= 5:
                    done[i] = True
            elif t == tok.guess_id:
                committed[i] = True
            if int(cur[i]) >= maxL:
                done[i] = True
    return gen, collected


@torch.no_grad()
def rollout_batch(model, secrets, rows=ROWS):
    games = [Game(s, max_guesses=rows) for s in secrets]
    records: list[list] = [[] for _ in secrets]
    for _ in range(rows):
        active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
        if not active:
            break
        prompts = [board_only(games[i].turns) for i in active]
        gen, collected = batched_generate(model, prompts)
        for j, i in enumerate(active):
            word = "".join(tok.id_to_token(t) for t in collected[j][:5])
            games[i].guess(word if len(word) == 5 else "zzzzz")
            records[i].append((prompts[j] + gen[j], [False] * len(prompts[j]) + [True] * len(gen[j])))
    return games, records


RCFG = RewardConfig()  # full shaped reward incl. the new repeat + drop-present penalties


def reward(g):
    return compute_reward(g, RCFG).total


@torch.no_grad()
def play(model, secret, rows):
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


def evaluate(model, secrets, rows):
    model.eval()
    games = [play(model, s, rows) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return {"win": len(wins) / len(games), "valid": v / n,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def grpo_step(model, ref, opt, seqs_adv):
    """One GRPO update over (seq, amask, advantage) action-trajectory pieces."""
    model.eval()  # NO dropout in the policy forward -> clean ratio/KL (the instability fix)
    rng = Random(0)
    rng.shuffle(seqs_adv)
    total_tok = sum(sum(m) for _, m, _ in seqs_adv)
    if total_tok == 0:
        return 0.0, 0.0
    opt.zero_grad()
    surr_sum, kl_sum = 0.0, 0.0
    for c0 in range(0, len(seqs_adv), 64):
        chunk = seqs_adv[c0 : c0 + 64]
        L = max(len(s) for s, _, _ in chunk)
        ids = torch.full((len(chunk), L), tok.pad_id, dtype=torch.long)
        am = torch.zeros((len(chunk), L))
        adv = torch.zeros((len(chunk), 1))
        for i, (s, m, a) in enumerate(chunk):
            ids[i, : len(s)] = torch.tensor(s)
            am[i, : len(m)] = torch.tensor([float(x) for x in m])
            adv[i, 0] = a
        ids, am, adv = ids.to(DEV), am.to(DEV), adv.to(DEV)
        logits = model.forward(ids)
        logp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            rlogits = ref.forward(ids)
            rlogp = torch.log_softmax(rlogits[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = am[:, 1:]
        ratio = torch.exp(logp - logp.detach())  # =1 at K=1; gradient flows -> REINFORCE w/ baseline
        surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
        d = rlogp - logp
        kl = torch.exp(d) - d - 1.0  # k3, >= 0
        loss = (-(surr * mask).sum() + BETA_KL * (kl * mask).sum()) / total_tok
        loss.backward()
        surr_sum += float((surr * mask).sum().detach())
        kl_sum += float((kl * mask).sum().detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return surr_sum / total_tok, kl_sum / total_tok


train, held = split(seed=0)
curve = tuple(held[:96])
PROG = "runs/rl_grpo_reward_progress.jsonl"  # per-update rollouts+grades -> scripts/live_viz.py
if os.path.exists(PROG):
    os.remove(PROG)  # start each run clean
print("[load] runs/dpo.pt (Phase-1 result) …", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/dpo.pt", model)
ref = copy.deepcopy(model).to(DEV)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)

b6 = evaluate(model, curve, 6)
b10 = evaluate(model, curve, ROWS)
print(f"[base] held6={b6['win']:.3f}  held{ROWS}={b10['win']:.3f} valid={b10['valid']:.3f}", flush=True)

UPDATES, B_SECRETS, G = 40, 8, 8
opt = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.0)  # smaller steps for stability
rngS = Random(7)
order = list(train)
rngS.shuffle(order)
best = b10["win"]
save_checkpoint("runs/rl_grpo_reward.pt", model, opt, 0, SFTConfig())
ptr = 0
for u in range(UPDATES):
    t0 = time.time()
    secs = [order[(ptr + i) % len(order)] for i in range(B_SECRETS)]
    ptr += B_SECRETS
    batch_secrets = [s for s in secs for _ in range(G)]
    model.eval()
    games, records = rollout_batch(model, batch_secrets)
    rewards = [reward(g) for g in games]
    game_adv = []  # per-game group-relative advantage (the GRPO grade), for the live dashboard
    for si in range(B_SECRETS):
        rs0 = [rewards[si * G + k] for k in range(G)]
        m0 = sum(rs0) / len(rs0)
        game_adv.extend(r - m0 for r in rs0)
    seqs_adv, kept_groups, win_n = [], 0, sum(g.won for g in games)
    for si in range(B_SECRETS):
        rs = [rewards[si * G + k] for k in range(G)]
        mean_r = sum(rs) / len(rs)
        if max(rs) - min(rs) < 1e-6:
            continue  # zero-variance group: no learning signal
        kept_groups += 1
        for k in range(G):
            a = rs[k] - mean_r
            for seq, amask in records[si * G + k]:
                seqs_adv.append((seq, amask, a))
    model.train()
    surr, kl = grpo_step(model, ref, opt, seqs_adv) if seqs_adv else (0.0, 0.0)
    eval_win = None
    if u % 8 == 0 or u == UPDATES - 1:
        e6 = evaluate(model, curve, 6)
        e10 = evaluate(model, curve, ROWS)
        eval_win = e10["win"]
        flag = ""
        if e10["win"] > best:
            best = e10["win"]
            save_checkpoint("runs/rl_grpo_reward.pt", model, opt, u + 1, SFTConfig())
            flag = "  <- best, saved"
        print(f"[upd {u:>2}] win/grp {win_n}/{B_SECRETS * G} kept {kept_groups}  surr={surr:.3f} kl={kl:.4f}  "
              f"held6={e6['win']:.3f} held{ROWS}={e10['win']:.3f} valid={e10['valid']:.3f}  ({time.time() - t0:.0f}s){flag}", flush=True)
    else:
        print(f"[upd {u:>2}] win/grp {win_n}/{B_SECRETS * G} kept {kept_groups}  surr={surr:.3f} kl={kl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    # live dashboard: every rollout this update + its grade (reward) and group-relative advantage
    rl_metrics = {"reward_mean": sum(rewards) / len(rewards), "kl": kl, "win_train": win_n / (B_SECRETS * G)}
    if eval_win is not None:
        rl_metrics["eval_win"] = eval_win
    append_epoch(PROG, u, rl_metrics, games, sample=12, kind="rl",
                 grades=[{"reward": rewards[i], "adv": game_adv[i]} for i in range(len(games))])

print(f"\n=== GRPO polish: full held-out ({len(held)}), best checkpoint ===", flush=True)
b = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/rl_grpo_reward.pt", b)
f6 = evaluate(b, tuple(held), 6)
f10 = evaluate(b, tuple(held), ROWS)
print(f"  held-out @6-row  : win {f6['win']:.3f}  ({int(round(f6['win'] * len(held)))}/{len(held)})  valid {f6['valid']:.3f} avg {f6['avg']:.2f}", flush=True)
print(f"  held-out @{ROWS}-row : win {f10['win']:.3f}  ({int(round(f10['win'] * len(held)))}/{len(held)})  valid {f10['valid']:.3f} avg {f10['avg']:.2f}", flush=True)
print("\n[GRPO DONE]", flush=True)
