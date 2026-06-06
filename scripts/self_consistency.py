"""Self-consistency probe (driver, not committed): convert the commit gap into wins, honestly.

The bottleneck is now the COMMIT gap (reachability 99.5%, pass@N >> greedy): the model can find the
line but greedy commits wrong. Self-consistency attacks it with ZERO external info: at each turn sample
N reasoning+guess traces and commit the MAJORITY-VOTED guess (marginalize over reasoning). Strictly
honest — no dictionary, no consistency filter, no candidate list; just the model's own samples.

Reports, on held-out (6-row): greedy (baseline) vs self-consistency vote vs pass@N (ceiling = any of N
sampled full games wins). If vote >> greedy, the next step is to DISTILL the voted answers back into
greedy (zero inference cost). Base = runs/cot_eph_aux.pt (the 0.616 honest best).
"""

from __future__ import annotations

import statistics
from collections import Counter
from random import Random  # noqa: F401  (parity with other drivers)

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint

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
N_VOTE, TEMP, ROWS, N_EVAL = 12, 0.9, 6, 150
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
def batched_generate(model, prompts, temp, max_new=44):
    B = len(prompts)
    lens0 = [len(p) for p in prompts]
    maxL = max(lens0) + max_new
    ids = torch.full((B, maxL), tok.pad_id, dtype=torch.long, device=DEV)
    for i, p in enumerate(prompts):
        ids[i, : lens0[i]] = torch.tensor(p, device=DEV)
    cur = torch.tensor(lens0, device=DEV)
    committed, collected, done = [False] * B, [[] for _ in range(B)], [False] * B
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
            if committed[i]:
                if t in LETTER_SET:
                    collected[i].append(t)
                if len(collected[i]) >= 5:
                    done[i] = True
            elif t == tok.guess_id:
                committed[i] = True
            if int(cur[i]) >= maxL:
                done[i] = True
    return collected


def _word(c):
    w = "".join(tok.id_to_token(t) for t in c[:5])
    return w if len(w) == 5 else "zzzzz"


@torch.no_grad()
def play_greedy(model, secret, rows=ROWS):
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
        g.guess(_word(guess))
    return g


@torch.no_grad()
def play_vote(model, secret, n, temp, rows=ROWS):
    """Self-consistency: each turn, commit the majority-voted guess over n sampled traces."""
    g = Game(secret, max_guesses=rows)
    while g.status is Status.ONGOING:
        collected = batched_generate(model, [board_only(g.turns)] * n, temp)
        words = [_word(c) for c in collected]
        g.guess(Counter(words).most_common(1)[0][0])  # pure majority vote — no dictionary/filter
    return g


@torch.no_grad()
def pass_at_n(model, secret, n, temp, rows=ROWS):
    """Ceiling: n independent sampled full games — did ANY win."""
    games = [Game(secret, max_guesses=rows) for _ in range(n)]
    for _ in range(rows):
        active = [i for i, gg in enumerate(games) if gg.status is Status.ONGOING]
        if not active:
            break
        collected = batched_generate(model, [board_only(games[i].turns) for i in active], temp)
        for j, i in enumerate(active):
            games[i].guess(_word(collected[j]))
    return any(gg.won for gg in games)


_, held = split(seed=0)
secrets = tuple(held[:N_EVAL])
print(f"[load] runs/cot_eph_aux.pt (0.616 base)  N_vote={N_VOTE} temp={TEMP} rows={ROWS} eval={N_EVAL}", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_eph_aux.pt", model)
model.eval()

g_wins = v_wins = p_wins = 0
g_avg, v_avg = [], []
flips = []  # secrets where vote won but greedy lost
for k, s in enumerate(secrets):
    gg = play_greedy(model, s)
    gv = play_vote(model, s, N_VOTE, TEMP)
    pn = pass_at_n(model, s, N_VOTE, TEMP)
    g_wins += gg.won
    v_wins += gv.won
    p_wins += pn
    if gg.won:
        g_avg.append(gg.guesses_used)
    if gv.won:
        v_avg.append(gv.guesses_used)
    if gv.won and not gg.won:
        flips.append((s, gv))
    if k % 25 == 0 or k == len(secrets) - 1:
        n = k + 1
        print(f"  ...{n}/{len(secrets)}  greedy={g_wins / n:.3f}  vote={v_wins / n:.3f}  pass@{N_VOTE}={p_wins / n:.3f}", flush=True)

n = len(secrets)
print("\n=== SELF-CONSISTENCY PROBE (held-out 6-row) ===", flush=True)
print(f"  greedy           : win {g_wins / n:.3f}  avg {statistics.mean(g_avg):.2f}", flush=True)
print(f"  self-consistency : win {v_wins / n:.3f}  avg {statistics.mean(v_avg):.2f}   (vote over {N_VOTE}, honest)", flush=True)
print(f"  pass@{N_VOTE} ceiling  : win {p_wins / n:.3f}", flush=True)
print(f"  => vote - greedy = {(v_wins - g_wins) / n:+.3f}   |   pass@N - vote = {(p_wins - v_wins) / n:+.3f}", flush=True)
print(f"\n  {len(flips)} secrets where VOTE won but greedy lost:", flush=True)
for s, gv in flips[:6]:
    print(f"    {s}: " + " ".join(t.guess for t in gv.turns), flush=True)
print("\n[SC DONE]", flush=True)
