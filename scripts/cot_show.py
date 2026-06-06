"""Show the model's chain-of-thought verbatim + integrity A/B (teacher-context vs honest self-context).

Loads the best CoT model (runs/cot_50m.pt) and plays held-out games, printing the FULL token stream
the model generates each turn (<think> candidates + the committed <GUESS> word), decoded readably and
once as raw tokens. ALSO runs the integrity check: the training serialization puts TEACHER-built
think-blocks in the context for PAST turns; cot_play rebuilds that at inference (uses the consistency
filter to reconstruct past reasoning). The honest version carries the model's OWN generated think-
blocks forward as context — no consistency filter at inference at all. We compare win rates.
"""

from __future__ import annotations

import statistics
from random import Random

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, load_answers, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import consistent_candidates
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
ANSWERS = load_answers()
K_CANDS = 3
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def pick_cands(history, guess, rng):
    cons = [w for w in consistent_candidates(history, ANSWERS) if w != guess]
    rng.shuffle(cons)
    cands = [guess] + cons[: K_CANDS - 1]
    rng.shuffle(cands)
    return cands


def cot_prompt(turns, rng):  # TEACHER-context: rebuilds past think-blocks via the consistency filter
    ids = [tok.bos_id]
    for k, turn in enumerate(turns):
        for c in pick_cands(turns[:k], turn.guess, rng):
            ids += [THINK, *_letters(c)]
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


@torch.no_grad()
def gen_turn(model, seq, budget=60):
    """Generate the model's own <think>…<GUESS>word for the current step; return (trace, word)."""
    start = len(seq)
    guess, committed = [], False
    for _ in range(budget):
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
    return seq[start:], (word if len(word) == 5 else "zzzzz")


@torch.no_grad()
def play_teacher(model, secret):  # past turns reconstructed with teacher think-blocks (the 0.456 path)
    g = Game(secret)
    rng = Random(0)
    while g.status is Status.ONGOING:
        _, word = gen_turn(model, cot_prompt(g.turns, rng))
        g.guess(word)
    return g


@torch.no_grad()
def play_honest(model, secret, record=False):  # rolling self-context: model's OWN think-blocks only
    g = Game(secret)
    seq = [tok.bos_id]
    traces = []
    while g.status is Status.ONGOING:
        trace, word = gen_turn(model, seq)
        g.guess(word)
        if record:
            traces.append((trace, g.turns[-1]))
        seq += _fb(g.turns[-1]) + [tok.sep_id]  # append real feedback; continue rolling
    return (g, traces) if record else g


def render(trace):
    """Decode a generated token stream into readable CoT."""
    parts, i = [], 0
    while i < len(trace):
        t = trace[i]
        if t == THINK and i + 5 < len(trace):
            parts.append("💭 " + "".join(tok.id_to_token(x) for x in trace[i + 1 : i + 6]))
            i += 6
        elif t == tok.guess_id and i + 5 < len(trace):
            parts.append("🎯 GUESS " + "".join(tok.id_to_token(x) for x in trace[i + 1 : i + 6]))
            i += 6
        else:
            i += 1
    return parts


def win_rate(fn, secrets):
    games = [fn(model, s) for s in secrets]
    wins = [g for g in games if g.won]
    v = sum(is_valid(t.guess) for g in games for t in g.turns)
    n = sum(len(g.turns) for g in games)
    return len(wins) / len(games), v / n, (statistics.mean(g.guesses_used for g in wins) if wins else float("nan"))


train, held = split(seed=0)
print("[load] runs/cot_50m.pt (the 0.456 best) …", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/cot_50m.pt", model)
model.eval()

print("\n================ FULL CoT TRACES (honest self-context, greedy) ================", flush=True)
for secret in ["sleek", "joist", "pride"]:
    g, traces = play_honest(model, secret, record=True)
    print(f"\n  secret = {secret}   [{g.status.value} in {g.guesses_used}]", flush=True)
    for turn_i, (trace, turn) in enumerate(traces, 1):
        fb = "❌invalid" if turn.feedback is None else "".join(SQ[c] for c in turn.feedback)
        print(f"    --- turn {turn_i} — the model generates: ---", flush=True)
        for line in render(trace):
            print(f"        {line}", flush=True)
        print(f"        => played '{turn.guess}'  {fb}", flush=True)

# one raw-token dump so the literal stream is visible
g, traces = play_honest(model, "pride", record=True)
raw = traces[0][0]
print("\n  raw token ids (turn 1, 'pride'):", raw, flush=True)
print("  raw tokens     :", [("💭" if t == THINK else "🎯GUESS" if t == tok.guess_id else tok.id_to_token(t)) for t in raw], flush=True)

print("\n================ INTEGRITY A/B (held-out 120) ================", flush=True)
probe = tuple(held[:120])
tw, tv, ta = win_rate(play_teacher, probe)
hw, hv, ha = win_rate(play_honest, probe)
print(f"  teacher-context (past think-blocks via consistency filter) : win {tw:.3f}  valid {tv:.3f}  avg {ta:.2f}", flush=True)
print(f"  honest self-context (model's OWN think-blocks, no filter)   : win {hw:.3f}  valid {hv:.3f}  avg {ha:.2f}", flush=True)
print(f"  => delta (honest - teacher) = {hw - tw:+.3f}", flush=True)
print("\n[SHOW DONE]", flush=True)
