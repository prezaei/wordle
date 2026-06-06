"""DAgger (driver, not committed): put the model's OWN failure states into training, labeled by the teacher.

The constraint-aux was a no-op because the teacher transcripts never exhibit the failures (no green-break,
no repeat, no invalid word) — so there was no gradient. DAgger fixes that at the source: roll out the
CURRENT model greedily on TRAIN secrets, collect the boards where it does something BAD (emits an invalid
word / repeats a past guess / drops a known green), and add a training example at each such board whose
target is the INFOMAX TEACHER's correct valid guess. Because the teacher never repeats, keeps greens, and
plays valid words, one DAgger pass fixes all three behaviors AT the states the model actually visits.

Honest: states visited by the model, labels from the InfoMax teacher (our standard) over TRAIN secrets;
inference = greedy ephemeral-CoT on held-out, no rules. Fine-tunes runs/dpo.pt (0.631) -> runs/dagger.pt.
"""

from __future__ import annotations

import statistics
from random import Random

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.baselines.policies import InfoMaxGuesser
from wordle_slm.data import is_valid, load_answers, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import consistent_candidates
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft import pretrain_lm, pretrain_words  # noqa: F401
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
TEACHER = InfoMaxGuesser()
K_CANDS = 3
LAM_V = 0.5
N_SECRETS, SAMPLE_S = 1400, 16
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


def example(history, target, rng):
    ids = board_only(history)
    mask = [False] * len(ids)
    for c in pick_cands(history, target, rng):
        ids.append(THINK)
        mask.append(True)
        ids += _letters(c)
        mask += [True] * 5
    ids.append(tok.guess_id)
    mask.append(True)
    ids += _letters(target)
    mask += [True] * 5
    ids.append(tok.eos_id)
    mask.append(False)
    return ids, mask


def teacher_examples(game, rng):
    return [example(game.turns[:k], t.guess, rng) for k, t in enumerate(game.turns)]


def is_bad(turn, history):
    """Invalid word, repeat of a past guess, or drops a known green = a state DAgger should correct."""
    if not turn.valid:
        return True
    if any(turn.guess == pt.guess for pt in history):
        return True
    greens = {}
    for pt in history:
        if pt.valid and pt.feedback:
            for i, c in enumerate(pt.feedback):
                if c is Color.GREEN:
                    greens[i] = pt.guess[i]
    return any(turn.guess[i] != ch for i, ch in greens.items())


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
def batched_generate(model, prompts, temp=0.02, max_new=44):
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


@torch.no_grad()
def rollout_failures(model, secrets):
    """Greedy-roll the model; return DAgger (history, teacher_guess) for every BAD-guess board."""
    out = []
    for c0 in range(0, len(secrets), SAMPLE_S):
        secs = secrets[c0 : c0 + SAMPLE_S]
        games = [Game(s, max_guesses=6) for s in secs]
        for _ in range(6):
            active = [i for i, g in enumerate(games) if g.status is Status.ONGOING]
            if not active:
                break
            cols = batched_generate(model, [board_only(games[i].turns) for i in active])
            for j, i in enumerate(active):
                w = "".join(tok.id_to_token(t) for t in cols[j][:5])
                games[i].guess(w if len(w) == 5 else "zzzzz")
        for g in games:
            for k, turn in enumerate(g.turns):
                hist = g.turns[:k]
                if not is_bad(turn, hist):
                    continue
                cons = consistent_candidates(hist, ANSWERS) if hist else ()
                tgt = TEACHER.choose(hist, cons) if hist else OPENERS[0]
                if is_valid(tgt):
                    out.append((list(hist), tgt))
        if (c0 // SAMPLE_S) % 10 == 0:
            print(f"    [dagger] rolled {c0 + len(secs)}/{len(secrets)} secrets  failure-boards={len(out)}", flush=True)
    return out


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
print("[load] fine-tune runs/dpo.pt (0.631) for DAgger", flush=True)
model = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/dpo.pt", model)
model.eval()

rng = Random(0)
secrets = list(train)
rng.shuffle(secrets)
secrets = secrets[:N_SECRETS]
b = evaluate(model, curve)
print(f"[base] win={b['win']:.3f} valid={b['valid']:.3f}", flush=True)

print("[dagger] rolling out the model to collect failure boards …", flush=True)
fails = rollout_failures(model, secrets)
print(f"[dagger] {len(fails)} failure boards collected", flush=True)
dagger_ex = [example(h, t, rng) for h, t in fails]
print("[teacher-mix] 1 InfoMax pass …", flush=True)
tgames = [tr.game for tr in generate_transcripts(tuple(train), weak_frac=0.2, openers=OPENERS, seed=0)]
teach_ex = [e for g in tgames for e in teacher_examples(g, rng)]
# pool: all DAgger corrections, repeated 2x for emphasis, + teacher mix (breadth)
pool = dagger_ex * 2 + teach_ex
rng.shuffle(pool)
print(f"[data] {len(dagger_ex)} dagger + {len(teach_ex)} teacher = {len(pool)} examples", flush=True)

EPOCHS, BATCH = 10, 128
opt = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=6e-6)
rng2 = Random(0)
best = b["win"]
save_checkpoint("runs/dagger.pt", model, opt, 0, SFTConfig())
for epoch in range(EPOCHS):
    model.train()
    for idx in _batches(len(pool), BATCH, rng2):
        bs = [pool[i] for i in idx]
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
        vpos = (vm.sum(-1) > 0).float() * tmask[:, 1:]
        aux = (-(logp_let.exp() * vm).sum(-1).clamp_min(1e-9).log() * vpos).sum() / vpos.sum().clamp_min(1.0)
        loss = imit + LAM_V * aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()
    if epoch % 2 == 0 or epoch == EPOCHS - 1:
        m = evaluate(model, curve)
        flag = ""
        if m["win"] > best:
            best = m["win"]
            save_checkpoint("runs/dagger.pt", model, opt, epoch, SFTConfig())
            flag = "  <- best, saved"
        else:
            load_checkpoint("runs/dagger.pt", model)
            flag = "  (reverted)"
        print(f"  epoch {epoch:>2}  lr={sched.get_last_lr()[0]:.1e}  loss={float(loss.detach()):.3f}  "
              f"win={m['win']:.3f} valid={m['valid']:.3f} avg={m['avg']:.2f}{flag}", flush=True)

print(f"\n=== DAgger: full held-out ({len(held)}, 6-row), best ckpt ===", flush=True)
bm = WordleGenerator(CFG, VOCAB).to(DEV)
load_checkpoint("runs/dagger.pt", bm)
f = evaluate(bm, tuple(held))
print(f"  held-out @6-row : win {f['win']:.3f}  ({int(round(f['win'] * len(held)))}/{len(held)})  "
      f"valid {f['valid']:.3f}  avg {f['avg']:.2f}   [base 0.616, DPO 0.631]", flush=True)
print("\n=== sample games ===", flush=True)
bm.eval()
for s in held[:8]:
    g = play(bm, s)
    out = [f"  {g.secret} [{g.status.value} {g.guesses_used}]"]
    for t in g.turns:
        out.append(f"      {t.guess}  " + ("❌" if t.feedback is None else "".join(SQ[c] for c in t.feedback)))
    print("\n".join(out), flush=True)
print("\n[DAGGER DONE]", flush=True)
