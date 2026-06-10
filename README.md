# wordle-slm

Train a small language model **from scratch** and teach it to play Wordle with reinforcement
learning (GRPO). A learning-first project; runs locally on Apple Silicon (PyTorch / MPS).

- **Why & what:** [`PRD.md`](./PRD.md)
- **How (spec):** [`docs/design/wordle-slm.md`](./docs/design/wordle-slm.md)
- **Build plan:** [`docs/design/wordle-slm-plan.md`](./docs/design/wordle-slm-plan.md)
- **Working agreement:** [`AGENTS.md`](./AGENTS.md)

## Setup

```bash
uv sync
uv run pytest
uv run wordle-slm --help
```

## Status

**Best honest model: `runs/validity_max_v4.pt` — held-out free-gen win 0.333** (verified across two seeds:
0.332 / 0.335), on the strict **clean protocol** (greedy free-generation, **no dictionary or any
inference-time rules**, non-words counted-but-not-fed-back, evaluated on the disjoint TEST split that is
never trained on). Reproduce: aux-validity loss (λ6) + constrained self-distillation on all 1,852 train
secrets (`scripts/validity_max.py`).

**The honest arc:** from-scratch char transformer → spell warm-up → SFT on a near-optimal teacher →
in-weights **validity** (the model learns to spell real words) → **maxing in-distribution data**. That
lifted the honest free-gen number **0.243 → 0.333**. The remaining wall is **deduction-generalization**,
bounded by the ~1,852-word answer set — confirmed by an exhaustive set of nulls (RL ×11, DPO, DAgger,
on-policy RFT/STaR, scale-turns-over-past-50M, the green/yellow infill, ensembles). The full lineage is
the [Experiment Log](#experiment-log) and the visual map (`EXPERIMENTS.svg`, 75+ experiments).

*(Aside: with the public dictionary allowed as an inference-time spell-checker — a different, clearly-
labeled tier — the same weights reach ~0.55 (beam-over-trie) to ~0.72 (best-of-N). We keep that separate
from "the model," which is dictionary-free.)*

## Why a transformer + RL for Wordle (a deliberately sub-optimal tool)

**The game.** Wordle: guess a hidden five-letter word in six tries. Each guess scores every letter —
🟩 green (right letter, right spot), 🟨 yellow (right letter, wrong spot), ⬜ gray (not in the word) —
and you deduce the answer from the accumulating clues.

**What we did.** We trained a small (~50M-parameter) decoder-only **transformer from scratch** — no
pretrained weights — to play it: a "learn to spell" warm-up, supervised fine-tuning on teacher games,
then **reinforcement learning (GRPO)**, all running locally on an Apple M5 Max (MPS).

**Why this tool, when it's the wrong one.** Wordle does not need a language model. The optimal solver is
a few lines of information theory — at each turn, pick the guess that most evenly splits the
still-possible answers (max-entropy / minimax) — and it wins essentially every game in ~3.6 guesses. A
from-scratch transformer is wildly overkill *and worse at the task.* We chose it **on purpose.** The
goal was never to *solve* Wordle; it was to **learn the machinery hands-on** — tokenization, training a
transformer from scratch, the SFT→RL pipeline, GRPO, reward shaping, the discipline of honest held-out
evaluation — i.e. **the same stack the frontier labs (including Microsoft AI / MAI) use to build real
models** — and to **see how far an AI agent could drive that research loop itself** (propose
experiments, analyze results, adversarially audit its own conclusions).

**Why Wordle is the right sandbox for the wrong tool.** Precisely *because* Wordle has a known-optimal
classical solution, every failure of the LLM approach is **legible**: you can watch the model fail to
spell, drift mid-word, memorize the training answers instead of generalizing, and watch RL go flat — and
understand exactly *why*. Small vocabulary, an exact and cheap-to-compute reward, fully reproducible
locally: a clean, fully-verifiable lab for exercising (and stress-testing) these techniques, where the
answer key is always there to check our honesty against.

## How this was built — recursive self-improvement, with human steering

This model wasn't built from a fixed recipe but through a tight **iterate → analyze → self-critique**
loop over one week (2026-06-02 → 06-06). An AI agent proposed an approach, ran it locally on the M5
Max, measured the honest held-out result, formed a hypothesis about what was capping it, and ran the
next experiment — dozens of times. The loop was deliberately **self-correcting**: at key moments the
agent turned a structured **adversarial-investigation team** on its *own* prior conclusions — a Lead, a
Code-Analyst, an Explorer, and a Devil's Advocate, each required to back every claim with a `file:line`
or tool-output receipt. That self-audit repeatedly caught the agent's *own* mistakes: it found four
held-out-contamination channels leaking the answer set into training, **overturned its earlier
"the leaks are inert" verdict** once a clean re-run collapsed the headline from 0.62 to 0.17, and later
proved a risky refactor was bit-identical *while* catching a real config regression the optimistic
checks had missed. This is effectively **recursive self-improvement** — the AI reasoning over its own
code, its own results, and even its own earlier reasoning.

The human's role wasn't to write code or tune hyperparameters; it was to supply the **values and the
hard questions** the AI couldn't set for itself. *"Never train on the answer set." "Ranking a
pre-filtered candidate list is cheating — make the model genuinely generate words." "How do you expect
it to learn the words if you hold them all out?"* — that last correction reframed the entire honesty
regime (the model may *know* the dictionary; only *answer-hood* is held out) and unlocked the jump from
an over-hobbled 0.17 to a fair-honest **0.28**. Each human nudge redirected the AI's search: toward
chain-of-thought, an auxiliary spelling loss, a diagnostic that *proved* the model already knows the
words and only mis-spells them under pressure (constrained decode: **0.436 win / 1.0 valid**), and the
on-policy distillation + RL now closing that last gap. The full chronological record — every experiment,
every number, every adversarial audit and the times it overturned a prior conclusion — is below in the
**[Experiment Log](#experiment-log)**, with deep-dives in the
**[validity push report](./VALIDITY_REPORT.md)** and the
**[overnight clean-run analysis](./OVERNIGHT_ANALYSIS.md)**.

## Experiment Log

The honest headline metric is the **held-out win rate** on the immutable 463-word split
(`data/wordlists.split` — train/held disjoint, held never trained on), greedy, free-generation,
**no inference-time rules** (no dictionary, no consistency filter, no candidate list). Numbers that
are *not* honest-greedy-held-out (seen/train probes, beam+dict decoding, leaked CoT) are labeled as
such. The whole thread runs 2026-06-02 → 06-08 on the M5 Max (MPS). All experiment drivers live in
[`scripts/`](./scripts/) (uncommitted; one script == one experiment, docstring at top states the test).

### 🔤 Context-format bake-off (interleaved / keyboard layouts) — NULL; layout changes learning *speed*, not converged ability ([scripts/format_sweep.py](./scripts/format_sweep.py))

"Does a different board layout help?" Tested rigorously after a chain of methodology fixes. **Diagnostic
that motivated it:** the 0.338 model *violates its own clues 14% of the time* (places a wrong letter on a
green square, reuses a gray) — a real "tracking" target formatting could attack. Metric used: **clue-respect**
(of valid guesses, fraction consistent with the clues) — measurable even when win≈0 from-scratch.

- **Layouts:** baseline (`s t a r e 🟩⬜⬜⬜🟨`, letters then colors) · **interleaved** (`s🟩 t⬜ a⬜ r⬜ e🟨`,
  each letter beside its clue) · **keyboard** (baseline + an alphabet-state suffix).
- **Methodology (the hard part):** warm-start confounds (a format change needs from-scratch training, as the
  user noted — interleaved warm-started from baseline started at VAL 0.000); from-scratch under-trains
  deduction (win≈0), so compare on clue-respect; per-epoch VAL respect is *noisy* (±0.2) and the best-by-win /
  best-by-respect selections both have degenerate failures. Final read = **respect on the full 367-game TEST**.
- **Result (clean, full-TEST):** baseline **0.368** · interleaved **0.402** (+0.034 respect, but −0.033 valid →
  ≈ wash) · keyboard ~0.37. **win ≈ 0 for all.** Early-epoch "leads" (interleaved hit 0.61 mid-run) were
  **learning-speed artifacts** that converged back to baseline.
- **Verdict: formatting is not a lever.** The model can *read* the clues in any layout; it just can't *deduce*
  from them — the discrete-search wall, unchanged by representation. A real interleaved inference confirms it:
  valid CoT opener (`stare`), then non-words that ignore the clues on turn 2.
- **Operational find:** the real cause of GPU slowness all session was a **3-day `live_viz.py` dashboard**
  re-playing games on MPS every checkpoint change — killed it; GPU then ran ~3–4× faster. (Also: concurrent
  training on one MPS GPU backfires — contention makes it ~6× slower; sequential is optimal.)

### 🏗️ Different encodings & architectures to beat 0.34 — all NULL, but a unifying principle emerged ([full report](./DAYRUN_REPORT.md))

"Can a different encoding or model arch push past 0.34?" Tried two new architectures (after the dense-encode
port). Both null on honest TEST — but together with dense-encode they reveal **why** the wall holds:

- **Reasoning-CoT** (≈50M): condition on raw history, then *derive and write out* the constraint state as
  CoT, then guess (teacher supervises the derivation at train time; free ephemeral decode at inference).
  **NULL** — the reasoning **collapsed to a constant** (derive-acc pinned at exactly 0.679 = all-BLANK) and
  the model routed around it, guessing straight from history. Win plateaued ~0.04. *Why:* the constraint
  state is a **deterministic function of the board** — restating it adds no information, so CoT can't help
  (CoT only helps when it does search/computation hard to do in one pass).
- **Iterative Refine** (≈50M, lean 36-vocab): a learned *edit* operator — condition on history + a DRAFT
  word, output a better one; at play, feed each guess back K passes (full-word lookahead). **NULL, TEST
  0.098** — and the pass-ablation is the tell: **passes 0/1/3 are byte-identical (0.104 VAL)**. The
  refinement is **identity** — the model ignores the draft and commits the same word regardless of passes.

**The unifying principle (3 architectures, 1 failure mode):** any auxiliary channel the model can *route
around* — a reasoning prefix (reason-CoT), a draft to edit (iter-refine) — collapses to a no-op, because
the guess attends to the raw history directly and the channel carries no information the model can't already
compute. The *opposite* fix — **bottlenecking** the guess through an explicit constraint state (dense-encode)
— **memorizes** instead (train 0.32 / held 0.07). Route-around → no-op; bottleneck → memorize; **raw history
only → best generalization (0.338)**. That tension is *why* 0.34 is hard to beat honestly with a single
forward pass.

**Product-of-Experts — the escape that confirmed the wall (NULL 0.332, delta +0.000).** The one mechanism
that can't be routed around: fuse the frozen generator G (validity_max_v4, common words) with a *separate*
consistency expert C (trained answer-**agnostically** on 50k full-dict clue→consistent-word states) by
**multiplying** their letter-logits. Fusion worked mechanically (C *did* change the output at high β), so
this wasn't a route-around collapse — yet **C added nothing** (β=0/0.5/1 identical to G; β=2/3 *hurt*; TEST
0.332 = G-alone, +0.000). *Why:* C learned "what is *a* consistent word" but **can't pin THE answer among
many consistent candidates on held-out** — that requires either knowing the answer (memorization, banned) or
constraints tight enough to be unambiguous. The deduction-generalization wall, relocated into a second
network. **Four honest architectures (dense, reason-CoT, iter-refine, PoE) now converge on the same wall.**

**…but the ambiguity diagnostic reframes the wall (and reopens the attack).** Measuring *why* G loses on
held-out (`ambiguity_diag.py`, exact full-dict consistent-set count at each losing guess): **61% of losses
have ≤3 consistent words, and 30/150 games have a UNIQUELY determined answer (consistent set = 1) that G
still misses.** Only 16% of losses are genuinely ambiguous. So the wall isn't "too many candidates" — it's
**G failing on near-determined positions** (median consistent set = 3; implied ceiling +0.41). PoE's C was
just too *soft* (trained on loose random-guess states). The fix under test (`poe_sharp.py`): a **sharp C**
trained on a curriculum of realistic *tight* states (the endgame), measured by whether it can generate the
unique consistent word on held-out size-1 states.

**Sharp-C result — the root cause, proven (NULL 0.332, but it explains everything).** Even a C trained
*explicitly* on tight endgame states (unlimited answer-agnostic data) is null in PoE (TEST 0.332, delta
+0.000), and the decisive metric says why: **size-1 deduction accuracy = 0.07 held-out (1/14), 0.14 train
(2/14)** — when the clues *uniquely determine* the answer, the expert produces it only ~7% of the time on
unseen answers (and can't even *fit* it on train). The `constraint → unique answer` mapping is a **discrete
combinatorial search, not a smooth function a generative net learns to generalize.** That is the mechanistic
root of the ~0.34 honest ceiling: solving a near-determined Wordle position requires either explicit
search/verification (the banned solver) or having seen the answer (banned contamination). A common-word
frequency prior doesn't escape it either — ranking the consistent candidates needs the consistent set, which
needs the banned consistency filter at inference.

**Final honest conclusion (6 architectures + 2 diagnostics):** the honest free-gen ceiling is **0.338**
(validity-max). It is *not* a missing trick or a tuning gap — it is that Wordle deduction is discrete search,
which generative weights cannot learn to generalize honestly. The journey *is* the result.

### 🧪 Honest RL (free-gen GRPO) on the 50M base — **NULL 0.332 TEST** (RL sharpens train, doesn't generalize) ([full report](./DAYRUN_REPORT.md))

"Are there RL techniques to push the win rate higher?" Answered with the first **clean-lineage** RL run: a
correct free-gen GRPO on `validity_max_v4` — CoT-bearing rollout, win-dominance + non-word-penalty reward,
group-relative advantage (no ÷std), zero-variance filter, clipped surrogate, **k3 KL to a frozen ref**;
train-only secrets; eval = greedy CoT free-gen, zero rules, disjoint TEST.

- **First attempt degraded** — found a real bug: `old_logp` was computed in eval mode (dropout off) but the
  inner-update `new_logp` in **train mode (dropout on)**, so the surrogate ratio `exp(new−old)` was
  corrupted by dropout noise. Keeping the policy in eval mode through the update (gradients still flow) fixed it.
- **Fixed RL works — mechanically.** Over 50 iters, **train rollout win climbed 0.41 → 0.60** (+19 pts) and
  train non-words nearly halved (0.57 → 0.36). The CoT is load-bearing (no-CoT commit collapses win
  0.333→0.073 at equal spelling), so RL was shaping real reasoning, not just spelling.
- **…and transfers nothing.** Held-out **TEST = 0.332 (122/367) ≈ base 0.338** (the noisy peak VAL 0.354
  collapsed on the 4×-larger sample). Held-out validity also flat (0.855 vs 0.866). **Verdict: RL does not
  raise the honest win rate** — the gains are memorization/sharpening of the train answers; the wall is
  deduction-*generalization*, which no RL knob touches. (Consistent with every prior 5M RL null, now proven
  on a correct, clean-lineage 50M GRPO.)

### 🔬 Colleague's "90% without cheating" — reproduced HONESTLY: **0.065 TEST** (the encoding memorizes) ([full report](./DAYRUN_REPORT.md))

A colleague's repo (`rynowak/mm`) reportedly hit **90% without cheating**. Investigated deeply, then ported
their genuinely-good idea — the **dense constraint-state encoding** (replace the raw board history with a
clean digest: greens-by-position, yellows-with-excluded-position, grays-with-exact-counts) — **honestly**
(train-only secrets, free char-gen, letter-mask only, disjoint TEST). Result: **clean-protocol TEST win
0.065 (24/367)** — **~5× worse than our validity-max 0.338**.

- **The 90% was train-test contamination, not the encoding.** Their pipeline trains + evals on the **full**
  2,315-answer set with only a per-*example* split (no held-out *by answer*), so 90% = memorizing answers
  (same class as the rejected oreo 0.89). Reproduced with an honest answer-level split, the same
  architecture barely generalizes.
- **"Is it a bug?" — no.** The discriminator: **TRAIN-secret win 0.317 vs HELD-OUT 0.075.** A broken
  pipeline can't win 31.7% on trained secrets — so the encoding/mask/eval are correct; the gap *is* the
  result. The dense encoding learns `constraint→answer` for **seen** answers and ignores the clue facts on
  unseen ones (guesses `salvo` with `v,o` already gray; greedy then loops on a byte-identical state — the
  dashboard "stuck" symptom). Raw history can't loop that way (the board always grows), so the dense state
  is structurally *worse* honestly, not just under-trained. **Verdict: rejected (0.065 ≪ 0.338).**

### 🌙 Overnight #4 (2026-06-07→08): the honest protocol + the data lever — best honest model **0.332** ([full report](./DAYRUN_REPORT.md))

Two honesty rules were tightened (and they're the right ones): **(1) no dictionary at inference, ever**
(retires beam-over-trie / best-of-N / constrained-decode as "the model"); **(2) a non-word guess is
counted as a turn but NOT fed back** to the model (kills the all-gray "lottery" the old eval gave). Under
this **clean protocol** (pure free-gen), the true stage-1 number is **0.243** (the old 0.281 was inflated
~+0.04 by the lottery).

- **The validity lever works but plateaus.** In-weights validity (aux + constrained self-distillation) is
  monotonic with honest win: 0.243 (valid 0.66) → 0.281 → 0.302 (valid 0.76); pushing aux further (v3,
  validity 0.77) gave **no** gain — the per-position aux is exhausted.
- **The data lever broke the plateau.** Same recipe (aux 6 + constrained self-distill) on **all 1852 train
  secrets** → **clean TEST 0.332** (+0.030; clean-VAL 0.333 ≈ clean-TEST across 463 independent secrets;
  validity held ~0.76, so the gain is **fewer deduction losses from more in-distribution data**). This is
  the **best honest model** (`validity_max_v4.pt`), **verified across two seeds** (seed 0 = 0.332, seed 1 =
  0.335). The data lever is now maxed (held-out is off-limits).
- **What failed (honestly):** the green/yellow **infill/template** (clue logic taught in training, free-gen
  at inference) — both aux configs landed ~0.05; the model spells a valid *opener* but the explicit
  template *breaks* constrained turns (the "structured-context hurt" lesson again). On-policy RFT/STaR,
  scale (turns over past 50M), and aux-cranking were all confirmed at/under the ceiling too.

**Net:** honest free-gen held-out climbed **0.243 → 0.332** via in-weights validity + maxed in-distribution
data; the remaining wall is **deduction-generalization**, bounded by the ~1852-secret answer set.

### 📈 Day run (2026-06-07): scale sweep — held-out **turns over** past 50M ([full report](./DAYRUN_REPORT.md))

"Would a bigger model fix it?" — answered, and the answer is **no**. The fair recipe trained at four
sizes gives a **non-monotonic** honest held-out curve: **tiny 1.2M → 0.163 · base 12M → 0.251 ·
large 50M → 0.281 · xl 99M → 0.270.** Held-out win **peaks at ~50M and regresses at 99M**: the larger
net memorizes train secrets harder (TRAIN[:200] 0.55 vs TEST 0.27 — the widest generalization gap in
the sweep) *and* spells worse (valid 0.662 → 0.591). The deduction/vocabulary wall is **not** a
capacity problem; more parameters past 50M buy memorization, not generalization. The levers remain
data honesty + test-time compute, not weight count. (Visual lineage: `EXPERIMENTS.svg`.)

### 🧪 Autonomous push #3 (2026-06-07): on-policy distillation + the composition lever — the win ceiling holds ([full report](./DAYRUN_REPORT.md))

A from-scratch attack on the **free-gen greedy** number (0.281, the only no-aid metric), with a control
to keep us honest. The governing fact: **best-of-N reaches ~0.70 on the same held-out TEST greedy gets
0.281 on** — the capability *generalizes into the model's distribution*; greedy decoding won't surface it.

- **On-policy self-distillation (RFT/RAFT, `rft_distill.py`):** distill the model's *own* best-of-N
  winning games back into greedy (reward-weighted MLE). A gentle LR (3e-5) initially looked like a win
  (TEST 0.319), but a **STaR iteration did not compound** and a **teacher-only control busted it**: the
  same best-of-16-epochs-on-96-VAL selection hits VAL 0.365 *by chance* and scores TEST anywhere in
  **[0.26, 0.32] regardless of method** (the control landed 0.259, *below* stage-1). The "gains" were
  selection noise. ⇒ honest free-gen win is **~0.28–0.30 across every method**.
- **Composition lever (`validity_max.py`):** crank the aux-validity loss (λ3) + distill the model's own
  *constrained-decode* (always-valid) games. This **robustly raised in-weights validity 0.662 → 0.712**
  (the control stayed 0.656) — but it **bought no win**. That answers the composition question: **spelling
  is not the win bottleneck; deduction-generalization is** — and it's data-bound (only ~1,852
  in-distribution train secrets; held-out answers are off-limits).

**Tally: 9 training approaches now confirm the ~0.28–0.30 free-gen ceiling** (RL ×11, DPO ×3, DAgger ×2,
info-gain, distill, RFT/STaR, validity-max). The model *knows* more than greedy says; the only honest way
to surface it is **test-time compute + the public dictionary as a spell-checker** (beam-over-trie 0.55
deterministic; best-of-N 0.72) — the model generates freely, the dictionary only validates spelling, never
filters by clue-consistency. **That is the best honest model.** *(Method note: a 96-secret VAL is too
noisy for best-epoch selection of ~0.02 effects — the control is the cautionary tale; trust TEST, not VAL.)*

### ✅ Fair-honest run (2026-06-06): dictionary pools, not answer-set — honest TEST **0.281**

The clean, *fair* recipe: candidate/teacher pools = the full **valid-guess dictionary** (the model may
know a word *exists* + how to spell it — public knowledge), but **answer-hood stays train-only** and
the headline is reported on a disjoint TEST (`held[96:]`). This is the legitimate middle between the
over-hobbled overnight run (train-only pools → 0.166) and the answer-set-leak contaminated runs (~0.62).
Plus a spelling push (pretrain warm-up 10→30, aux-validity λ 0.5→1.0). `scripts/cot_ephemeral_aux_fair.py`
→ `cot_eph_aux_fair.pt`. Full write-up: [`VALIDITY_REPORT.md`](./VALIDITY_REPORT.md).

| set | win | valid | avg | note |
| --- | --- | --- | --- | --- |
| **TEST** `held[96:]` (HONEST) | **0.281** (103/367) | **0.662** | 4.33 | +69% over the over-hobbled 0.166 |
| VAL `held[:96]` (selected-on) | 0.344 | 0.680 | — | best-ckpt selection |
| TRAIN[:200] (memorization ref) | 0.515 | 0.740 | — | gen gap (TRAIN−TEST) ≈ 0.23 |

**Validity findings (the "only valid words" push):** validity plateaus **~0.66 on TEST** for pure
free-gen on 50M — cranking pretrain+aux did *not* push it dramatically higher, and **validity-targeted
DAgger v1 was a NULL result** (validity flat 0.66→0.63 across 4 rounds) because the corrective set was
only **399 / 14,867 = 2.6%** of the data — drowned out (`scripts/dagger_validity.py`). **DAgger v2**
oversampled corrective ×20 and was **also null** (validity flat ~0.66, and it *hurt* win) → the ~0.66
ceiling is **structural** (no-lookahead autoregressive drift), not a data-weighting issue. Note the
fair recipe's 0.66 validity is *lower* than the old
contaminated ~0.75 **by design** — dictionary candidate pools train on the hard full 14k vocab (rare,
hard-to-spell), which is also *why* win is higher; it's a validity-for-honest-play trade. Pure free-gen
to ≈1.0 ("only valid words") is not reachable on 50M — the last ~30% needs dictionary-constrained
decoding (spelling-only; weaker than the rejected candidate-ranking).

**🔑 Constrained-decoding diagnostic (`constrained_decode_eval.py`):** same stage-1 weights, committed
guess masked to dictionary-valid continuations (spelling-only — the model still does its own deduction):
**free-gen TEST 0.281/0.662 → constrained 0.436/1.000 (win +0.155, +55%).** Decisive: **the model KNOWS
the words + does the deduction; free-gen *spelling drift* was the bottleneck.** So 0.436 is a legitimate
"deduce + spell-checker" number, and chasing validity is high-value.

**Stage 3 — on-policy self-distillation (`distill_constrained.py`):** SFT the free-gen model on its own
*constrained* rollouts (valid guesses reflecting its own deduction) — no inference crutch. **This lifts
free-gen validity** where DAgger couldn't: VAL valid **0.62 → 0.80** by round 3, with win ~flat (0.302
vs 0.344, within MPS noise). The on-policy validity lever works. (Selection fixed to best-by win+valid
so the validity gain is actually banked — pure best-by-win was discarding it.)

**Stage 4 — long GRPO on the full reward (`rl_grpo_full.py`, queued):** the full shaped reward already
encodes both (−invalid penalty **and** +win bonus), so GRPO optimizes win **and** validity together
on-policy — and the *continuous* reward dodges the zero-variance-filter trap a pure-validity reward
hits. Run **from the distilled (~0.80-valid) base**, 150 updates, stabilized settings, best-by-win,
frequent held-out eval. The dashboard shows **all the exact sampled rollouts** GRPO trains on, **every
update** (real-time), each with its **grade** (reward `r=` + advantage `A=`) and a validity line.
GRPO previously underdelivered here, but those were *vocabulary-injection* failures — validity is an
*expression* problem (the model knows the words), which RL suits.

### 🌙 Overnight push #2 (2026-06-07): gains are at inference, not training ([full report](./OVERNIGHT_REPORT_2.md))

Six training methods tried against the free-gen deduction wall — **all null**: DAgger, distillation,
GRPO, info-gain expert-iteration (constrained *and* free rollouts), and DPO. **Pure free-gen held-out
stays 0.281** — training cannot inject the missing produce-and-deduce-unseen-words capability. The
**honest decode ladder** (same stage-1 weights) is where the gains live:

| decode | TEST win | valid | aid |
| --- | --- | --- | --- |
| free-gen greedy (**pure headline**) | **0.281** | 0.662 | none |
| best-of-16 vote, no dict | 0.243 | 0.635 | compute only — *worse* |
| constrained-decode greedy | 0.436 | 1.000 | spelling |
| best-of-16 vote, valid-filter | 0.632 | 0.925 | spelling + compute |
| best-of-64 vote, valid-filter | 0.703 | 0.963 | spelling + more compute |
| **best-of-128 vote, valid-filter** (**best honest**) | **0.719** | 0.979 | spelling + most compute |

The **dictionary/spelling aid at inference is the lever** (0.281→0.436); sample-and-vote adds on top and
**scales with compute to a ~0.72 plateau (0.436→0.632→0.703→0.719 at N=1/16/64/128)** — but only because
it has valid candidates to vote among (compute over raw free-gen samples *hurts*, 0.243). **The honest
0.72 beats the old contaminated 0.62 — cleanly.** Honesty win: DPO's earlier 0.616→0.631 was
**contamination-dependent** (null on the clean base). Two numbers to quote: **0.281 (model alone)** and
**~0.72 (model + honest aided, compute-scaled decoding).**

**How best-of-N decoding works** (to read the ladder above): at each turn, **sample N independent
free-gen guesses** (temperature 1.0 — not the one greedy answer), **discard the ones that aren't real
words** (the `valid-filter`: the public dictionary, *not* clue-consistency), and **play the most common
surviving word** (majority vote / self-consistency). In plain English: *ask the model N times, keep the
real-word answers, go with the one it gave most often.* It costs **~N× the inference compute** and uses
the dictionary at decode time, so it's **aided** (not the pure-model 0.281) — but it never uses the clues
to pick candidates, so the model still does all the deduction. `scripts/best_of_n_eval.py` (`BON_N`,
`BON_FILTER=valid|none`). Why it works: a single greedy guess often commits to a wrong/non-word, but the
model's *distribution* usually contains the answer — sampling + valid-filtering + voting surfaces it.

### ⚠️ Adversarial audit (2026-06-05): held-out contamination — methodology violation, win-impact refuted

A 4-agent adversarial investigation (Lead, Code-Analyst, Telemetry-Analyst, Devil's-Advocate) asked
**"is the model built correctly?"** Two-part verdict, stated precisely:

1. **The library is CORRECT** — confirmed by 231 passing tests + per-claim tool receipts (Code-Analyst
   + Devil's-Advocate). Causal attention (perturbation-tested), weight-tied head, pos-embed bounds;
   `generate` emits exactly 5 letters length-masked to the 26 letter-ids and samples on CPU; SFT loss
   gradient is exactly on the guess-letter positions (zero on board/feedback); aux-validity trie aligned
   + gated to supervised positions; reward no-double-count + dominance; GRPO (Dr.-advantage, k3 KL,
   zero-variance filter, eval-mode forward) and DPO math; two-pass duplicate scoring; disjoint immutable
   split. No correctness defect found.
2. **The headline TRAINING/eval pipeline contaminates the held-out discipline (CONFIRMED methodology
   violation)** — held-out words become **loss-True training targets** via 4 channels. **But the
   measured win-impact is REFUTED (inert): the numbers are most likely real — do NOT retract them.**

**The 4 leak channels (held-out words as loss-True training targets — confirmed):**

| # | Channel | Mechanism | Blast radius | Receipts |
| --- | --- | --- | --- | --- |
| **FM-2** | CoT candidates | `pick_cands`→`consistent_candidates(history, ANSWERS)` over the **full** answer pool (incl. all 463 held-out) → held-out words are loss-True `<think>` targets | whole CoT family (cot_eph_aux **0.616**, cot_eph 0.430, cot_50m, dagger, expert-iter) → **both headline numbers** | `scripts/cot_ephemeral_aux.py:44,66,79-83` |
| **FM-3** | Teacher guesses | `generate_transcripts` used the full valid/answer pools → InfoMax/Consistent teacher **plays held-out words as guesses** on train games (loss-True commit targets) | **every SFT**, incl. the pre-CoT char-50M+aux **0.436** | `src/wordle_slm/teacher/transcripts.py:50,59` |
| **Opener** | Opener leak | `trace` ∈ `DEFAULT_OPENERS` **and** is a held-out answer → teacher opens with it → free win on held-out secret `trace` | any run using default openers | `teacher/transcripts.py` openers |
| **FM-1** | Selection | best-checkpoint selected on `held[:96]`, headline reported on full `held` (a **superset**) | universal (SFT/RL/DPO/CoT) | all drivers |

**Win-inflation = NOT supported (the Devil's-Advocate ran the measurements):**

| Probe | Measurement | Reading |
| --- | --- | --- |
| FM-1 (selection bias) | SAME eval, SAME checkpoint, two runs **disagreed**: Code-Analyst held[:96]=**0.635** vs held[96:]=**0.600** (+3.6pt) — DA held[:96]=**0.604** vs held[96:]=**0.619** (**−1.4pt**) | the divergence is **MPS greedy non-determinism ~±2-3pt on ~100-game slices** → FM-1's effect is **within noise**, not a reliable inflation |
| FM-2/FM-3 (vocab lean) | at inference the trained **0.616** model emits held-out words as candidates **9.4%** / commits **9.2%** — **below** the 20% answer base-rate | the leak is **INERT** (the model does not lean on held-out vocab); inference is board-only free-gen and **never calls the consistency filter / ANSWERS** |

**Conclusion: 0.616 / 0.631 are most likely REAL numbers produced by a contaminated-but-inert training
pipeline — NOT inflated.** The remediation is to clean the channels and re-run to confirm a ≤small delta,
**not** to retract.

**Legitimate carve-outs (NOT leaks, DA-confirmed):** the pretrain **spell warm-up** and the **aux-validity
trie** over the full **valid-guess** list teach **spellability only** (context-free) — held-out words are
valid *guesses*, excluded only as training *secrets*; a principled distinction. `rl/curriculum.py:36-44`
**already** excludes held-out (the correct pattern the candidate/teacher code bypassed).

**Fixes (committed `a41b1b2`) + clean re-run (pending):** `generate_transcripts` gains optional
`valid_pool`/`answer_pool` (pass **train** → the teacher never plays held-out) plus a leak test;
`scripts/cot_ephemeral_aux_clean.py` re-runs the **0.616** recipe with train-only candidate + teacher
pools, held-out-free openers, and **disjoint VAL(`held[:96]`)/TEST(`held[96:]`)** selection-vs-report.
The **clean re-run is in progress** (`runs/cleanrun.log`) to confirm the honest number + true leak
magnitude (**expected ≈ 0.60–0.62**).

**New methodology caveat (important going forward):** **MPS greedy eval is non-deterministic ±2-3pt on
~100-game slices.** Many reported deltas — 0.616 vs 0.631, the +1.5 DPO gain, the +2.8 CoT gain — sit
near or within this noise band. **Prefer full-463 and/or multi-seed evals for any claimed delta.**

### 🔬 Overnight clean re-run (2026-06-06): honest held-out ≈ 0.17 — the audit's "inert" call is OVERTURNED

The clean re-run (the remediation the audit set up) is in and it **reverses** the audit's win-impact
verdict. The 2026-06-05 DA had concluded the leaks were **inert** ("numbers most likely real, ~0.60–0.62");
the leak-free pipeline — **train-only candidate + teacher pools, held-out-free openers, disjoint
VAL(`held[:96]`)/TEST(`held[96:]`)** — shows the honest held-out **collapses from 0.616 to ~0.17**. The
whole **0.40 → 0.62 progression was largely held-out vocabulary leakage** (teacher guesses + CoT `<think>`
candidates were drawn from the full answer pool, so the model was taught the held-out answer words as
playable, clue-consistent guesses). Commit `d00f89d`.

**This is verified, not a bug.** The clean SFT **converged** (loss 1.93 → 0.43) and shows a textbook
generalization gap — it learned, it just cannot produce words it was never trained to output as answers:

| check | clean SFT | reading |
| --- | --- | --- |
| win on **TRAIN[:120]** | **0.858** | learned the task |
| win on **TEST (`held[96:]`)[:120]** | **0.175** | does not generalize |
| final training loss | **0.43** (from 1.93) | converged, not under-trained |
| held-game guesses that are **held-out words** | **28 / 690 = 4%** | almost never emits an unseen secret |
| held-game guesses that are **train answers** | 457 / 690 = 66% | plays the vocabulary it was trained on |
| sample held secret `befit` | `slate meter debit hefit begut penco` → LOSE | never emits `befit` |

The 0.858/0.175 split is the proof: trained only on **train-answer outputs**, the model plays train answers
(66%) and almost never produces a held-out word (4%), so it cannot commit a held-out secret → ~0.17 win.

**Clean leaderboard (all stages leak-free; TEST = `held[96:]`, honest):**

| setup | VAL (`held[:96]`) | **TEST (honest)** | full | valid | avg |
| --- | --- | --- | --- | --- | --- |
| SFT (ephemeral-CoT + aux) | 0.188 | **0.166** | 0.171 | 0.749 | 4.66 |
| DPO (on SFT) | 0.188 | 0.166 | 0.171 | 0.749 | 4.66 |
| GRPO (on best, clean reward) | 0.188 | 0.166 | 0.171 | 0.749 | 4.66 |
| DAgger (on SFT) | 0.146 | 0.144 | 0.145 | 0.813 | 4.64 |
| DPO ∘ DAgger | 0.146 | 0.144 | 0.145 | 0.813 | 4.64 |
| DAgger ×2 | 0.125 | 0.169 | 0.160 | 0.794 | 4.89 |

Everything sits in a **0.14–0.17 band**: the plain clean **SFT (TEST 0.166)** wins; **DPO and GRPO revert to
the SFT base (no gain)** and **DAgger slightly hurts**. The 0.166 is stable — confirmed by a **3-seed TEST
mean 0.1662** (not MPS noise). **No downstream lever (DPO/GRPO/DAgger) moves the clean ceiling.**

**Why the audit was wrong about "inert":** the DA read "the 0.616 model emits held-out words 9.2%, below the
20% base-rate → leak is inert." That was a misread — 9.2% of *all* guesses being held-out words is the very
mechanism that lets it win held-out secrets, and "20%" was the wrong baseline. The contaminated pipeline fed
held-out answer words in as **loss-True targets** (teacher guesses over the full pool + `consistent_candidates(history, ANSWERS)`),
which **taught the held-out answer vocabulary as playable words** — exactly the knowledge needed to commit a
held-out secret. The clean re-run is ground truth. (FM-1 selection bias *was* within noise, as the DA found;
but the **FM-2/FM-3 vocabulary leak was the dominant ~45-point effect**, not inert.)

**The fundamental insight — the vocabulary wall.** Wordle "deduction" *requires* the candidate-word
vocabulary: clue-narrowing only works if you can enumerate words that fit the pattern, i.e. **knowing the
words** (from `_oist` you cannot produce `joist` unless you know `joist` is a word). So "generalize to
held-out" = **play words never seen as targets ≈ impossible**; the honest held-out ceiling is essentially
capped by **spelling knowledge alone (~0.17 here)**. The only way to lift it is to give the model the answer
vocabulary — which, under the strict held-out rule, **is** the leak.

**Two legitimate framings (decide which game you're measuring):**

1. **Strict held-out generalization** ("deduce, don't look up"): honest = **~0.17** — the real ceiling.
   RL/DAgger/DPO/GRPO are **exhausted** (they can't inject unseen vocabulary). The next move is a different
   framing, not more training.
2. **Deployed real Wordle** (the answer set is **fixed and known** — 2,315 words, no novel secrets):
   training on all answers is **not cheating for the actual game**; that model plays the real game at
   **~0.62**, a legitimate *deployed-player* score. The held-out split is an artificial generalization probe
   we imposed.

Both are real; they measure different things — **0.17 = "can it deduce novel words" (no); 0.62 = "can it play
the real, fixed-vocabulary game" (yes, decently)**.

### Results leaderboard

Honest held-out only (greedy, free-gen, no rules), best-first. Yardsticks and the inference-aided
high-water mark are listed separately at the bottom — they are **not** comparable honest-greedy numbers.

> 🔬 **STRICT-HELD-OUT CORRECTION (2026-06-06):** under a fully leak-free pipeline the honest
> strict-held-out numbers are **~0.17**, not 0.40–0.62. The clean re-run (commit `d00f89d`) shows the
> entire 0.40 → 0.62 column reflects **held-out-vocabulary leakage** (held-out answer words fed in as
> loss-True teacher/CoT targets). **Read the column below under one of two framings:** (a) these are
> **deployed-real-Wordle / contaminated-methodology** numbers — legitimate only if the fixed, known
> 2,315-answer set is accepted as fair game (no novel secrets); under that framing the best *player*
> scores **~0.62**. (b) Under **strict held-out generalization** (deduce unseen words) the honest ceiling
> is **~0.17** (plain clean SFT; RL/DPO/DAgger/GRPO all 0.14–0.17, exhausted against the vocabulary wall).
> The rows are kept for the record but are **not** strict-honest numbers. See the overnight subsection above.

> ⚠️ **Prior audit caveat (2026-06-05, now superseded on win-impact):** these held-out numbers were
> produced with a now-fixed **training-target** leak (FM-2 CoT candidates / FM-3 teacher guesses over the
> full answer pool) plus selection-on-`held[:96]` (FM-1). The audit's DA called the leak **inert** — the
> **2026-06-06 clean re-run OVERTURNED that** (honest collapses to ~0.17; the leak was the dominant lever,
> not inert). See the overturn subsection above.

| Approach | Size | Held-out win | Valid-rate | Avg guesses | Notes |
| --- | --- | --- | --- | --- | --- |
| **DPO commit-sharpening** 🏆 | 50M | **0.631** | 0.779 | 3.89 | **Honest best.** `dpo.pt` (`dpo_commit.py`); DPO on self-play win/loss commit pairs atop the model below; first preference method to move held-out (GRPO was flat) |
| ephemeral-CoT + aux trie-validity | 50M | 0.616 | 0.788 | 3.87 | `cot_eph_aux.pt` (`cot_ephemeral_aux.py`); the two honest SFT levers stack **super-additively**; the DPO base |
| char + aux trie-validity loss | 50M | 0.436 | 0.675 | 3.62 | `sft_aux.pt` (`train_auxvalid.py`); the spelling lever alone |
| ephemeral-CoT (honest scratchpad) | 50M | 0.430 | 0.671 | 3.69 | `cot_eph.pt` (`cot_ephemeral.py`); plain-CE, the search lever alone |
| char SFT, deep + converged | 50M | 0.402 | 0.664 | 3.58 | `sft_deep.pt` (`train_deep.py`); the matched no-CoT/no-aux plain-CE baseline |
| char SFT, strong InfoMax teacher | 25M | 0.391 | 0.66 | 3.52 | `sft_xl.pt` (`train_path_a.py`); big memorization gap but best generalizer at the time |
| char SFT | 4.8M | 0.300 | 0.61 | 3.66 | `sft_big.pt` (`train_sft_big.py`) |
| BPE+TinyStories recipe (honest split) | 11M | 0.257 | 0.96 | 3.24 | `oreo_recipe.py`; great spelling, memorizes seen, weak on novel |
| char SFT | 3.2M | 0.205 | 0.54 | 3.66 | `train_run.py` (`sft_strong.pt`) |
| char SFT, diverse secret set | 25M | 0.220 | 0.525 | 3.48 | `train_path_a_div.py`; diversity killed the gap but hurt win |
| BPE-on-flat-wordlist | 12M | 0.212 | 0.849 | 3.27 | `bpe_wordle.py`; valid words, bad strategy |
| char + diverse curriculum (co-design) | 50M | 0.188 | 0.526 | 3.63 | `train_codesign.py`; rare-word dilution hurt answers-only eval |
| BPE+TinyStories recipe (honest split) | 50M | 0.190 | 0.956 | 3.10 | `oreo_recipe.py` @50M; overfits early, generalizes worse than char |
| BPE-on-flat-wordlist | 50M | 0.188 | 0.806 | 3.05 | `bpe_wordle.py` @50M; win flat vs 12M → flat-wordlist is the limiter, not size |
| CoT-50M (honest self-context) | 50M | ~0.192 | 0.662 | 3.39 | corrected number; the 0.456 was a leak (see retraction below) |
| RL / GRPO (8 formulations) | 4.8M–50M | ≤ base (0.436) | — | — | all flat or degrading on held-out; dead end |
| — *yardsticks & inference-aided (NOT honest-greedy)* — | | | | | |
| InfoMax teacher (live consistency filter) | — | 0.99 | 1.00 | 3.55 | strategy ceiling; not a learned free-gen model |
| Consistent teacher (plays a real word each turn) | — | 0.967 | 1.00 | 4.46 | weaker strategy yardstick |
| oreo-ai recipe, **SEEN/train** secrets | 11M | 0.87 (≈ oreo 0.89) | 0.96 | 3.24 | **train/test contamination** — evaluated on trained secrets |
| beam+dict decoding (sft_xl) | 25M | 0.580 | 1.00 | 3.70 | inference-aided high-water mark (dictionary trie at decode) |
| beam+dict+norepeat decoding (sft_xl) | 25M | 0.596 | 1.00 | 3.77 | same, never re-emit a prior guess |
| Random baseline / Consistent floor | — | floor | — | — | floor/yardstick references |

### Chronological log

#### 2026-06-02 — scale ladder, teacher choice, first RL

| Time | Experiment (script) | What it tested | Config | Result (held-out) | Takeaway |
| --- | --- | --- | --- | --- | --- |
| 16:30 | 3.2M SFT (`train_run.py` → `sft_strong.pt`) | baseline free-gen SFT | 3.2M char, pretrain + InfoMax teacher | win **0.205**, valid 0.54, avg 3.66 | floor of the scale ladder |
| 16:48 | 4.8M SFT (`train_sft_big.py` → `sft_big.pt`) | scale up + deeper pretrain | 4.8M char, cosine decay | win **0.300**, valid 0.61, avg 3.77 | scale lifts win; valid-rate still the bottleneck |
| 17:27 | GRPO on 4.8M base (`train_grpo_run.py`) | does RL move greedy play | lr 8e-5, loose KL, full train set | win ~0.29 (no gain over base), probe gap +0.38 | RL moves but memorizes train; no held-out gain |
| 17:44 | GRPO, diverse 14k secrets (`train_grpo_run.py`) | RL on a non-memorizable secret pool | replay off, 14,392 RL secrets | win ~0.27, **reward negative** | rare-word secrets too hard; gap halved but reward went negative |
| 20:15 | 25M SFT, strong teacher (`train_path_a.py` → `sft_xl.pt`) | how far pure imitation climbs | 25M, 5-pass InfoMax-on-answers | win **0.391**, valid 0.66, avg 3.52 | **best of the day**; large memorization gap (probe 0.77 vs held 0.39) but best generalizer |

#### 2026-06-03 — diversity, self-distill, decoding probes, the 8-formulation RL sweep, model+curriculum redesign, aux-validity (the best)

| Time | Experiment (script) | What it tested | Config | Result (held-out) | Takeaway |
| --- | --- | --- | --- | --- | --- |
| 08:20 | 25M SFT diverse (`train_path_a_div.py`) | break memorization w/ full valid-word secrets | InfoMax-answers + Consistent-on-valid | win **0.220**, valid 0.525, gap collapsed to +0.05 | diversity killed the gap but **hurt win**; weak Consistent teacher dominated corpus |
| 08:31 | beam+dict decoding (`beam_eval.py`, sft_xl) | how much win greedy leaves behind | beam width 12, ±dict trie | greedy 0.392 / beam 0.392 / **beam+dict 0.580** | dictionary at decode banks +19pts spelling → 58% (inference-aided ceiling) |
| 09:21 | self-distillation (`self_distill.py`) | bank the beam+dict spelling gain into greedy | SFT on model's own beam+dict games | greedy **0.384** (was 0.391), valid 0.719 | spelling improved but win didn't transfer; gibberish→valid not enough |
| 09:39 | no-repeat decoding (`norepeat_eval.py`, sft_xl) | forbid duplicate guesses | beam w10, ±dict ±norepeat | beam+dict+norepeat **0.596** | small decode gain; new inference-aided high-water mark |
| 09:56 | RL validity+consistency reward (`rl_consistency.py`) | RL #3: reward legality, not just win | diverse secrets, +0.1 consistent / −1 invalid | win **0.384** (base 0.422), reward negative | no gain over base |
| 10:01 | turn-budget probe (`turnbudget.log`) | do more guesses recover wins | sft_xl, max 6/8/10 | win 0.392 flat (>6 adds gibberish) | extra turns don't help; >10 overflows context_len 128 |
| 10:19 | per-guess GRPO (`rl_perguess.py`) | RL #4: clean per-guess credit | single-guess episodes, mean-centered | win **0.384** (base 0.422) | cleanest credit, still no gain |
| 10:46 | dict-in-the-loop RL (`rl_dict.py`) | RL #5: trie surfaces words in training | trie-sampled candidates, free-gen eval | win **0.384**, solved/board 0.04 | trie favors common words, rarely surfaces the answer |
| 10:59 | consistency-constrained RL (`rl_constrained.py`) | RL #6 (decisive): sample from still-consistent set | answer surfaced ~22% boards | win **0.389** (base 0.422), consistency 0.82→0.83 | **answer surfaced and reinforced, win still flat** → barrier is generalization/capacity, not signal |
| 11:49 | 99M scale test (`train_scale.py`) | does scale lift the ceiling | 99M, old answer-only data | plateaued (under-converged) | scale alone on old data doesn't help |
| 12:38 | co-design 50M + diverse (`train_codesign.py`) | redesigned model + diverse curriculum | `large` 50M, InfoMax + Consistent-rare | win **0.188**, valid 0.526, gap +0.09 | diversity-closing-the-gap was illusory; rare-word dilution hurt answers-only eval |
| 13:50 | deep 50M, converged (`train_deep.py` → `sft_deep.pt`) | isolate the **model** redesign | `large` 512×16 ~50M, dropout 0.15, 5-pass InfoMax | win **0.402**, valid 0.664, consistency 0.88 | depth+dropout+convergence = small real win (0.391→0.402) |
| 15:36 | **aux trie-validity loss (`train_auxvalid.py` → `sft_aux.pt`)** ⭐ | bake the dictionary into the weights | train_deep + λ·(−log P next-letter ∈ trie); **no trie at inference** | win **0.436** (202/463), valid 0.675, consistency 0.901, avg 3.62 | **HONEST BEST.** Clean +3.4pts over train_deep; lifted win more than valid-rate |
| 15:54 | RL polish on best base (`rl_polish.py`) | RL #7: squeeze points on sft_aux | GRPO, validity+consistency reward | win **0.436** (unchanged), reward negative | RL can't improve even the strongest base |
| 17:35 | info-gain RL (`rl_infogain.py`) | RL #8: add the missing info-gain term + 12 train guesses | +β·log(\|C_before\|/\|C_after\|) on sft_aux | win **0.436** (unchanged) | the missing reward term was info-gain; adding it still no gain → RL closed conclusively |

> **RL verdict (8 formulations):** trajectory lr1e-5/8e-5 · diverse-secrets · validity+consistency · per-guess clean-credit · more-rounds · dict-in-loop · consistency-constrained · info-gain. **All flat or degrading on held-out win.** The decisive consistency-constrained run surfaced and reinforced the answer yet held-out stayed ~0.39 → the wall is **generalization/capacity, not signal/sampling/reward.**

#### 2026-06-04 — BPE/tokenization, the oreo contamination teardown, context management, pass@N, and the CoT thread (+ retraction)

| Time | Experiment (script) | What it tested | Config | Result (held-out) | Takeaway |
| --- | --- | --- | --- | --- | --- |
| 09:34 | BPE-on-wordlist 12M (`bpe_wordle.py`) | does subword tokenization fix validity | from-scratch BPE on valid list (vocab 433), 12M | win **0.212**, **valid 0.66→0.849** | BPE is THE validity lever, but 12M too small → weak strategy (cycling/repeats) |
| 10:35 | BPE-on-wordlist 50M (`bpe_wordle.py`) | is it a capacity problem | same recipe @50M | win **0.188**, valid 0.806 | win flat at 12M & 50M → **flat wordlist** (no frequency signal) is the limiter, not size |
| 12:19–14:02 | oreo recipe iterations (`oreo_recipe.py`, recipe.log…recipe5.log) | replicate oreo-ai (TinyStories BPE pretrain + SFT) | byte-BPE on TinyStories → pretrain → SFT, 11M | win climbs 0.000→0.132→**0.261/0.257**; **SEEN/train 0.870** | reproduced oreo's ~0.87-0.89 **on seen secrets** = train/test contamination (no held-out split). Honest held-out only 0.257 |
| 15:33 | oreo recipe @50M, honest split (`oreo_recipe.py`) | the recipe at scale, strict held-out | TinyStories BPE pretrain + SFT, ~50M | held-out **0.190**, seen 0.67, valid 0.956 | overfits early (peak ~epoch 9), generalizes WORSE than char. Final honest ranking: char-50M+aux 0.436 > recipe-11M 0.257 > recipe-50M 0.190 |
| 16:26 | structured-context A/B (`structured_context.py`) | does explicit derived-state help | raw board vs board+greens/present/absent block, 14M aux-SFT, held 200 | raw **0.260** vs +state **0.170** (Δ −0.090) | explicit state **hurts** (redundant + longer seq); context mgmt is a non-lever |
| 16:31 | **pass@N probe (`passk.log`, on sft_aux)** | is the wall capacity or decoding | 150 held-out, sample N games | greedy 0.453 · pass@1 0.353 · pass@5 0.720 · **pass@10 0.787** | **MAJOR CORRECTION: it's a decoding/search gap, not a capacity wall** — knowledge generalizes, greedy just doesn't find the line |
| 17:23 | CoT A/B 14M (`cot.py`) | does reasoning surface the latent pass@10 | no-CoT vs `<think>`-cands-then-guess, held 200 | no-CoT 0.155 vs **CoT 0.415** (Δ +0.26) | looked like the path to a new best (later RETRACTED — leak) |
| 19:33 | CoT-50M (`cot_50m.py` → `cot_50m.pt`) | scale the winning CoT | 50M, teacher reasoning traces | reported win **0.456** | beat 0.436 — **but this number was inflated by an inference-time leak** |
| 20:37 | CoT-50M + aux (`cot_50m_aux.py`) | stack the two honest levers | CoT + aux trie loss, 50M | incomplete (killed at epoch 24, subsample 0.406; no full-463 milestone) | superseded by the leak finding before completion |
| 20:37 | **CoT integrity teardown (`cot_show.py` → cotshow.log)** | are the CoT numbers honest | A/B on the SAME 0.456 model, held 120 | teacher-context (past `<think>` via consistency filter) **0.450** vs honest self-context **0.192** (Δ −0.258) | ⚠️ **RETRACTION:** CoT numbers were leaked — past `<think>` blocks were rebuilt at inference using the banned consistency filter. Honest CoT-50M ≈ **0.192**, far below 0.436 |
| 20:46–22:00 | **ephemeral-CoT (`cot_ephemeral.py` → cot_eph.pt)** | honest fix: throwaway scratchpad (history is board-only, regenerate think each turn, discard) | 50M, plain-CE, train==infer distribution, no filter at inference | **held-out 0.430** (199/463), valid 0.671, avg 3.69 (best ckpt e29; still climbing — subsample 0.469 at e29) | ✅ honest CoT **works**: +2.8pts over the matched no-CoT plain baseline (0.402), ≈ ties the 0.436 best, 2.2× the leaked-model honest 0.192 |
| **2026-06-05 22:27–00:40** | **ephemeral-CoT + aux 🏆 (`cot_ephemeral_aux.py` → cot_eph_aux.pt)** | stack the two honest levers, run long | 50M, CoT (ephemeral) + aux λ=0.5 **gated to current-turn**, 50 ep, cosine 4e-4→4e-5, 5 teacher passes | **held-out 0.616** (285/463), valid 0.788, avg 3.87 (best ckpt e45; curve 0.604) | 🏆 **NEW HONEST BEST.** Super-additive: 0.402 → +aux 0.436 → +CoT 0.430 → **+both 0.616** (+0.214). Honest-greedy now **beats** the inference-aided beam+dict mark (0.58–0.60). Wins sleek/surer/weedy; loses only the hard tail (joist `_oist`, salsa) |

> **CoT status (resolved → breakthrough):** Done honestly (ephemeral scratchpad, no filter at inference, train==infer), CoT works: 0.402 → 0.430 alone, and **stacked with aux-validity it reaches 0.616 honest held-out** — the two levers are super-additive (CoT enumerates candidates, aux makes the enumeration valid). The earlier 0.415/0.456 were a leak (past `<think>` rebuilt via the consistency filter; honest ≈ 0.192). The model genuinely reasons (traces: 💭candidate → 🎯GUESS) and **pass@10 = 0.787** holds (leak-free), so the decoding/search gap was real and reasoning closed it. The honest-greedy 0.616 now exceeds the inference-aided beam+dict mark (0.58–0.60). Remaining losses are the hard tail (joist `_oist`; salsa double-s/a).

#### 2026-06-05 — Wordle-rules audit, then RL on the 0.616 base (10-row), self-consistency, and DPO

Rules audit first: verified the engine against the official rules (`scoring.py` two-pass duplicate
handling exact — `lever`/`eaten`, `geese`/`these`; 5 letters; 6 guesses; win = all-green). One
deliberate deviation — the engine **accepts** a non-word and burns the turn (the app rejects it), which
makes our benchmark **stricter** than real Wordle. Hard mode (reuse hints) not enforced (standard mode).

| Time | Experiment (script) | What it tested | Result | Takeaway |
| --- | --- | --- | --- | --- |
| 08:00–09:08 | **expert-iteration / ReST (`rl_expert_10row.py` → `rl_expert.pt`)** | distill self-play wins, 10-row | held10 0.604→0.635→**0.646**→revert; full-463 ≈ 6-row 0.62 / **10-row 0.637** | **the RL that works** — taught the model to *use* rows 7–10 (base wasted them: held6==held10==0.604). +4pts on 10-row, ~flat on 6-row. Naive STaR on the model's own noisy think first **degraded** it (0.604→0.531) → fixed by clean teacher-think rebuild (RAFT) |
| 09:42–14:01 | **higher-ceiling / reachability (`rl_expert_tail.py`)** | full-coverage + tail-focus high-K | **solved 1843/1852 = 99.5%**; full-coverage SFT reverted (no gain) | **coverage is NOT the bottleneck** — sampling wins ~every train secret; the gap is commit/generalization, not reach. More expert-iter = dead end |
| 14:07–14:56 | **GRPO polish (`rl_grpo_polish.py`)** | token-level GRPO on the CoT policy, 10-row | full-463: 6-row 0.622 / 10-row 0.637 (flat) | **9th GRPO confirmation: flat.** First attempt blew up (KL 12→294, degraded) from a dropout-in-forward bug; fixed (eval-mode forward + KL 0.05 + lr 5e-6) → stable but barely moves |
| 14:56–15:08 | **self-consistency probe (`self_consistency.py`)** | vote vs pass@N (held-out 150, 6-row) | greedy 0.607 · **vote@12 0.627 (+0.02)** · **pass@12 0.953** | PIVOTAL: pass@12=0.95 (huge latent ceiling) but **voting barely helps** — the winning line is a *minority*; the lever is **selection**, not voting |
| 15:11–15:32 | **DPO, noisy pairs (`dpo_commit.py` → `dpo.pt`)** | DPO on first-divergence win/loss pairs | full-463 6-row **0.631** (+1.5 over 0.616); pref_acc 0.60→0.73 | **first preference method to move held-out.** Capped by noisy credit (outcome ≠ caused by the first divergent guess) |
| 15:34–17:20 | **DPO, decisive-board (`dpo_decisive.py`)** | clean pairs: commit-the-secret vs commit-a-wrong-consistent-word at the same reachable board (6519 pairs) | **flat — reverted to 0.616** (all 5 epochs regressed) | clean credit didn't help: DPO logp over the whole `<think>`+guess let the **long think dilute/hijack** the 5-token commit (loss fell but held6 dropped). Fix = score **guess-tokens only** |
| 17:23–18:00 | **DPO, guess-tokens only (`dpo_guessonly.py`)** | score DPO on the 5 committed letters only | **knife-edge — no gain**: lr 5e-6 too weak (loss flat, never learned), lr 3e-5 collapses (held6 0.000, valid 0.000) | the 5-token gradient is either too small to move or destroys the letter distribution; the think tokens in full-response DPO actually **regularize** it — which is why full-response (0.631) worked. **Preference-method ceiling reached: 0.631** |

> **RL/technique verdict (2026-06-05):** The bottleneck is the **commit gap**, not knowledge — reachability is 99.5% and pass@12 = 0.95, yet greedy commits wrong. Methods that *reweight outcomes* barely move it: GRPO flat (9×), self-consistency voting +2pts. **Full-response DPO is the one that helped (0.616 → 0.631)** — but the commit gap is fragile: the decisive-board variant flattened (think-token dilution) and guess-tokens-only was a knife-edge (too-weak → no learning, or too-strong → collapse). The think tokens in full-response DPO turned out to **regularize** the update, which is why it alone worked. **Preference-method ceiling here ≈ 0.631.** Remaining honest levers: richer reasoning targets (generalize the commit) or scale.

#### 2026-06-05 (evening) — reward update, constraint-aux, GRPO-with-reward, DAgger

Shipped the **reward-model update** (committed: `repeat_penalty=0.4` + `drop_present_penalty=0.3` in
`rl/reward.py` + `RewardConfig`, 2 new exact-value tests, suite green — see Shared machinery), then ran
the failure-mode fixes the new reward implies: train-in the constraint (no-op), feed the fixed reward to
GRPO (dead), isolate the commit (KL-explodes), and put the failure states into the data (DAgger).

| Time | Experiment (script) | What it tested | Result | Takeaway |
| --- | --- | --- | --- | --- |
| (commit) | **reward-model update (`rl/reward.py`, `RewardConfig`)** | shape the two slip-through failures | `repeat_penalty=0.4` (re-emit a prior valid guess) + `drop_present_penalty=0.3` (omit a known-yellow letter); 2 new exact-value tests, full suite green | the "update the reward model" deliverable — greens were already penalized by the clue term; these add the SALAD-twice + yellow-drop cases |
| 22:40 | **constraint-aux (`cot_constraint.py`)** | train-in green-keep + yellow-reuse + anti-repeat aux on `cot_eph_aux.pt` | **NO-OP — reverted.** base win 0.604; aux terms `g=0 y=0 r=0` every epoch (e0 0.562, e3 0.573, all reverted) | **OOD lesson:** teacher transcripts never break greens / repeat / play invalids → **zero gradient on clean data.** You can't train-in a fix for failures the training data doesn't contain |
| 22:54 | **DAgger v1 (`dagger.py`)** | roll the model out greedily on train secrets, relabel its bad-guess boards with the InfoMax teacher's correct guess, SFT | **REVERTED** (base 0.615; e0 0.542, e2 0.552). **616** failure boards over 1400 secrets; corrections diluted **~1:6** by the teacher mix | the honest fix for the constraint-aux no-op — puts the OOD failure states INTO training (train-secret labels, held-out eval, no inference aid). v1 under-weighted the corrections |
| 23:07 | **GRPO full-traj + new reward (`rl_grpo_reward.py`)** | stabilized token-GRPO but reward = full shaped `compute_reward().total` (incl. the new penalties), on `dpo.pt` | **greedy DECLINED** held6 0.615 → upd0 0.604 → upd8 0.583 (valid 0.629→0.610) while *sampled* rollout wins ROSE 33→51/64; KL sane (≤0.009) | proxy-hacking the shaped reward + the GRPO↔greedy mismatch. **The reward fix did NOT rescue GRPO** (best-ckpt held at base, no damage) |
| 23:13 | **GRPO guess-only (`rl_grpo_guessonly.py`)** | same GRPO but action mask credits ONLY the 5 committed guess letters (not the think) | **KL EXPLODED** 0.010 → 0.052 → 0.24 → 1.54 → 2.64 → 12.7 by upd6; stopped | unstable — same knife-edge as guess-only DPO; the full-trajectory **think tokens were acting as a regularizer** |
| 23:15 | **DAgger v2 (improved `dagger.py`)** | all **1852** train secrets for failure coverage; corrections upweighted **×4** to ~50% of the pool, matched teacher sample | **IN PROGRESS** — latest log: rolled 176/1852, failure-boards 83 (no full-463 line yet) | fixes v1's dilution (×4 + full coverage); pending |

> **Evening verdict:** **GRPO is conclusively dead here (10th confirmation) — the reward fix didn't
> help it** (full-trajectory proxy-hacks the shaped reward → greedy *declines*; guess-only KL-explodes).
> The honest best remains **DPO commit-sharpening 0.631**. constraint-aux taught the OOD lesson (no
> gradient where the teacher never fails); **DAgger** is the honest attempt to put those failure states
> in the data (train-secret labels + held-out eval = not memorization) — v1 reverted on dilution, v2
> (×4 corrections, full 1852-secret coverage) is in progress.

#### 2026-06-05 — adversarial audit: held-out contamination (methodology violation, win-impact refuted)

A 4-agent adversarial investigation audited the build. **Library = correct** (231 tests + tool receipts;
no defect). **Pipeline = contaminated** (CONFIRMED methodology violation): held-out words become loss-True
training targets via **4 channels** — **FM-2** CoT candidates (`consistent_candidates(history, ANSWERS)`
over the full pool, `cot_ephemeral_aux.py:44,66,79-83`), **FM-3** teacher guesses (`generate_transcripts`
full pools, `teacher/transcripts.py:50,59`), the **opener leak** (`trace` ∈ `DEFAULT_OPENERS` is a held-out
answer), and **FM-1** selection-on-`held[:96]` reported on full `held`. **But win-inflation was REFUTED:**
FM-1 vanishes into MPS greedy non-determinism (**±2-3pt** on ~100-game slices: held[:96] vs held[96:] flipped
sign across two runs of the same checkpoint, +3.6pt vs −1.4pt), and FM-2/FM-3 are **inert** (the 0.616 model
emits held-out vocab at 9.4%/9.2%, *below* the 20% base-rate; inference is board-only free-gen, never calls
the filter). **0.616 / 0.631 are most likely real — not retracted.** The spell-warmup + aux-validity trie
over the valid-guess list are **legitimate** (spellability only). **Fixed `a41b1b2`** (`generate_transcripts`
gains `valid_pool`/`answer_pool` + leak test; `cot_ephemeral_aux_clean.py` re-runs with train-only pools,
held-out-free openers, disjoint VAL/TEST); **clean re-run in progress** (`runs/cleanrun.log`, expected
≈ 0.60–0.62 to confirm a ≤small delta).

> **Audit verdict:** the **methodology was contaminated and is now fixed**; the **win numbers are likely
> real** (the leak measured inert), with confirmation pending the clean re-run. New standing caveat: **MPS
> greedy eval is ±2-3pt noisy on ~100-game slices** — prefer full-463 / multi-seed for any claimed delta.

#### 2026-06-06 — overnight clean re-run: the audit's "inert" is OVERTURNED; honest held-out ≈ 0.17

The leak-free re-run (train-only candidate + teacher pools, held-out-free openers, disjoint
VAL(`held[:96]`)/TEST(`held[96:]`)) landed and **reversed the audit's win-impact call**. Commit `d00f89d`,
full analysis in [`OVERNIGHT_ANALYSIS.md`](./OVERNIGHT_ANALYSIS.md).

| Stage (all CLEAN) | What it tested | Result (TEST = `held[96:]`) | Takeaway |
| --- | --- | --- | --- |
| SFT (ephemeral-CoT + aux) | the 0.616 recipe, leak-free | **TEST 0.166** (VAL 0.188, full 0.171, valid 0.749, avg 4.66) | **honest collapses 0.616 → ~0.17**; converged (loss 0.43), 3-seed TEST mean 0.1662 (stable) |
| DPO (on clean SFT) | preference sharpening, clean | TEST 0.166 (reverted to base) | **no gain** — can't inject unseen vocabulary |
| GRPO (on best, clean reward) | RL, clean | TEST 0.166 (reverted to base) | **no gain** — 11th GRPO confirmation |
| DAgger / DPO∘DAgger | failure-state relabeling, clean | TEST 0.144 | **slightly hurts** |
| DAgger ×2 | full-coverage ×4 corrections | TEST 0.169 | within the 0.14–0.17 band |
| train/test split proof | is 0.17 a bug? | **TRAIN[:120] 0.858 vs TEST[:120] 0.175** | textbook generalization gap — learned (86% train), cannot generalize; held games emit held-out words **4%** (28/690) vs train answers **66%** (457/690) |

> **Clean re-run verdict:** the audit's "leaks inert" was a **misread** — the FM-2/FM-3 vocabulary leak was
> the **dominant ~45-point lever**, not inert (FM-1 selection bias *was* within noise). The 0.40 → 0.62
> progression was largely **held-out-vocabulary leakage** (held-out answer words fed in as loss-True
> teacher/CoT targets, teaching the held-out vocabulary as playable). **Fundamental wall:** Wordle deduction
> needs the candidate-word vocabulary, so "generalize to held-out" = play words never seen as targets ≈
> impossible; the honest ceiling ≈ spelling knowledge **~0.17**, and RL/DAgger/DPO/GRPO are exhausted
> against it. **Two framings:** (1) strict held-out generalization = **~0.17** (the real ceiling); (2)
> deployed real Wordle (fixed, known 2,315-answer set, no novel secrets) = **~0.62** is a legitimate
> deployed-player score (training on all answers isn't cheating for the real game). **Decide which game
> you're measuring** before the next step.

### Algorithm reference (exact)

The exact per-run algorithm. Every driver is `scripts/<name>.py`; the canonical pieces they call
live in `src/wordle_slm/`. Stated values are the real knobs read from source; "(library default)"
means the run didn't set it and inherited `src/wordle_slm/config/__init__.py`.

#### Shared machinery

Inherited by every run unless its row says otherwise.

- **Engine scoring** (`engine/scoring.py`): two-pass color scoring — pass 1 marks GREEN where
  `guess[i]==answer[i]` and builds a remaining-letter multiset; pass 2 marks YELLOW for a non-green
  position only while that letter has remaining count (decrement on use), else GRAY. An invalid guess
  still consumes a turn (`feedback=None`, rendered as 5×`<gray>`).
- **§5.2 board grammar** (`model/serialization.py`): char vocab = **34** (`<PAD> <BOS> <EOS> <SEP>
  <GUESS> <green> <yellow> <gray>` + a–z; `tokenizer.py`). A completed turn = `<GUESS>` + 5 letters
  + 5 feedback + `<SEP>` (12 tokens; `<GUESS>` IS kept in history — deliberate departure from §5.2 so
  generation and log-prob recompute share an identical context). Prompt = `<BOS> (turn)* <GUESS>`;
  finished game = `<BOS> (turn)* <EOS>`. context_len = **128** (`ModelConfig` default).
- **Model** (`model/transformer.py`): decoder-only **pre-norm** (`norm_first=True`) causal
  transformer, GELU, learned token+positional embeddings, **weight-tied** output head, dropout per
  preset. The action space is the **26 letters only** — generation/log-probs always `log_softmax`
  over the 26 letter logits (a special token can never be emitted). `MODEL_PRESETS`
  (`config/__init__.py`), tuple = d_model × n_layers × n_heads, d_ff, dropout:

  | preset | d_model | layers | heads | d_ff | dropout | ≈params (vocab 34) |
  | --- | --- | --- | --- | --- | --- | --- |
  | `tiny` | 128 | 6 | 4 | 512 | 0.10 | ~1.2M |
  | `base` | 320 | 10 | 8 | 1280 | 0.10 | ~12M |
  | `large` | 512 | 16 | 8 | 2048 | **0.15** | ~50M |
  | `xl` | 640 | 20 | 10 | 2560 | 0.15 | ~98M |

  Several pre-`large`-preset runs hand-build a `ModelConfig` instead (sizes given per row).
- **Spell warm-up** (`sft/pretrain.py`): masked-letter LM over every valid guess (each word →
  `<BOS> <GUESS> w0..w4`), same masked loss as SFT, AdamW. Run-specified epochs/batch/lr.
- **Teacher data** (`teacher/transcripts.py`): plays the **train** secrets with a blend —
  `weak_frac` via `ConsistentGuesser` (opener then a uniform still-consistent word from the **valid**
  list, ~96.7%/4.46) and `1−weak_frac` via `InfoMaxGuesser` (opener then the candidate minimizing
  expected remaining consistent answers over the **answer** pool, ~99%/3.55; `baselines/policies.py`).
  Openers default `("slate","crane","trace","stare","raise","crate")`; most ≥25M runs override with
  `OPENERS=("salet","crane","slate","trace","stare","raise","crate")`. N "passes" = N reseeded
  replays of the train set.
- **SFT objective** (`sft/train.py:sft_loss`): masked next-token NLL over the 26-letter space at the
  5 guess-letter positions after each `<GUESS>` only — `imit = Σ(nll·mask)/Σmask`. With the
  **aux-validity** lever, `loss = imit + λ·aux` where `aux = mean over masked positions of
  −log Σ_letter (softmax · trie_valid_mask)` and `trie_valid_mask` (`valid_continuation_mask` +
  `_valid_trie`) is the set of dictionary-valid next letters given the realized prefix. `λ`
  (`aux_validity_lambda`) **default 0.5**; trie is a training signal only — **inference is never
  trie-aided**. Optimizer AdamW (`weight_decay` default 0.01); best-by-held-out checkpoint kept.
- **GRPO objective** (`rl/grpo.py`, `rl/tracer.py`): group = G same-secret rollouts (sampled
  free-gen). Advantage `A_i = r_i − mean(r_group)`, **no ÷std** (Dr. GRPO `advantage_norm="mean_center"`);
  **zero-variance groups filtered** (`filter_zero_variance=True`). Trajectory log-prob = Σ log p(letter)
  over the guess-letter positions (teacher-forced; logit q−1 predicts letter q). Clipped surrogate
  `min(ratio·A, clip(ratio,1−ε,1+ε)·A)` with **ε=0.2**; ratio uses a frozen θ_old per batch (≡1 at
  inner-epoch 0; K = `inner_epochs` default 1). KL = **k3** `exp(Δ)−Δ−1` to a **frozen π_ref** (the
  SFT checkpoint), Δ = `ref_logp − cur_logp`, **β=0.01** (`kl_beta`); `loss = −surrogate/|tok| +
  β·KL/|tok|`, grad-clip 1.0, linear LR warmup (`warmup_ratio` 0.05), γ=1. Defaults: G=16,
  secrets/update=8, lr=1e-5.
- **Reward** (`rl/reward.py`, `RewardConfig`): per game, knowledge-state carried across turns —
  new-green `a=0.2` (once/pos), new-yellow `b=0.1` (only when it raises a known min-count), invalid
  `−p_invalid=0.5`, clue-violation `−q=0.5` (drops a known green / reuses a known gray),
  **repeat `−repeat_penalty=0.4`** (re-emitting a prior *valid* guess — the SALAD-twice wasted-turn
  case, which is clue-consistent so the clue term misses it), **drop-present
  `−drop_present_penalty=0.3`** (omitting a known-present/yellow non-green letter; greens were already
  covered by the clue term), step `−c=0.02`, terminal **win** `+(win_base 3.0 + win_speed
  0.5·(max_guesses − t))`, **loss** `−loss_penalty 1.0`. Dominance held: `p_invalid>b`, `q>b`, max
  farmable < win_base. The repeat/drop-present terms were **added 2026-06-05 evening** (committed; 2 new
  exact-value tests, suite green) — the "update the reward model" deliverable. Several RL runs
  **replace** this with their own reward (noted per row).
- **Rollout / decode** (`rl/rollout.py`): `play_game` generates each guess letter-by-letter and the
  engine validates it (no candidate list, no consistency filter). Eval = **greedy** argmax,
  free-generation, 6 guesses, on the **463 held-out** (or a stated subsample); training samples
  multinomial. Split (`data/wordlists.py`): seed 0, 80/20 → **1,852 train / 463 held-out**; valid
  list 14,855; answers 2,315; `train_probe` = a fixed train subset for the memorization gap.

#### Per-run algorithm & deltas

Each row gives only what differs from Shared machinery; numbers are exact. "Δ vs …" is the
algorithmic change relative to the named run.

**2026-06-02**

- **3.2M SFT — `scripts/train_run.py`** (`sft_strong.pt`). Model = `ModelConfig()` default
  256×4×8, d_ff 1024 (~3.2M), dropout 0.10. Pretrain 4 ep, batch 512, lr 1e-3. Teacher **3 passes,
  weak_frac 0.5** (default openers). SFT plain NLL (**no aux**), AdamW lr **5e-4**, 60 ep, batch 96,
  grad-clip 1.0, eval every 3 ep on held[:100], best-by-curve. *Baseline run.*
- **4.8M SFT — `scripts/train_sft_big.py`** (`sft_big.pt`). Model 256×**6**×8, d_ff 1024 (~4.8M).
  Pretrain **8 ep**. Teacher **4 passes, weak_frac 0.45**. SFT lr **6e-4** with **CosineAnnealingLR**
  (η_min 6e-5), 70 ep, batch 96. Δ vs train_run: +2 layers, deeper pretrain, +1 teacher pass, more
  InfoMax (weak 0.5→0.45), cosine decay.
- **GRPO on 4.8M — `scripts/train_grpo_run.py`**. Loads `sft_big.pt`; π_ref = frozen copy. Library
  reward + GRPO. **G=16, secrets/update=8, lr 8e-5, kl_beta 0.005**, 200 updates. Secret pool =
  **full valid list minus held-out (~14,392)**, single tier `(None,)`, **replay OFF**; best-by-held-out
  only overwrites if it beats the SFT base. Δ vs the library RL defaults: lr 1e-5→**8e-5**, β
  0.01→**0.005**, non-memorizable 14k pool, no replay.
- **GRPO diverse-14k — `scripts/train_grpo_run.py`** (same script, the 14k-pool/replay-off path
  above). Δ vs "GRPO on 4.8M": identical knobs; logged separately as the diverse-secret RL test
  (reward went negative on rare-word secrets).
- **25M SFT strong teacher — `scripts/train_path_a.py`** (`sft_xl.pt`). Model **512×8×8, d_ff
  2048 (~25M)**. Pretrain **12 ep**. Teacher **5 passes, weak_frac 0.2** (80% InfoMax), `salet`-led
  openers. SFT manual **warmup(300 steps)+cosine**, peak 4e-4 → floor 4e-5, 100 ep, batch 128;
  tracks the probe/held gap. Δ vs train_sft_big: ~5× params, 80% InfoMax (weak 0.45→0.2), 5 passes,
  warmup+cosine schedule, larger batch.

**2026-06-03**

- **25M SFT diverse — `scripts/train_path_a_div.py`** (`sft_div.pt`). Same 25M model; pretrain 10
  ep. Data = **2 passes InfoMax-on-answers (weak_frac 0.3)** + **9,000 `ConsistentGuesser`-on-valid
  games** (random valid secrets minus held-out) for late-game spelling breadth. SFT peak 4e-4 cosine,
  50 ep. Δ vs train_path_a: replaces 3 of the 5 InfoMax passes with 9k Consistent-on-rare-valid games
  (kills the memorization gap, hurts win).
- **beam+dict decode — `scripts/beam_eval.py`** (on `sft_xl.pt`, 250 held). No training. **Beam
  width 12** over the model's own letter distribution; `beam+dict` additionally constrains each beam
  step to a **valid-word trie** (cumulative log-prob; emits the top valid word). Δ vs greedy eval:
  beam search ± dictionary-trie constraint at decode (inference-aided).
- **self-distillation — `scripts/self_distill.py`** (`sft_distill.pt`). Warm-start `sft_xl.pt`;
  generate **1,200 beam+dict (width 8) self-play games** on train secrets + **2 InfoMax passes
  (weak 0.2)**; SFT on the mix, AdamW cosine **1.5e-4→1.5e-5**, 30 ep, best-by-**greedy**-held. Δ vs
  train_path_a: targets are the model's own always-valid beam+dict words (bank spelling into greedy);
  trie touches training data only.
- **no-repeat decode — `scripts/norepeat_eval.py`** (on `sft_xl.pt`, 250 held). **Beam width 10**;
  4 conditions = beam ±dict-trie ±no-repeat (skip any word already guessed this game, else fall back
  to the top beam). Δ vs beam_eval: width 12→10, adds the never-re-emit-a-prior-guess rule.
- **RL #3 validity+consistency — `scripts/rl_consistency.py`** (`rl_cons.pt`). Base
  `sft_distill.pt`. **Monkeypatches `compute_reward`**: per turn invalid −1.0, valid-but-inconsistent
  (via `is_consistent`) −1.0, valid+consistent +0.1, step −0.02, win +3.0+0.5·(6−t), loss −1.0.
  GRPO **G=8, secrets/update=4, lr 5e-5, kl_beta 0.01**, 80 updates, secrets = full-valid-minus-held,
  warmup 8. Δ vs library reward: replaces shaped letter-progress with a flat validity/consistency
  reward (no green/yellow shaping); smaller G and lr.
- **RL #4 per-guess — `scripts/rl_perguess.py`** (`rl_pg.pt`). Base `sft_distill.pt`. **Episode =
  one guess**: at each on-policy board sample **G=8** candidate guesses, reward each (invalid −1,
  inconsistent −1, else 0.2 + 0.2·greens + 2.0 if solves), **mean-center per board**, clipped
  surrogate (ε 0.2) + k3 KL (β 0.01) on the 5-letter per-position log-probs. 5 rollouts/update, 70
  updates, lr 5e-5. Δ vs rl_consistency: trajectory credit → clean per-guess advantage; reward adds
  greens + solve bonus.
- **RL #5 dict-in-the-loop — `scripts/rl_dict.py`** (`rl_dict.pt`). Base `sft_distill.pt`. Behavior
  policy samples guesses **trie-constrained** (`trie_sample`); reward per word = −0.5 if inconsistent
  else 0.1 + 0.15·greens + 1.0 if solves; advantage = mean-centered, clamped ±1.5; loss pushes
  **free-gen** log-prob toward high-advantage words (advantage-weighted, KL-anchored β 0.02), **G=6**,
  5 rollouts, lr 3e-5, 70 updates. Δ vs rl_perguess: candidates drawn from the dictionary trie (not
  free-gen), surrogate is advantage-weighted free-gen log-prob (not a clipped ratio).
- **RL #6 consistency-constrained — `scripts/rl_constrained.py`** (`rl_constr.pt`). Base
  `sft_distill.pt`. Behavior policy samples from the **still-consistent candidate set** (filtered each
  turn, capped 48/board, answer kept reachable); reward 0.1 + 0.15·greens + 1.0 if solves;
  advantage-weighted free-gen push, KL β 0.02, **G=8**, 5 rollouts, lr 3e-5, 70 updates. Δ vs rl_dict:
  candidate pool = consistent set (not full trie) → surfaces the answer ~22% of boards.
- **99M scale — `scripts/train_scale.py`** (`sft_xxl.pt`). Model **768×14×12, d_ff 3072 (~99M)**.
  Pretrain 12 ep (batch 256). Teacher **5 passes weak_frac 0.2** (the train_path_a recipe, old
  answer-only data). SFT peak **2.5e-4** warmup(400)+cosine, 45 ep, batch 64. Δ vs train_path_a:
  only the model size (25M→99M); data/recipe held fixed.
- **co-design 50M + diverse — `scripts/train_codesign.py`** (`sft_codesign.pt`). Model =
  `large` preset (~50M, dropout 0.15). Data = `build_curriculum_pool` (≈14k, difficulty-ordered):
  **3 InfoMax passes (weak 0.2) on the answer secrets** + **5,500 `ConsistentGuesser` games on rarer
  valid words**. SFT peak 3e-4 warmup(400)+cosine, 45 ep, batch 96. Δ vs train_scale: 99M→`large`
  50M, answer-only data → curriculum-pool diverse data.
- **deep 50M converged — `scripts/train_deep.py`** (`sft_deep.pt`). Model = `large` preset (~50M,
  dropout 0.15). Pretrain 12 ep. Teacher **5 passes weak_frac 0.2** (same as train_path_a).
  SFT peak 3e-4 → floor **2e-5**, warmup **500**, **90 ep** (long convergence), batch 96. Δ vs
  train_codesign: drops the diverse Consistent-on-rare data back to pure InfoMax-on-answers (5
  passes), longer training — isolates the model redesign.
- **⭐ aux trie-validity — `scripts/train_auxvalid.py`** (`sft_aux.pt`, **honest best 0.436**).
  Identical to train_deep (50M `large`, pretrain 12 ep, 5 InfoMax passes weak 0.2, peak 3e-4
  warmup(500)+cosine, 80 ep, batch 96) **plus** the aux-validity term in-line: `loss = imit + λ·aux`,
  **λ=0.5**, `aux = −log P(next letter ∈ trie continuations)` at every guess-letter position
  (precomputed per-game trie masks). No trie at inference. Δ vs train_deep: adds the λ=0.5
  aux-validity loss (the only change).
- **RL #7 polish — `scripts/rl_polish.py`** (`rl_polish.pt`). Base **`sft_aux.pt`** (0.436).
  Same monkeypatched validity+consistency reward as rl_consistency. GRPO **G=8, secrets/update=4, lr
  4e-5, kl_beta 0.01**, ≤80 updates / ~1100 s cap, best-checkpoint seeded at the base (cannot
  regress). Δ vs rl_consistency: base is the strongest model (sft_aux), lr 5e-5→4e-5, time-capped.
- **RL #8 info-gain — `scripts/rl_infogain.py`** (`rl_infogain.pt`). Base `sft_aux.pt`. Reward =
  validity/consistency **plus** `+0.1 + β·log(|C_before|/|C_after|)` per valid+consistent guess
  (**β=0.2**, candidate pool = full valid list), step −0.02, win/loss as before. **Rollouts use 10
  guesses** (`G.play_game` monkeypatched, max that fits context_len 128); **eval stays at 6**. GRPO
  G=8, secrets/update=4, lr 4e-5, β 0.01, ≤120 updates / ~1300 s. Δ vs rl_polish: adds the
  information-gain shaping term + 10-guess training rollouts.

**2026-06-04**

- **BPE-on-wordlist 12M — `scripts/bpe_wordle.py`**. **From-scratch BPE on the flat valid list,
  400 merges** → vocab = 7 specials + occurring chunks (~2 tokens/word); guesses generated as subword
  chunks. ~12M model (earlier `base`-class config). Pretrain = word-list LM 8 ep (lr 8e-4); SFT on
  **5 InfoMax passes (weak 0.2)** action-masked, lr 4e-4, 60 ep. Δ vs char SFT: char-34 tokenizer →
  BPE-on-wordlist subwords (the validity lever); guess = chunk sequence.
- **BPE-on-wordlist 50M — `scripts/bpe_wordle.py`** (current on-disk config). Same recipe, model =
  `large` preset (~50M). Δ vs BPE-12M: only model size (win flat → flat-wordlist, not size, is the
  limiter).
- **oreo recipe 11M (honest split) — `scripts/oreo_recipe.py`** (recipe…recipe5.log). **Byte-level
  BPE on 10k TinyStories docs, vocab 2048**; pretrain the transformer on the TinyStories token stream
  (2,000 steps, block 256, lr 6e-4, batch 32), then SFT on InfoMax-teacher games as an **oreo-style
  text transcript** (`guess <word> fb GYBB… win/lose`), action-masked, lr 4e-4 warmup(400)+cosine.
  ~11.5M model, **6 teacher passes (weak 0.2)**. Reports **SEEN/train (0.870)** vs honest held-out
  (0.257). Δ vs char SFT: real-text BPE pretrain instead of spell warm-up + char tokenizer; text
  transcript serialization.
- **oreo recipe 50M (honest split) — `scripts/oreo_recipe.py`** (recipe50.log). Same recipe, model
  **512×16, d_ff 2048, context_len 256 (~50M), 10 teacher passes**, 28 SFT ep, best-by-held-out. Δ vs
  oreo-11M: ~11.5M→50M, 6→10 passes (overfits earlier, generalizes worse).
- **structured-context A/B — `scripts/structured_context.py`**. Model 384×8×6, d_ff 1536,
  **context_len 256** (~14M). Shared spell warm-up; **5 InfoMax passes (weak 0.2)** with **aux λ=0.5**.
  Two SFTs from the same init (25 ep, lr 4e-4): **raw board** vs **board + a derived-state block**
  (`<green>` slots / present / absent letters inserted before each guess). Held 200. Δ vs aux SFT:
  smaller model, +explicit derived-state tokens in the context (the tested lever).
- **pass@N probe — `passk.log`** (on `sft_aux.pt`). No training: **multinomial-sample** N full games
  per secret on 150 held-out and count any-win. greedy 0.453 · pass@1 0.353 · pass@5 0.720 ·
  **pass@10 0.787**. Δ vs greedy eval: sampled decoding, N tries (measures the search gap).
- **CoT A/B 14M — `scripts/cot.py`**. Vocab **35** (adds `<think>`); model 384×8×6, d_ff 1536,
  context_len 256 (~14M). Shared spell warm-up; **5 InfoMax passes (weak 0.2)**. Plain CE (no aux) over
  the loss-masked CoT target = `<think>` + each of **K=3** candidates (1 = the teacher guess + 2 random
  still-consistent answers, shuffled, via `consistent_candidates`) then `<GUESS>` + the guess. A/B from
  the same init: no-CoT (board→guess) vs CoT. Held 200. Δ vs char SFT: +`<think>` token, K=3 candidate
  reasoning block before the guess. ⚠️ later RETRACTED as leaked (cot_show).
- **CoT-50M — `scripts/cot_50m.py`** (`cot_50m.pt`). Same CoT serialization (K=3), model
  **512×16, d_ff 2048, context_len 256, vocab 35 (~50M)**, pretrain 10 ep, **5 InfoMax passes (weak
  0.2)**, 32 SFT ep, lr 4e-4. Δ vs CoT-14M: 14M→50M, otherwise identical CoT recipe. Reported 0.456 —
  inflated by the leak.
- **CoT-50M + aux — `scripts/cot_50m_aux.py`** (`cot_50m_aux.pt`). CoT-50M **plus** the aux-validity
  term at **every** word-letter position (`<think>` candidates AND the committed guess), **λ=0.5**
  (`cot_valid_mask`). Δ vs CoT-50M: adds the aux trie loss (stacks the two honest levers). Killed at
  epoch 24 (subsample 0.406); superseded by the leak finding before a full-463 milestone.
- **CoT integrity teardown — `scripts/cot_show.py`** (cotshow.log). No training; loads `cot_50m.pt`.
  A/B on the SAME model over held 120: **teacher-context** (`play_teacher` rebuilds each past turn's
  `<think>` block with the consistency filter via `cot_prompt` — the leak) **0.450** vs **honest
  self-context** (`play_honest` carries only the model's OWN generated `<think>` forward, board+real
  feedback, filter never called) **0.192**. Δ vs cot_50m eval: removes the consistency-filter
  reconstruction of past reasoning → exposes the −0.258 leak.
- **ephemeral-CoT — `scripts/cot_ephemeral.py`** (`cot_eph.pt`, coteph.log). **One training example
  per turn**: history is **board-only** (`<GUESS>` guess + feedback + `<SEP>`, **no `<think>`**), target
  = fresh `<think>`(K=3) + `<GUESS>` + guess. At inference the prompt is board-only, the model
  regenerates reasoning each turn and **discards** it (never enters later context) — train and inference
  distributions are now identical, filter never called. Model = 50M CoT config (vocab 35); **4 teacher
  passes (weak 0.2)**; 30 ep, batch 128, lr 4e-4. Δ vs cot_50m: past-turn `<think>` removed from
  context (ephemeral scratchpad) — removes both the leak and the train/infer shift. **Full held-out 463
  = 0.430** (best ckpt e29); +2.8pts over the matched no-CoT baseline (0.402).
- **ephemeral-CoT + aux — `scripts/cot_ephemeral_aux.py`** (`cot_eph_aux.pt`). Ephemeral-CoT **plus** the
  aux-validity term, **λ=0.5 GATED to the current-turn supervised positions** (past guesses live in the
  board-only history → must not get aux; `aux_pos = (vmask>0) * loss_mask`). 50M, **50 ep, cosine LR
  4e-4→4e-5, 5 teacher passes**, batch 128. Δ vs ephemeral-CoT: adds gated aux + longer cosine schedule.
  **Full held-out 463 = 0.616** (285/463), valid 0.788 — super-additive with CoT; the prior honest best
  and the base for all 2026-06-05 RL/DPO runs.

**2026-06-05** (RL on the 0.616 base + DPO; shared: batched multi-game roller — sample ~80 games in
parallel, right-pad causal, per-seq finish; `play` = greedy ephemeral-CoT; honest = TRAIN-secret labels
only, held-out greedy eval, no inference rules)

- **expert-iteration — `scripts/rl_expert_10row.py`** (`rl_expert.pt`). Per iter: sample **K=12**
  rollouts/secret (10-row, temp 0.9, 400 secrets/iter), keep ≤2 shortest wins (≤8 turns), **rebuild with
  CLEAN teacher think + the model's winning guesses** (RAFT — *not* the noisy sampled think, which
  degraded it), SFT (aux λ=0.5 + teacher-mix), lr 3e-5, 3 ep, **revert-on-regress**. Δ vs SFT: RL via
  self-play win distillation at 10 rows. held10 0.604→**0.646** (full-463 ≈ 6-row 0.62 / 10-row 0.637).
- **higher-ceiling / reachability — `scripts/rl_expert_tail.py`** (`rl_expert_tail.pt`). Pass 0 = **all
  1852 secrets, K=10, temp 1.0** (coverage); passes 1–3 = unsolved tail only, **K=24/32/48, temp
  1.1/1.2/1.3**; accumulate the union of clean-think wins, SFT, revert-on-regress. Δ vs expert-iter:
  full coverage + tail-focused high-K. **Reachability = 1843/1852 = 99.5%**; full-coverage SFT reverted
  (no gain) → coverage is not the bottleneck.
- **GRPO polish — `scripts/rl_grpo_polish.py`** (`rl_grpo.pt`). Token-level GRPO on the CoT policy:
  **G=8, 8 secrets/update**, advantage `A=r−mean(group)` (no ÷std), clip **ε=0.2**, **k3 KL to frozen
  ref β=0.05**, reward `win·(2+0.25·(10−t)) − 0.1·invalid`, lr **5e-6**, **eval-mode policy forward (no
  dropout)**. Δ vs the failed first try: eval-mode forward + β 0.01→0.05 + lr 1e-5→5e-6 (the first
  attempt's dropout-on forward made KL explode 12→294 and degraded). Result: stable but **flat** (6-row
  0.622, KL ~0.005 — policy barely moves).
- **self-consistency probe — `scripts/self_consistency.py`** (no training). Per turn, sample **N=12**
  traces (temp 0.9), commit the **majority-voted** guess (pure vote — no dict/filter); also pass@N (any
  of N sampled games wins). held 150, 6-row: greedy 0.607 · vote **0.627** · **pass@12 0.953**.
- **DPO, noisy pairs — `scripts/dpo_commit.py`** (`dpo.pt`). Sample N=8 rollouts/secret; pair at the
  **first win/loss divergence board** (chosen=winning think+guess, rejected=losing). DPO `−logσ(β·((logπ−logπ_ref)_chosen − (…)_rejected))`, **β=0.1**, lr 5e-6, ref=frozen base, 4 ep, response-token
  logp, revert-on-regress. Δ vs SFT: preference loss on commits. **Full-463 6-row 0.631** (+1.5);
  pref_acc 0.60→0.73 — capped by noisy credit.
- **DPO, decisive-board — `scripts/dpo_decisive.py`** (`dpo_decisive.pt`). Find decisive boards (last 2
  turns of winning rollouts, secret reachable), **resample M=14 responses/board**, pair **chosen =
  commits the secret** vs **rejected = valid + consistent + ≠ secret** at the same board (clean label).
  Same DPO (β=0.1, lr 5e-6, 5 ep, revert-on-regress). Δ vs dpo_commit: clean credit (decisive-board
  re-sampling) instead of noisy first-divergence pairs. **Flat → reverted to 0.616**: DPO logp over the
  whole think+guess let the long `<think>` dominate (loss fell, held6 dropped — think dilution).
- **DPO, guess-tokens only — `scripts/dpo_guessonly.py`** (`dpo_go.pt`). Same noisy first-divergence
  pairs, but `guess_logp` masks the DPO logp to the **5 committed guess-letter positions only** (the
  commit, not the scratchpad). Δ vs dpo_commit: score guess-tokens only. **Knife-edge, no gain**: lr
  5e-6 → loss flat at 0.693 (5-token gradient too weak to move); lr 3e-5 → collapse (held6 0.000, valid
  0.000 — the letter distribution breaks). The think tokens in full-response DPO regularize the update;
  isolating the commit removes that. **Full-response DPO 0.631 stands.**

**2026-06-05 (evening)** (reward-model update + 4 failure-mode runs; all on the 0.616/0.631 CoT policy,
batched roller, honest = TRAIN-secret labels, held-out greedy eval, no inference rules)

- **reward-model update — `src/wordle_slm/rl/reward.py` + `RewardConfig`** (committed). Adds two shaped
  terms to the per-game reward (see Shared machinery): **`repeat_penalty=0.4`** (re-emitting a prior
  *valid* guess — clue-consistent so the `_violates_clue` term misses it; the SALAD-twice wasted turn)
  and **`drop_present_penalty=0.3`** (committing a guess that omits a known-present/yellow non-green
  letter). Greens were already penalized by the existing clue-violation term `q`. 2 new exact-value
  tests; full suite green. Δ vs the prior reward: +2 penalty fields (default-on, dominance preserved).
- **constraint-aux — `scripts/cot_constraint.py`** (`cot_eph_constraint.pt`). Fine-tunes
  `cot_eph_aux.pt` (0.616) with three extra differentiable aux terms gated to current-turn positions:
  **green-keep** `−log P(known-green letter)` (λ_g=0.5), **yellow-reuse** `−log P(known-yellow appears
  across the 5 slots)` (λ_y=0.3), **anti-repeat** `−log(1 − P(commit == any past guess))` (λ_r=0.3), on
  top of the existing aux-validity (λ_v=0.5). 20 ep, cosine lr 1e-4→1e-5, batch 128, revert-on-regress.
  Δ vs cot_ephemeral_aux: +3 constraint aux terms. **NO-OP — reverted:** the aux terms computed **g=0
  y=0 r=0 every epoch** (teacher transcripts never break greens / reuse-fail / repeat → no gradient);
  held win 0.604 base → 0.562 (e0) / 0.573 (e3), all reverted. The OOD lesson.
- **DAgger v1 — `scripts/dagger.py`** (`dagger.pt`). Roll the model out greedily (temp 0.02) on **1400**
  train secrets; collect every board where it does something bad (invalid word / repeats a past guess /
  drops a known green); add a training example at each whose target is the **InfoMax teacher's** correct
  valid guess; SFT on `corrections + 1 InfoMax teacher pass`. lr 6e-5. Δ vs constraint-aux: puts the OOD
  failure *states* into the data (honest — model-visited states, teacher labels on TRAIN secrets), the
  fix for the no-gradient no-op. **616 failure boards collected**; corrections diluted **~1:6** by the
  teacher mix → **REVERTED** (base 0.615; e0 0.542, e2 0.552).
- **GRPO full-traj + new reward — `scripts/rl_grpo_reward.py`** (`rl_grpo_reward.pt`). The stabilized
  token-level GRPO of rl_grpo_polish (eval-mode forward, **ε=0.2, k3 KL β=0.05, lr 5e-6**, advantage
  `r−mean(group)` no ÷std, zero-variance filter, G=8, 8 secrets/update, 40 updates) on `dpo.pt`, but
  reward = **`compute_reward(g, RewardConfig()).total`** — the full shaped reward including the new
  repeat + drop-present penalties (the prior GRPO used an ad-hoc win+speed−invalid reward). Δ vs
  rl_grpo_polish: reward swapped to the full shaped `compute_reward`. **Greedy DECLINED** held6 0.615 →
  0.604 (upd0) → **0.583 (upd8)**, valid 0.629→0.610, while *sampled* in-group wins rose **33→51/64** and
  KL stayed sane (≤0.009) = proxy-hacking the shaped reward + the GRPO↔greedy mismatch; best-ckpt held at
  base (no damage). The reward fix did NOT rescue GRPO.
- **GRPO guess-only — `scripts/rl_grpo_guessonly.py`** (`rl_grpo_guessonly.pt`). Identical to
  rl_grpo_reward (same shaped reward, ε=0.2, β=0.05, lr 5e-6, G=8) **except** the action mask credits
  **only the 5 committed guess-letter positions** (after the last `<GUESS>`), not the `<think>`. Δ vs
  rl_grpo_reward: guess-only credit (mirrors the dpo_guessonly experiment). **KL EXPLODED:** 0.010 →
  0.052 → 0.24 → 1.54 → 2.64 → **12.7 by upd6** → stopped. Same knife-edge as guess-only DPO — the
  full-trajectory think tokens were acting as a regularizer.
- **DAgger v2 — `scripts/dagger.py`** (improved; `N_SECRETS=1852`). Same DAgger loop but over **all 1852
  train secrets** (full failure coverage, not 1400) with the corrections **upweighted ×4** to ~50% of the
  pool (`dagger×4 + teacher_keep`, `len(teacher)=min(·, 4·len(dagger))`) — fixes v1's ~1:6 dilution. Δ vs
  DAgger v1: full coverage + ×4 correction upweight. **IN PROGRESS** — runs/dagger2.log shows rolled
  176/1852, failure-boards 83 (no full-463 line yet).

### Current standing

**Headline (2026-06-06 clean re-run, commit `d00f89d`):** the honest **strict-held-out** best is
**~0.17 — plain clean SFT** (ephemeral-CoT + aux), 3-seed TEST mean **0.1662**. **RL/DPO/DAgger/GRPO are
exhausted** against the vocabulary wall (clean leaderboard all 0.14–0.17; DPO/GRPO revert to base, DAgger
hurts). The **deployed-fixed-answer-set** best (training on all 2,315 known answers, no novel secrets) is
**~0.62** — a legitimate *deployed-player* score for the real game, **not** a strict-generalization number.
**Recommendation: decide which game is being measured** — (a) "deduce unseen words" → ~0.17 is the ceiling
for this approach and the productive next step is a reframing (e.g. a separately-trained candidate proposer,
or explicitly adopting the fixed answer set), not more RL; (b) "best real-Wordle player" → the full-answer
~0.62 model is correct and honest *for that task*.

Everything below documents the *pre-correction* (contaminated-methodology) progression. Under the
deployed-fixed-answer framing it is the real-game player; under strict held-out it is held-out-vocabulary
leakage — read it accordingly.

The pre-correction best (6-row greedy, free-generation, no inference rules) was **DPO commit-sharpening = 0.631 held-out**
(`runs/dpo.pt`, `scripts/dpo_commit.py`), built on top of **ephemeral-CoT + aux-validity = 0.616**
(`runs/cot_eph_aux.pt`) — both now **superseded** by the 2026-06-06 clean re-run above (~0.17 strict-honest /
~0.62 deployed). The SFT
base stacks two **orthogonal honest levers** super-additively
(no-CoT/no-aux 0.402 → +aux 0.436 → +CoT 0.430 → **+both 0.616**): the *ephemeral CoT scratchpad* (enumerate
candidates, commit, discard the reasoning — filter never at inference) for **search/strategy**, and the
*aux-validity trie loss* (dictionary baked into the weights; no trie at inference) for **spelling**. That
already **exceeds** the inference-aided beam+dict mark (0.58–0.60), honestly.

The remaining bottleneck is the **commit gap**: reachability is **99.5%** (sampling wins almost every train
secret) and **pass@12 = 0.95**, yet greedy commits wrong — so the knowledge is there; the model just doesn't
output it. Methods that *reweight sampled outcomes* can't move it: **GRPO is now conclusively dead — 10
formulations** (the stabilized run barely moves, and the 2026-06-05 evening run that fed it the *updated*
shaped reward made greedy **decline** 0.615→0.583 by proxy-hacking the reward while its sampled rollout wins
rose; the guess-only variant KL-exploded), and **self-consistency voting adds only +2pts** (the winning line
is a minority, not the mode). What helps is *sharpening the commit with clean training signal*:
**expert-iteration** unlocked the extra rows (10-row 0.604→0.637), and **DPO** is the only method to lift
honest 6-row greedy (0.616→0.631), currently limited by credit-assignment noise. The **reward model was
updated** (committed: repeat + drop-present penalties) but it neither rescued GRPO nor could be trained in
directly: **constraint-aux was a no-op** (the teacher transcripts never break greens / repeat / play invalids,
so there's no gradient for an in-weights fix — the OOD lesson). The honest answer to OOD failures is to put
those states into the data — **DAgger** rolls the model out on train secrets and relabels its bad boards with
the teacher (train-secret labels + held-out eval = not memorization); v1 reverted on ~1:6 dilution and **v2
(×4 corrections, full 1852-secret coverage) is in progress**.

Dead ends remain dead: **BPE/real-text only wins by memorizing the answer set** (oreo-ai's 0.89 was train/test
contamination — reproduced as SEEN 0.87 / honest held-out 0.257), and **context management is a non-lever**
(explicit state hurt; length neutral). The standing **honesty rule**: answer-derived signals are fine in
*training* (teacher, engine-labeled wins, aux trie, DPO preference labels — all on TRAIN secrets); inference is
always **greedy on the strict held-out split with zero rules** — no dictionary, filter, candidate list, or
verifier.

**Audit standing (2026-06-05 → OVERTURNED 2026-06-06).** The **library is audited-correct** (231 tests +
tool receipts; no defect — unchanged). The training/eval **pipeline had 4 held-out-leak channels** — FM-2
(CoT candidates over the full answer pool), FM-3 (teacher guesses over the full pools), the `trace` opener
leak, and FM-1 (selection on `held[:96]`) — a **confirmed methodology violation, now fixed** (`a41b1b2`). The
2026-06-05 audit called the leaks **inert** ("numbers likely real ~0.60"); the **2026-06-06 clean re-run
(`cot_ephemeral_aux_clean.py`, `runs/cleanrun.log`, commit `d00f89d`) OVERTURNED that** — the honest
strict-held-out **collapsed from 0.616 to ~0.17**, verified by the TRAIN 0.858 vs TEST 0.175 generalization
gap (converged, loss 0.43; held games emit held-out words only 4% vs train answers 66%). **FM-1 selection
bias *was* within noise (the DA was right there); but the FM-2/FM-3 vocabulary leak was the dominant
~45-point effect — not inert.** The 0.40 → 0.62 progression was largely held-out-vocabulary leakage. New
eval-discipline caveat (still holds): **MPS greedy eval is ±2-3pt non-deterministic on ~100-game slices** —
prefer **full-463 / multi-seed** evals; the clean 0.166 was confirmed by a 3-seed TEST mean (0.1662).
