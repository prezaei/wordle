# Overnight clean-pipeline run — deep analysis (2026-06-06)

## TL;DR (the hard, honest result)

When **all held-out leakage is removed**, the honest held-out win rate of the best recipe collapses
from the reported **0.616 to ~0.17**. This is **verified, not a bug**: the clean model wins **85.8% on
train** but only **17.5% on held-out** — it converged (loss 0.43), it just *memorized the train answer
vocabulary and cannot generalize to unseen words*. **The project's 0.40 → 0.62 "progress" was largely
held-out contamination** (the teacher and CoT-candidate pools were drawn from the full answer set, so
the model was being taught the held-out answer words as playable guesses). The clean re-run overturns
the 2026-06-05 audit's "leaks were inert" conclusion.

## The overnight run (all stages CLEAN: train-only pools, safe openers, disjoint VAL/TEST)

| setup | VAL (held[:96]) | **TEST (held[96:], honest)** | full | valid | avg |
|---|---|---|---|---|---|
| SFT (ephemeral-CoT + aux) | 0.188 | **0.166** | 0.171 | 0.749 | 4.66 |
| DPO (on SFT) | 0.188 | 0.166 | 0.171 | 0.749 | 4.66 |
| GRPO (on best, clean reward) | 0.188 | 0.166 | 0.171 | 0.749 | 4.66 |
| DAgger (on SFT) | 0.146 | 0.144 | 0.145 | 0.813 | 4.64 |
| DPO ∘ DAgger | 0.146 | 0.144 | 0.145 | 0.813 | 4.64 |
| DAgger ×2 | 0.125 | 0.169 | 0.160 | 0.794 | 4.89 |

Winner by VAL: the plain **SFT base, TEST = 0.166** (3-seed TEST mean **0.1662**, so it is stable, not
MPS noise). DPO and GRPO reverted to the SFT base (no gain); DAgger slightly *hurt*. Everything sits in
a 0.14–0.17 band: **the clean ceiling is ~0.17 and no downstream lever moves it.**

## The decisive verification (why 0.17 is real, not a broken run)

| check | clean SFT |
|---|---|
| win on **TRAIN[:120]** | **0.858** |
| win on **TEST (held[96:])[:120]** | **0.175** |
| final training loss | 0.43 (converged from 1.93) |
| held-game committed guesses that are **held-out words** | **28 / 690 = 4%** |
| held-game committed guesses that are **train answers** | 457 / 690 = 66% |
| sample held secret `befit` | `slate meter debit hefit begut penco` → LOSE (never emits `befit`) |

The 0.858/0.175 split is a textbook generalization gap: the model **learned** (86% on train) but **does
not generalize**. The mechanism is explicit — trained only on train answers as output targets, it plays
train answers (66%) and almost never produces a held-out word (4%), so it cannot guess held-out secrets.

## Why the leak was the lever (and the audit's "inert" was wrong)

The contaminated pipeline fed held-out answer words into training as **loss-True targets** via teacher
guesses (`generate_transcripts` over the full pool) and CoT `<think>` candidates
(`consistent_candidates(history, ANSWERS)`). That **taught the model the held-out answer vocabulary as
playable, clue-consistent words** — exactly the knowledge needed to commit a held-out secret. Remove it
and the model can still *spell* held-out words (pretrain/aux validity — legitimate) but its *policy*
never learned they are answers to play → 4% emission → 0.17 win.

The audit's DA concluded "leaks inert" from the contaminated model emitting held-out words "9.2%, below
the 20% base-rate." That was a **misread**: 9.2% of *all* guesses being held-out words is the mechanism
by which it wins held-out secrets, and "20%" was the wrong baseline. The clean re-run is ground truth and
refutes that call. (FM-1 selection bias *was* within noise, as the DA found — but the FM-2/FM-3
vocabulary leak was the dominant, ~45-point effect.)

## The fundamental insight

**Wordle "deduction" requires the candidate-word vocabulary.** Clue-narrowing only works if you can
enumerate words that fit the pattern — i.e. *knowing the words*. Asking the model to "generalize to
held-out secrets" is asking it to **play words it was never told exist** (from `_oist` you cannot produce
`joist` unless you know `joist` is a word). So the honest-held-out ceiling is essentially capped by
*spelling knowledge alone* (~0.17 here); the only way to lift it is to give the model the answer
vocabulary — which, under the strict held-out rule, *is* the leak.

## Two legitimate framings (your call)

1. **Strict held-out generalization** (the project's stated value, "deduce, don't look up"): honest =
   **~0.17**. The model cannot deduce unseen words; CoT/aux/DAgger/DPO/GRPO are exhausted.
2. **Deployed real Wordle** (the answer set is **fixed and known** — 2,315 words, no novel secrets):
   training on all answers is *not cheating for the actual game*; that model plays the real game at
   **~0.62** (the "contaminated" number is the legitimate *deployed* score). The held-out split is an
   artificial generalization probe we imposed.

Both are real; they measure different things — 0.17 = "can it deduce novel words" (no); 0.62 = "can it
play the real, fixed-vocabulary game" (yes, decently).

## Best setup + recommendation

- **Strict held-out standard:** the **plain clean SFT (ephemeral-CoT + aux), TEST 0.166** is the best —
  DAgger/DPO/GRPO add nothing (they can't inject unseen vocabulary). The honest ceiling for
  50M-from-scratch is ~0.17; more RL is futile against a vocabulary wall. The next move is a different
  framing, not more training.
- **Deployed-Wordle standard:** the full-answer **ephemeral-CoT + aux (+DPO)** at ~0.62 is the best
  *player* — legitimate if the fixed known answer set is accepted as fair game; just label it
  "deployed real-Wordle," not "held-out generalization."

**Recommendation:** decide which game you're measuring. (a) "Deduce Wordle on unseen words from scratch"
→ honest answer is **~0.17 and that's the ceiling for this approach**; the productive next step is a
reframing (e.g. a *separately-trained* candidate proposer, or explicitly adopting the fixed answer set),
not more RL. (b) "Best real-Wordle player" → the full-answer ~0.62 model is correct and honest *for that
task*.

## Honesty / methodology notes
All overnight stages were leak-free (train-only candidate + teacher pools, held-out-free openers,
disjoint VAL/TEST). MPS greedy eval is ±2–3 pt noisy on ~100-game slices, but 0.166 was confirmed by a
3-seed TEST mean (0.1662) and the 0.858/0.175 train/test split, so it is robust. Library code remains
audited-correct; this is purely an honest re-measurement of the *methodology*, not a code bug.
