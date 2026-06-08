# Day run — all-possibilities sweep (2026-06-07)

**Mandate:** a full day of compute, many diverse solutions across every axis, honest throughout, no
stopping; results captured here + in the README. Baseline to beat: stage-1 (`cot_eph_aux_fair.pt`) =
**free-gen TEST 0.281 / valid 0.662**; best honest aided = best-of-128 **0.719**. Confirmed dead: all
6 training methods on the 50M base (DAgger, distillation, GRPO, info-gain XIT ×2, DPO) → null at 0.281.

Honesty rules unchanged: train-only secrets, dictionary pools, eval = pure free-gen on disjoint TEST
(aided numbers — constrained decode, best-of-N — always labeled "aided"). Every run best-by-checkpoint
from its own baseline, so nothing regresses.

## The four axes (sequential on one GPU; report + README updated as each lands)
1. **Scale** — param-scaling SFT at tiny(1.2M) / base(12M) / large(50M, done) / xl(99M). *Does honest
   held-out scale with model size?* (`scale_sft.py`, `SIZE=`)
2. **Data** — SFT with the FULL 14,855-word dictionary as TRAINING SECRETS (deduction is answer-agnostic,
   so not data-capped). *Does more diverse deduction practice lift held-out?*
3. **Inference search** — beam search over the validity trie; best-of-N temperature sweep; best-of-N on
   the largest model (*does inference-compute gain stack with scale?*).
4. **Framing** — deployed (train on all answers) + best-of-N = the highest *honest-for-the-real-game*
   number (labeled deployed, not held-out generalization).

## Results (TEST, updated live)
| axis | experiment | free-gen win | best-of-N | valid | note |
|---|---|---|---|---|---|
| — | stage-1 (large, 50M) baseline | 0.281 | 0.719 (N=128) | 0.662 | the bar |
| scale | tiny (1.2M) SFT | **0.163** | … | 0.567 | TRAIN 0.230 — underfits |
| scale | base (12M) SFT | **0.251** | … | 0.636 | TRAIN 0.485 — gap growing with size |
| scale | large (50M) = stage-1 | **0.281** | 0.719 | 0.662 | the peak |
| scale | xl (99M) SFT | **0.270** | … | 0.591 | TRAIN[:200] 0.550 — **turns over**: held-out drops vs 50M, validity drops, gen-gap widest |
| data | full-dict-secrets SFT | — | — | — | **deprioritized**: InfoMax teacher is O(n²) over the candidate pool → full-dict secrets are compute-bottlenecked; cheap consistent-teacher version = non-strategic (uncertain benefit) |
| search | beam over trie (stage-1) | — | **0.550** | 1.000 | aided, **deterministic**: sequence-argmax over real words; +11pts over greedy-constrained 0.436, best non-sampling number |
| search | best-of-64 on xl | — | … | … | running |
| framing | deployed + best-of-N | … | … | … | queued |

## Running conclusion
*(updated through the day; final synthesis + best model at the end.)*

**Scale axis — DONE, and the answer is "no".** The honest held-out sweep is non-monotonic:
**tiny 0.163 → base 0.251 → large(50M) 0.281 → xl(99M) 0.270.** Held-out win **peaks at ~50M and
turns over at 99M**: the bigger net memorizes the train secrets harder (TRAIN[:200] 0.55 vs TEST 0.27,
the widest gap in the sweep) *and* spells worse (valid 0.662 → 0.591). So the deduction/vocabulary wall
is **not** a capacity problem — adding parameters past 50M buys memorization, not generalization. This
matches the audit story: the lever is data honesty + test-time compute, not weight count.

---

## Autonomous push #3 (2026-06-07, unattended): chase the REAL in-weights number

**Mandate:** "do anything for the best real and honest results, rework stages, question everything, do
not stop." So this push targets the **free-gen greedy held-out** number (0.281) — the only number with
*no* inference aid — rather than the dictionary-aided ones.

**The governing fact (questioned and re-derived):** on the *same* held-out TEST, **best-of-N reaches
~0.70 where greedy gets 0.281.** The capability *generalizes*; greedy decoding fails to surface it. And
honest inference *without* the dictionary can't close it (no-dict sample+vote = 0.243 < greedy) — so the
dictionary is the only inference lever, and it's labeled aided. ⇒ **To raise the real number we must move
capability into the weights.**

**Why the obvious training levers are dead ends (so we don't re-run them):**
- *Plain RL / RFT-on-train is redundant.* The near-optimal teacher already wins ~all TRAIN secrets, so
  the model's own wins on the same secrets add no coverage. (This is the most likely reason info-gain-XIT
  and the clean GRPO/DAgger family were all null.)
- *More scale* memorizes harder (above).
- *More diverse secrets* hurt before (0.188–0.220): obscure dictionary words as secrets don't match the
  answer-like held-out distribution.

**The one bet with theoretical grounding — on-policy self-distillation (RAFT/RFT), stage-2.** Distill the
model's *own* per-turn best-of-N **valid-vote winning games** (reward-1 samples = reward-weighted MLE =
a policy-improvement step) back into greedy, anchored by a teacher slice to avoid forgetting. Rationale:
teacher trajectories are *off-policy* (teacher's argmax word ≠ the model's), so greedy can't reproduce
them; the model's sampled wins are *on-policy* (words it already ranks high) → greedy-reproducible.
Honest: rollouts use only the model's samples + public spelling + majority vote (never the secret);
"won" is judged on TRAIN secrets; eval = free-gen greedy on disjoint TEST. Driver: `scripts/rft_distill.py`.

| stage-2 step | free-gen TEST | valid | read |
|---|---|---|---|
| stage-1 baseline | 0.281 | 0.662 | the bar |
| RFT pilot (LR 1e-4) | — | — | NULL (LR damaged the minimum) |
| **RFT best-shot** (LR 3e-5, on-policy ×3) | **0.319** (117/367) | 0.681 | **first method to beat stage-1** (+0.038, ~1.6σ) |
| **validity-max** (aux λ3 + constrained games) | **0.305** (112/367) | 0.712 | validity rose 0.662→0.712, win partly followed (+0.024) |
| STaR iter-2 (RFT from the 0.319 ckpt) | running | | the confirmation: real signal climbs, noise falls back |

**The turn.** Two warm-start fine-tunes **beat stage-1 on disjoint TEST** — the first to do so after 8
nulls. RFT best-shot **0.319** and validity-max **0.305** (vs 0.281). The gentle LR (3e-5) was the
unlock: the pilot's 1e-4 damaged the converged minimum; 3e-5 nudges it. **Honest caveat:** both select
the best of ~16 epochs on a 96-secret VAL → selection-over-noise inflation; +0.038 is only ~1.6σ on 367
TEST. So I'm **not** banking it yet — the STaR iteration (roll out from the improved model, distill again)
is the arbiter: a real gain compounds, a noise blip regresses. Also queued: a teacher-only-fine-tune
**control** to rule out "just more training," and stacking validity-max on top of the RFT checkpoint.

**Pilot read.** The rollout is healthy — best-of-12 wins **0.797** of 600 train secrets (789 winning games
kept) — so the data is there. But the LR-1e-4 fine-tune visibly *damaged* the converged minimum (0.344 →
0.24 → 0.281) and never recovered. **The best-shot (gentle LR 3e-5 + on-policy upweight ×3) tells a
different story:** it *did* clear stage-1 — VAL 0.281→0.302→**0.365** by epoch 2 (> warm-start 0.344).
So the pilot's null was partly an LR/optimization artifact, not purely a data verdict. Whether the lift is
real (vs noise on 96 VAL secrets) and whether it survives on disjoint TEST is pending the final eval —
**not claiming a win yet.**

## The composition lever (the actually-named lever, now actually pulled)

Caught a gap: I kept calling "valid-word composition into the weights" *the* honest lever, but nothing
running was pulling it (RFT targets win, not composition). The 3 spot-checked losses (mango/dried/hocky)
all show the same thing — **deduction works, spelling-composition fails**: the model has the answer's
letters in the right slots but emits near-words (manim, bried, hocky). Perfect spelling is worth +15pts
(constrained-mask at inference = 0.436 vs greedy 0.281). The closest prior (`distill_constrained`) raised
in-weights free-gen validity 0.62→**0.80** but win stayed flat — it *stopped at 0.80*. The untried
question: **does pushing in-weights validity past 0.80 toward ~0.95 make free-gen win climb toward 0.436,
or stay flat (proving deduction-generalization, not spelling, is the wall)?**

`scripts/validity_max.py`: warm-start stage-1, fine-tune on teacher games (hold deduction) + the model's
own *constrained-decode* (always-valid) games + a **cranked aux (λ=3 vs 1.0)**; track free-gen win AND
validity every epoch. Queued behind the best-shot. Decisive either way.
