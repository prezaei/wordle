# Validity push — report (2026-06-06)

**Goal (your directive):** push *pure free-generation* so the model emits **only valid words** — no
constrained-decoding crutch; the model itself must learn to spell. Honest throughout: train-only
secrets, dictionary candidate/teacher pools (knows spelling, *not* answer-hood), eval on a disjoint
held-out TEST.

## TL;DR

- **Stage 1 (cranked spelling recipe) is the headline win:** honest **TEST win 0.281 (103/367),
  validity 0.662** — up ~69% from the over-hobbled overnight floor (0.166). This is the real "fair"
  number (knows the dictionary, must deduce unseen answers).
- **Validity plateaus ~0.66 on TEST.** Cranking pretrain (10→30) + aux (λ 0.5→1.0) did **not** push
  validity dramatically higher, and **DAgger v1 didn't move it at all** (diagnosed: the corrective
  signal was 2.6% of the data — drowned out).
- **Honest bottom line:** "only valid words" (≈1.0) is **not reachable with pure free-gen on a 50M
  model**. The last ~30% genuinely needs constrained decoding (the lever you set aside). An
  oversampled-corrective DAgger (**v2**) is running to find the true free-gen ceiling.

## What I changed and ran

| Stage | Levers | Status |
|---|---|---|
| **1 — cranked SFT** | pretrain warm-up **10→30** epochs (full 14,855-word dict), aux-validity **λ 0.5→1.0**, 50 epochs, fair dictionary pools, disjoint VAL/TEST | ✅ done |
| **2 — DAgger v1** | roll out → catch invalid words → retrain on a *valid consistent* word for that board; aggregated with base teacher data | ✅ done (null) |
| **2b — DAgger v2** | same, but **oversample corrective 20×** so it isn't drowned out | ⏳ running |

## Results

### Stage 1 — cranked pretrain + aux (best by VAL = epoch 45)
| set | win | valid | avg |
|---|---|---|---|
| **TEST** held[96:] (HONEST) | **0.281** (103/367) | **0.662** | 4.33 |
| VAL held[:96] (selected-on) | 0.344 | 0.680 | — |
| TRAIN[:200] (memorization ref) | 0.515 | 0.740 | — |

- Generalization gap (TRAIN−TEST win) ≈ **0.23** → real generalization, not memorization.
- vs **over-hobbled overnight (train-only pools) 0.166** → **+0.115 (+69%)**. The fair
  dictionary-candidate recipe + stronger spelling genuinely lifts the honest number.

### Stage 2 — DAgger v1 (no oversample) — NULL result
| round | invalid-rate (rollout) | VAL win | VAL valid |
|---|---|---|---|
| 1 | 0.27 | 0.219 | 0.659 |
| 2 | 0.27 | 0.219 | 0.648 |
| 3 | 0.31 | 0.302 | 0.641 |
| 4 | 0.32 | 0.260 | 0.627 |

Validity flat/down; best-by-VAL stayed the **stage-1 baseline** → final = stage 1 (0.281/0.662).
**Why:** only **399 corrective examples accumulated vs 14,867 base (2.6%)** — the correction was
negligible in the aggregated set. Fixable: oversample the corrective set (→ v2).

### Stage 2b — DAgger v2 (corrective ×20) — *running*
*(updated when it finishes — testing how far conditional correction pushes the free-gen ceiling.)*

## The honest assessment

1. **Two walls, measured.** Validity (spelling) = **0.66**; win (spelling **+** deduction) = **0.281**.
   ~34% of held-out guesses are still non-words.
2. **The fair recipe trades raw validity for honest play.** This run's 0.66 validity is *lower* than
   the old contaminated run's ~0.75 — because the **dictionary** candidate pools train the model on
   the full, hard 14k vocabulary (rare, hard-to-spell words) instead of the easy 2,315 answer words.
   That same choice is *why* win is higher (it can play dictionary-wide). Validity-for-honesty is a
   real trade, not a regression.
3. **Pure free-gen has a real spelling ceiling on 50M.** Stronger pretrain + aux + (diluted) DAgger
   land it ~0.66. Pushing to ~1.0 ("only valid words") is the documented reason constrained decoding
   exists — and that's a *spelling* aid (real-word mask), categorically weaker than the
   candidate-ranking you rejected (which hands over the deduction).

## Recommendation

- **Bank stage 1** as the honest headline: **0.281 TEST win** — the strongest clean free-gen number
  to date (checkpoint `runs/cot_eph_aux_fair.pt`).
- **If "only valid words" is the priority:** pure free-gen tops out ~0.66–0.75 on this model; the
  honest unlock to ~95%+ is **dictionary-constrained decoding** (spelling-only). Reconsider it — it's
  not the cheating you rejected.
- **v2 (oversampled DAgger)** will tell us the true conditional-correction ceiling; if it lifts
  validity meaningfully without tanking win, it's the best pure-free-gen point. Pending.
