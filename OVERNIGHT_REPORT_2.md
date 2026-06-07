# Overnight push #2 — honest envelope, multi-experiment portfolio (2026-06-06 → 07)

**Goal:** push the honest **pure free-gen held-out** number past stage-1 (0.281) by attacking the
**deduction** wall from several angles. Honest throughout: train-only secrets, dictionary pools,
constrained decoding only ever as a *training wheel* (never at inference), eval = pure free-gen on the
disjoint TEST. Every experiment starts from the safe stage-1 base and uses best-by-quality
checkpointing, so **nothing can regress below stage-1.**

## Baseline (what we're trying to beat)
- **stage-1 (`cot_eph_aux_fair.pt`): free-gen TEST 0.281 / valid 0.662**; VAL 0.344; TRAIN 0.515.
- (reference, *aided*: same weights + spelling-constrained decode = 0.436 / 1.000 — not the headline.)
- Confirmed dead-ends: DAgger (null), distillation (trades win for validity), GRPO (flat 0.344→0.333,
  and collapses without heavy anchoring — RL can only re-weight, can't add capability).

## Portfolio (sequential — one GPU; chained via monitors; report updated after each)
| # | experiment | lever | status |
|---|---|---|---|
| 1 | info-gain XIT, **constrained** rollouts (Design B) | deduction, training-wheel | **running** |
| 2 | info-gain XIT, **free-gen** rollouts (pure STaR) | deduction, no wheel | queued |
| 3 | info-gain **DPO** (prefer higher-info-gain guesses) | deduction via preference (DPO worked here) | queued |
| 4 | diagnostics: constrained-decode + best-of-N on the best ckpt | *aided* upper bounds (labeled) | queued |
| 5 | combine best two (e.g. XIT → DPO polish) | stacking | if time |

*(98M scale deliberately skipped: the held-out wall is fundamental — more params won't break it — and
it would eat the night for a marginal, off-thesis gain. Better spent on diverse deduction levers.)*

## Results (free-gen held-out, updated as they land)
| experiment | TEST win | TEST valid | vs stage-1 | note |
|---|---|---|---|---|
| stage-1 baseline | 0.281 | 0.662 | — | the bar |
| 1 — info-gain XIT (constrained) | 0.281 | 0.662 | **flat (null)** | VAL dropped to 0.27–0.29 over rounds → best-by-quality kept baseline |
| 2 — info-gain XIT (free / STaR) | 0.281 | 0.662 | **flat (null)** | VAL 0.24–0.28 over rounds → kept baseline |
| 3 — DPO commit-sharpening (fair base) | … | … | … | running |

## Running conclusion
- **Exp 1 (info-gain XIT, constrained / Design B) = null.** The dense info-gain signal *did* select
  good-deduction turns (mean 1.6 nats/turn kept), but training the free-gen model on the constrained
  model's choices **did not transfer** to pure free-gen held-out (VAL even dipped). Same lesson as
  distillation: a training-wheel signal doesn't cross the train→inference gap for free-gen. The
  deduction wall holds against imitation of constrained play.
- **Exp 2 (info-gain XIT, free-gen / pure STaR) = null** too (0.281). Self-improvement without the
  training wheel also doesn't move it. Two independent expert-iteration variants now agree with GRPO
  (flat), DAgger (null), and distillation (trade): **every method that only re-weights or imitates
  existing behavior is stuck at ~0.28** — the wall is the model's inability to reliably produce+deduce
  unseen words in free-gen, which these methods cannot inject.
