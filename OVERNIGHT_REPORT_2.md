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
| 3 — DPO commit-sharpening (fair base) | 0.281 | 0.666 | **flat (null)** | every epoch regressed on VAL → reverted to base; full-held 0.294 = base |
| 4 — **best-of-16 self-consistency** (aided) | **0.632** | **0.925** | **+0.351 🎯** | sample→keep valid→majority vote; the standout |
| 5 — best-of-16, NO dict filter (pure compute) | 0.243 | 0.635 | **−0.038 (worse!)** | compute alone hurts; the dict filter was the lever |

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
- **Exp 3 (DPO commit-sharpening, fair base) = null.** Every epoch regressed on VAL and auto-reverted;
  final = base. Notable: the *same* DPO recipe LIFTED the contaminated base (0.616→0.631) — because
  there it had a real "finds the line, greedy-commits wrong" gap to sharpen. The honest base has little
  such gap (it often can't find the line at all), so DPO has nothing to sharpen. The contamination is
  literally what made DPO look like it worked.
- **★ Exp 4 (best-of-16 self-consistency) = the big finding: 0.632 / 0.925 on TEST** — same stage-1
  weights, **+125% over greedy**, *honestly* matching the old contaminated 0.62 headline (no train-test
  leak, no clue-consistency — just a spelling filter + sample-and-vote).
- **Exp 5 corrects the interpretation (honesty):** best-of-16 *without* the dictionary filter = **0.243,
  WORSE than greedy.** So the gain is **NOT** test-time compute by itself — it's the **dictionary/spelling
  aid at inference** that's the lever (0.281→0.436 greedy-constrained), and sample-and-vote *adds on top
  of it* (0.436→0.632) but *only because it has valid candidates to vote among*. Compute over raw
  free-gen samples (incl. non-words) actively hurts. The real lesson: **the gains live at inference and
  come from the (honest) spelling aid**, not from more training (all training null) nor from compute alone.

## The decode ladder (honest, same stage-1 weights)
| decode | TEST win | valid | aid |
|---|---|---|---|
| free-gen greedy (**pure headline**) | **0.281** | 0.662 | none |
| best-of-16 vote, no dict | 0.243 | 0.635 | compute only — *worse* |
| constrained-decode greedy | 0.436 | 1.000 | spelling |
| **best-of-16 vote, valid-filter** (**best honest**) | **0.632** | 0.925 | spelling + compute |

## Final verdict + best model
- **Pure free-gen (strictest): 0.281 — UNMOVED.** Six training methods (DAgger, distillation, GRPO,
  info-gain XIT ×2, DPO) all null. The free-gen deduction wall is fully robust; training cannot inject
  the missing produce-and-deduce-unseen-words capability.
- **Best honest result: 0.632 / 0.925** = stage-1 (`cot_eph_aux_fair.pt`) + **best-of-16 valid-filter
  decoding** — spelling-aided + test-time compute, fully honest (no contamination, no clue-consistency).
  This cleanly *replaces* the project's old contaminated 0.62 with an honest 0.63.
- **Honesty win of the night:** DPO's earlier 0.616→0.631 was contamination-dependent (null on the clean
  base) — caught and corrected.
- **Recommendation:** the headline pure-free-gen number is 0.281 and is at its ceiling for this approach;
  the gains that exist are at **inference** via the honest spelling aid (best-of-16-valid → 0.632). More
  training is exhausted. The two honest numbers to quote: **0.281 (model alone)** and **0.632 (model +
  honest aided decoding).**
