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
| scale | base (12M) SFT | … | … | … | running |
| scale | xl (99M) SFT | … | … | … | queued |
| data | full-dict-secrets SFT | … | … | … | queued |
| search | beam over trie | … | — | … | queued |
| search | best-of-N on xl | — | … | … | queued |
| framing | deployed + best-of-N | … | … | … | queued |

## Running conclusion
*(updated through the day; final synthesis + best model at the end.)*
