# Build Plan: Wordle SLM

**Status:** Draft v2 (hardened via adversarial review)
**Owner:** Pedram
**Last updated:** 2026-06-02
**Specs:** [`PRD.md`](../../PRD.md) · [`docs/design/wordle-slm.md`](./wordle-slm.md)

Ordered by the **critical path, earliest-feedback first** — the riskiest end-to-end behavior is
exercised before the components are fully built out (a *tracer bullet*). Steps are grouped into
**waves**; **∥** marks steps with no dependency on each other. For a **solo developer ∥ means
order-flexible** (do them in whatever order keeps you unblocked), *not* concurrent execution.

Legend: **∥** order-flexible group · **→ dep:** depends on · **DoD** done when · ★ milestone.
Every step ends green: `ruff format --check && ruff check && pytest`. Tests precede code.

---

## Guiding principle (why this order)

The #1 goal is **learning RL**, and the user already knows PyTorch — so we **front-load the RL
feedback loop**. A thin end-to-end slice (random model → one rollout → one reward → one GRPO
update, in TensorBoard) runs in **Wave 3**, long before the full components exist. This (a)
de-risks the GRPO mechanics + MPS hazards early, (b) satisfies the PRD's committed mitigation
to "validate the training loop on a trivial task first / early" (`PRD.md:230, 233`), and (c)
gets the learner watching RL dynamics on day ~3, not at the final step.

**The loop is validated in two layers** (you cannot skip the order):
- **Layer 1 — mechanics (random model OK):** does a reward become a gradient become a weight
  update on MPS, with the right shapes and masks? A random model is fine — we assert *plumbing*,
  not learning. (Step `V`.)
- **Layer 2 — learning (must be warm-started):** does reward actually rise? This needs a
  post-SFT model, because a random model almost never emits a valid word → all-loss groups →
  zero within-group variance → the §6.2 zero-variance filter eats the gradient. (Steps `W`, `X`.)

---

## Dependency graph (high level)

```
                    ┌──────────────────── S0 scaffold ────────────────────┐
        ┌─────┬─────┼───────┬──────────┬──────────┐                       │
        ▼     ▼     ▼       ▼          ▼          ▼                       ▼
  W1:  A     B     C       D          K                              (CI workflow)
     data  engine tok    telem    config-resolve
        │     │     │
        ├──┬──┴──┬──┴───┬───────┬────────┐
        ▼  ▼     ▼      ▼       ▼        ▼
  W2:  E Game  F serialize  G model  H reward  I curric+replay  Y play_game()
        │      │           │        │                          │
        └───┬──┴─────┬─────┴───┬────┴──────────────────────────┘
            ▼         ▼         ▼
  W3:  J baselines+teachers   V★ TRACER BULLET (random model, Layer-1 mechanics)
            │
        ┌───┴────────┬──────────────┐
        ▼            ▼              ▼
  W4:  L★ Phase0+budget   M SFT-data(70/30)   W SFT-overfit gate
                              │                  │
                              ▼                  ▼ (gates N)
  W5:                      N SFT train + reloadable checkpoint
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼            (P1 DoD)
  W6:        O rollout-bench(→Y)   P★ Phase1 eval(→Y)
                  └─────┬─────┘
                        ▼
  W7:                Q GRPO trainer (extends V; →N,H,G,F,Y)
                        │
              ┌─────────┼──────────┬───────────┬──────────┐
              ▼         ▼          ▼           ▼          ▼
  W8:  X Layer-2 overfit   R budget   S eval+gap   T echo-telemetry  (I wires in)
       (one secret, warm)   loop       +selection
              └─────────┴────┬─────┴───────────┘
                             ▼
  W9:                   U★ first RL run (measurable gate)
```

---

## Wave 0 — Foundation (blocks everything)

| Step | What | DoD |
| --- | --- | --- |
| **S0** | `uv` project (`pyproject.toml` + `uv.lock`: torch, numpy, tensorboard, pytest, ruff). `src/wordle_slm/` skeleton (§9 modules) + `tests/`. ruff + pytest config. `RunConfig`/sub-config dataclass skeletons. `cli.py` stubs (`phase0/sft/rl/eval/play`). | `uv sync` works; `pytest` runs; `ruff check` clean; CLI `--help` works. |

> Optional ∥: GitHub Actions running the PR gate (ruff + pytest).

## Wave 1 — Independent foundations  ∥ (all → dep: S0)

| Step | What | DoD / tests |
| --- | --- | --- |
| **A** | **Word data** (`data/`): source & commit 2,315 answers + 12,972 valid guesses; `load_*()`, seeded 80/20 split (held-out **immutable**), **fixed train-probe subset** (pinned, for the gen-gap), `is_valid()`. | split-stability; held-out-immutability; counts 2,315/12,972; **leakage asserts: `train ∩ heldout == ∅` as secrets · no dupes within either list · `answers ⊆ valid_guesses`**; **held-out words ARE valid guesses but the secret-sampler never draws them** (test); train-probe pinned + disjoint from held-out. |
| **B** | **Engine scoring** (`engine/`): `Color` enum + `score()` two-pass duplicate-safe. | exact-tuple tests: `ALLEY→EARLY`, `SPEED→ERASE`, `LLAMA→BALSA→[Y,X,Y,X,G]`, all-green, all-gray. |
| **C** | **Tokenizer** (`model/`): ~34-token char vocab, `encode`/`decode`. | round-trip; stable special-token ids. |
| **D** | **Telemetry** (`telemetry/`): TensorBoard scalar writer + JSON run log (config/seed/git-SHA) + per-game transcript logger. | writes scalar + JSON to temp dir in a test. |
| **K** | **Config resolution** (`config/`): load a named **preset**, apply **CLI overrides** (`--grpo.group-size 16`), produce one resolved `RunConfig`, log it. | preset-load; override mutates exactly the targeted field; unknown key errors loudly; resolved config round-trips. |

## Wave 2 — Build on foundations  ∥

| Step | What | → dep | DoD / tests |
| --- | --- | --- | --- |
| **E** | **Game** (`engine/`): 6-guess loop, win/lose/ongoing, validity, invalid-consumes-turn. | B, A | game-flow; invalid consumes a turn; win/lose detection. |
| **F** | **Board serialization** (§5.2 grammar): state→tokens, parse back; `<GUESS>` only on current turn. | C, B | round-trip; matches §5.2 example exactly. |
| **G** | **Model** (`model/`): decoder-only transformer (~6.3M), forward + `generate` (length-masked 5 letters), MPS device. | C | param-count 5–10M; forward shape; generate emits exactly 5 letters. |
| **H** | **Reward v1** (`rl/`): §6.4 `green_known`/`min_count` rule + all terms + dominance asserts (`p_invalid>b`, `q>b`, **win > max farming**). | B | first-green-once, yellow→green `a+b`, dup single `b`, re-confirm 0, invalid/clue dominance, win-dominance. |
| **I** | **Curriculum + replay** (`rl/`): tiered sets + hard-word FIFO replay (10% draw). | A | promotion logic; replay sampling probability. |
| **Y** | **Shared `play_game()`** (`rl/rollout.py`): `play_game(model, secret, *, sample\|greedy) -> Transcript` driving prompt→generate-5→score→append via E+F+B. **The trainer, all eval/benchmark steps, and the tracer bullet call this — no private copies.** | E, F, B, G | one game round-trips & matches §5.2; sample+greedy paths; ≥2 callers in tests (no duplicated rollout code). |

## Wave 3 — Baselines + first end-to-end signal  ∥

| Step | What | → dep | DoD / tests |
| --- | --- | --- | --- |
| **J** | **Baselines + teachers** (`baselines/`, `teacher/`): (1) random valid-word guesser (floor); (2) feedback-consistent guesser (weak teacher); (3) **near-optimal information-maximizing teacher** (strong, §5.4 — opens strong, then maximizes expected partition over remaining answers). | E, A, Y | plays full games; consistent guesser never violates a clue; **near-optimal pick shrinks the candidate set more than the consistent guesser on a fixed board; solves a known easy secret in ≤4.** |
| **V ★** | **Tracer bullet (Layer-1 mechanics, random model):** wire one thin end-to-end GRPO slice — random-init model → one rollout (via `Y`) → one reward (`H`) → one GRPO update (mean-center advantage, clipped surrogate, k3 KL, zero-variance filter) on one small group → scalars to TensorBoard. Hard-coded secret; NO curriculum/eval/budget/SFT. This minimal loop is the seed `Q` later extends. | E, G, H, F, D, Y | one update runs on MPS without shape/NaN errors; **gradient is non-zero on guess-letter tokens and zero on context/feedback tokens** (test); advantage non-zero on ≥1 group (uses ≥2 secrets or shaped reward so not all-filtered); KL≥0; `π_θ_old` ≠ `π_ref`; reward+L_clip+KL+entropy in TensorBoard. **Does NOT assert reward rises** (random model). Satisfies `PRD.md:230,233`. |

## Wave 4 — Phase-0 deliverable + SFT data + SFT-overfit gate  ∥

| Step | What | → dep | DoD / tests |
| --- | --- | --- | --- |
| **L ★** | **Phase-0 run + budget gate:** measure random floor (~0.26% headline + ~0.05% logged) and consistent yardstick (~96–99%) over held-out; engine games/sec. **Plug games/sec into the §6.6 formula and assert a full SFT(~15m)+RL(~45m) cycle is plausibly ≤ ~1 hr at the planned G, or record exactly what to shrink (model/train, never held-out) and the G that fits.** | J, D | **Phase 0 DoD (§4.6):** floor + yardstick + games/sec logged; **§6.6 back-of-envelope #updates computed, or the documented shrink decision.** |
| **M** | **SFT data generation** (`teacher/`): mix teachers (~70% consistent / ~30% **near-optimal from J**) + varied openers → transcripts in §5.2 format (target = guess letters). | J, F, A, E | sample transcript parses; **blend ratio test asserts ~70/30 across both teacher types**; openers vary. |
| **W** | **SFT overfit sanity (gate before N):** train `G` on a **single** teacher transcript (masked guess-letter loss) until CE → ~0. | G, F | loss on the one transcript → ≈0 within N steps; greedy generation reproduces the memorized guess; AdamW `exp_avg_sq` smoke-test passes (§12). |

## Wave 5 — Head start

| Step | What | → dep | DoD / tests |
| --- | --- | --- | --- |
| **N** | **SFT training** (`sft/`): masked-loss imitation, AdamW, MPS smoke test, outcome-based stop (cap 15 min). **Saves a reloadable checkpoint `{model, optim, step, rng, config}`** so `Q` can init the policy + freeze `π_ref`. *(gated by W.)* | G, M, F | loss-mask test; AdamW-updates smoke; trains a step on MPS; **checkpoint save→load round-trip restores model+optim+rng+config; loaded model gives identical greedy guesses.** |

## Wave 6 — Phase-1 deliverable + rollout benchmark  ∥

| Step | What | → dep | DoD / tests |
| --- | --- | --- | --- |
| **O** | **Model-rollout micro-benchmark** (§4.5): games/sec + peak MPS memory at G∈{4,8,16} via `Y`; pin G. | Y, G (N for realism) | benchmark + chosen G logged. |
| **P ★** | **Phase-1 eval** (`eval/`): valid-word rate ≥95%; clue-respect ≥80% green-retention. | N, A, Y | **Phase 1 DoD (§5.6).** |

## Wave 7 — RL core

| Step | What | → dep | DoD / tests |
| --- | --- | --- | --- |
| **Q** | **GRPO trainer** (`rl/`, §6.3) — **extends V's minimal loop** to full fidelity: sampling rollouts via `Y`; group on same secret; advantage (mean-center, **no ÷std**, **filter zero-variance**); clipped surrogate + `π_θ_old` snapshot + frozen `π_ref` (from N) + k3 KL; grad clip; LR warmup; inner epochs K. | N, H, G, F, Y | zero-variance-filter test; one update on MPS; KL/ratio shapes correct. |

## Wave 8 — Layer-2 overfit gate + RL integration  ∥ (all → dep: Q)

| Step | What | → dep | DoD / tests |
| --- | --- | --- | --- |
| **X** | **GRPO overfit-one-secret (Layer-2 gate before U):** run `Q` on **one fixed secret**, warm-started from N; confirm the policy learns to solve that word. | Q, H, G, F, Y | mean group reward strictly rises; model solves the secret in ≤6 by end; KL/entropy/grad-norm finite (no collapse). |
| **R** | **Budget-sized rollout loop** (§6.6): `secrets_per_update`, throughput formula (subtract `eval_overhead`), pin G from O. | Q, O | #updates estimate (net of eval) computed; loop respects ~45-min budget. |
| **S** | **Eval + gap + selection** (§6.7): two-tier held-out eval, gap vs **fixed train probe set** (from A), best-checkpoint-by-held-out (reuses N's checkpoint format). | Q, A, Y | gap on fixed probe; best-checkpoint saved + resumable. |
| **T** | **Echo-trap telemetry** (§8): entropy, reward/advantage variance, grad-norm, filtered-group fraction. | Q, D | scalars logged during a run. |
| | *(I curriculum+replay wires into the trainer.)* | Q, I | tiers advance; replay feeds secrets. |

## Wave 9 — First RL milestone

| Step | What | → dep | DoD |
| --- | --- | --- | --- |
| **U ★** | **First RL run:** full SFT→GRPO cycle within ~1 hr; held-out **learning curve clearing the measurable gate** — (i) win rate ≥ floor + **MARGIN (≥10 pts)** at ≥ **K=3** consecutive full-eval points; (ii) curve **rising**; (iii) **gap < GAP_MAX (15 pts)** at the selected checkpoint. *(MARGIN/K/GAP_MAX are H-tagged, tune in §13.)* | R, S, T, I, X | all three sub-conditions logged + asserted; ★ Phase-2 milestone; results → lab notebook (PRD §15). |

---

## Phase 3 (deferred levers — only if learning stalls)

Information-gain reward · turn-level advantage · constrained-candidate decoding at eval ·
entropy floor. Each = one hypothesis → result entry in the lab notebook.

## Critical path (longest dependency chain)

By **edge count**, the binding chain runs through the **SFT-data branch**, not the model branch:

`S0 → A → E → J → M → N → Q → X → {R,S,T} → U`

(data → game → teachers → SFT transcripts → SFT train → GRPO trainer → single-secret overfit
gate → integration → first run). The model branch `S0 → C → G → N` reaches `N` in fewer edges,
so `C`/`G` finish with slack alongside `A→E→J→M`. The tracer bullet `V` and overfit gates `W`/`X`
sit on the **feedback-latency** path (cheap but ordering-critical), not the throughput path.

**Caveat:** this is edge-count (unit effort per step). Weighted by wall-clock, the path likely
flips toward `G/N/Q/U` once the 15-min SFT and 45-min RL caps dominate. Add effort estimates
before treating this as a schedule.

## Notes

- **Spec sync (pending):** the spec's `§15` build order, `§11` tests (add overfit + leakage
  tests), and `§4.1/§5.5/§6` should mirror the tracer-bullet/overfit/leakage additions and the
  warm-start rule for the smoke test. The detailed plan here supersedes the brief `§15` list.
- `play` CLI (live game) is Phase-4, deferred — stubbed in S0 only.
