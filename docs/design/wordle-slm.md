# Detailed Spec: Wordle SLM

**Status:** Draft v2 (hardened via adversarial review)
**Owner:** Pedram
**Last updated:** 2026-06-02
**Companion doc:** [`PRD.md`](../../PRD.md) — the product-level *what & why*. This doc is the *how*.

This spec covers **Phase 0 and Phase 1 in depth** and **details Phase 2 (RL)**, per the
agreed scope. Phases 3–4 are sketched at the end. Routine hyperparameters are collected
in the **Hyperparameters** table (§13), each tagged **Invariant / Hypothesis / Routine**.

---

## 1. Goal recap (one paragraph)

Build a ~1–5M-parameter decoder-only transformer **from scratch** and teach it Wordle in
two stages: an **imitation head start** (supervised), then **reinforcement learning (GRPO)**.
The project's primary goal is **learning how SLMs + RL work**; success is the model learning
the *strategy* — using green/yellow/gray feedback to narrow the answer — measured on
**held-out words it never trained on**. **Pass/fail gate:** win rate on held-out clearly and
steadily above the random floor, with a small practiced-vs-held-out gap and an explainable
learning curve. **Aspirational stretch target: ≥80% held-out win rate** (see §6.7 for why
this is a stretch, not the gate). Everything runs locally on the M5 Max within a **~1-hour
training cycle**.

## 2. Decisions captured from the interview

| Area | Decision |
| --- | --- |
| Architecture | Decoder-only transformer (GPT-style) |
| Output unit | Character-level (letters + a few special tokens) |
| Model size | 1–5M params (default ~3.2M) |
| Board representation | Plain text (tokenized) |
| Guess validity | Free generation, then validate; invalid = explicit penalty **and** wastes a turn |
| Game interaction | Per-turn stepping (env steps externally) |
| Info given to model | Board history only (no candidate list) |
| Action selection | Sample during training, greedy at eval |
| RL algorithm | **GRPO** (full clipped surrogate, from day 1) |
| Reward | win bonus (faster = more) · per-guess letter progress · invalid-word penalty · anti-stall/clue penalty · small step cost · small negative on loss |
| Deferred reward parts | information-gain reward (Phase 3 tuning) |
| Head-start teacher | Mix of strategies (default: ~70% random-consistent / ~30% near-optimal) |
| Openers | Varied (sampled from strong starters) |
| Head-start amount | Outcome-based (train until Phase-1 bar) |
| Imitation target | Loss only on the guess letters |
| Word lists | Classic original Wordle: 2,315 answers + 12,972 valid guesses |
| Split | 80/20 random, seeded; held-out = test secrets only, **immutable** |
| Curriculum | Performance-triggered widening + hard-word replay queue |
| Game mode | Normal mode |
| Budget split | SFT until bar (cap ~15 min), RL gets the rest (~45 min) |
| Evaluation | Cheap subsampled curve + full held-out for checkpoint selection, + generalization gap |
| Model selection | Best checkpoint by held-out win rate |
| Reproducibility | One fixed seed; multi-seed optional (note: MPS is not fully deterministic) |
| Package manager | uv |
| Config | Typed dataclasses + CLI overrides |
| Experiment tracking | TensorBoard + structured logs |
| AGENTS.md | Update to the real Python/PyTorch stack |

## 3. System overview

Five components, matching the PRD's "game / player / head start / coach / scoreboard":

```
                ┌─────────────┐   secret word    ┌──────────────┐
                │  data/      │ ───────────────▶ │  engine/     │
                │ word lists  │                  │ game+scoring │
                │ train/held  │                  └──────┬───────┘
                └─────────────┘                board ▲  │ feedback
                                                      │  ▼
   teacher/ (heuristics) ──┐                   ┌──────┴───────┐
                           ├─ example games ──▶│  model/      │ guess (5 letters)
   baselines/ (random,     │     (SFT data)    │ transformer  │
   consistent)  ───────────┘                   │ + tokenizer  │
                                               └──────┬───────┘
   sft/  ── head start ──────────────────▶ trains the model
   rl/   ── GRPO coach + reward ─────────▶ trains the model
   eval/ ── held-out win rate + gap ─────▶ ┐
   telemetry/ ── TensorBoard + JSON ───────┴▶ scoreboard
```

**Flow per game (per-turn stepping):** engine picks a secret → loop up to 6 times: build
the board-text prompt → model produces a 5-letter guess → engine validates & scores →
append feedback → stop on win or 6 guesses.

---

## 4. Phase 0 — The playground (full spec)

Goal: a correct, fast, well-instrumented environment and the baseline numbers. **No model yet.**

### 4.1 Word data (`data/`)
- **Answers:** the classic 2,315-word Wordle answer list — the pool secret words are drawn from.
- **Valid guesses:** the classic 12,972-word list (superset including the answers). Membership
  = a "valid word" for the validity check.
- **Split:** shuffle the answer list with a fixed seed; **80% (~1,852) `train`**, **20% (~463)
  `held-out`**. Written to disk and version-pinned so the held-out set is identical across every
  run. **Held-out words are excluded only as secrets during training** — the model may still
  *guess* them. The held-out set is **immutable**: the §4.5 "shrink to fit budget" rule may
  shrink the *train* set or the model, **never the held-out set**.
- API: `load_answers()`, `load_valid_guesses()`, `split(seed) -> (train, heldout)`,
  `is_valid(word) -> bool` (O(1) set lookup).

### 4.2 Engine (`engine/`)
- `score(guess, answer) -> tuple[Color; 5]`, `Color ∈ {GREEN, YELLOW, GRAY}`. **Correct
  duplicate-letter handling is mandatory** — two-pass:
  1. Mark GREEN where `guess[i] == answer[i]`; build a per-letter count pool of the answer's
     remaining (non-green) letters.
  2. For each non-green position, mark YELLOW iff that letter still has a remaining count
     (decrement on use), else GRAY.
- `Game`: holds the secret + list of (guess, feedback) turns; `guess(word) -> (feedback, status)`,
  `status ∈ {WIN, LOSE, ONGOING}`; enforces the 6-guess cap; records invalid attempts (which
  still consume a turn — §6.4). Pure, deterministic, fully unit-tested (§11).

### 4.3 Baselines (`baselines/`)
Both also seed the head-start teachers (§5.4).
- **Random valid-word guesser** (the *floor*): each turn pick a uniformly random word from the
  12,972 valid list, ignoring feedback. **We report the answer-pool floor (~0.26%) as the
  headline** to match the PRD; the full-valid-list floor (~0.05%) is also logged. Either way this
  is a **sanity check, not the success bar.**
- **Feedback-consistent guesser** (the *honest yardstick*): open with a fixed strong word, then
  each turn pick a random word from the valid list **still consistent with all clues**. Expected
  **~96–99%**. A stretch reference, not the pass bar.

### 4.4 Telemetry (Phase 0)
For each baseline over the full held-out set: win rate, average guesses (wins), guess-count
distribution, per-game logs. Write the measured **random floor** and **consistent yardstick**
numbers into the run log; these calibrate §6.7 / PRD §6.

### 4.5 Speed & memory budget (the binding constraint)
Phase 0 measures and records, on this Mac:
- **engine games/sec** (pure rollout, no model), and
- a **model-rollout micro-benchmark** (filled once the Phase-1 model exists): **games/sec and
  peak MPS memory at group size G ∈ {4, 8, 16}.** Group size is the dominant memory cost
  (each parallel rollout needs its own KV cache); a 1–5M model's cache is tiny vs. a multi-B
  model, so 128 GB likely absorbs G=16 — but we **measure, not assume** (charbull saw large
  memory jumps from 2→4 generations on a 4B model). G is then pinned to the budget.

These feed §13 sizing so a full cycle fits **~1 hour**. Rule: if projections exceed the budget,
**shrink the model or the train set (never the held-out set), don't wait.**

### 4.6 Phase 0 — Definition of done
- Engine `score()` passes the full duplicate-letter test suite (§11).
- Random floor and consistent-yardstick win rates measured over the held-out set and logged.
- Engine games/sec measured; the model-rollout/memory micro-benchmark harness exists and a
  back-of-envelope (§6 budget formula) shows a full cycle is plausibly ≤ ~1 hour (or we know
  what to shrink, and at which G).

---

## 5. Phase 1 — The model & head start (full spec)

Goal: a 1–5M decoder-only transformer that, after an imitation warm-up, **reliably emits valid
words and respects obvious clues.** No RL yet.

### 5.1 Tokenizer (`model/`)
Custom character-level vocabulary (~34 tokens): 26 letters `a`–`z`; `<BOS>`, `<EOS>`, `<PAD>`;
`<SEP>` (turn separator); `<GUESS>` (marker after which the model emits exactly 5 letters);
`<green>`, `<yellow>`, `<gray>` (per-position feedback).

### 5.2 Board text format (the prompt) — byte-exact grammar
Each **completed** turn = 5 guess letters + 5 position-aligned feedback tokens + `<SEP>`.
`<GUESS>` marks **only the turn currently being generated** — it is never stored in history.

```
sequence       = <BOS> , { completed_turn } , current_turn ;
completed_turn = guess_letters , feedback , <SEP> ;
current_turn   = <GUESS> , guess_letters , <EOS> ;
guess_letters  = letter × 5 ;                 (* a–z *)
feedback       = fb × 5 ;     fb = <green> | <yellow> | <gray> ;
```

Both producers (the SFT data generator and the env prompt-builder) must follow this exactly.
After the env scores guess *k*, the builder rewrites turn *k* as a `completed_turn` (appends its
5 feedback tokens + `<SEP>`) and opens a fresh `<GUESS>`.

Example — turn 1 was `CRANE` (secret `NIGHT`: C,R,A,E gray; N yellow), now generating turn 2:
```
<BOS> c r a n e <gray> <gray> <gray> <yellow> <gray> <SEP> <GUESS> ▮ ▮ ▮ ▮ ▮ <EOS>
                                                                    └ model generates these
```
- **Context length 128** (a full 6-turn game is ~66 tokens + framing). Generation after `<GUESS>`
  is length-masked to exactly 5 letter tokens; the *word's* validity is checked by the engine
  afterward (free generation + validate) — the model can still spell a non-word.

### 5.3 Model (`model/`)
Decoder-only transformer (pre-norm, causal attention, learned position embeddings, weight-tied
embeddings). Target **1–5M params**; default ~3.2M (§13). PyTorch on the MPS (Metal) backend;
CPU fallback for unsupported ops. **MPS caveat:** see §12.

### 5.4 Head-start data (`teacher/`)
- **Mix of teachers** on the **train** answers only: *random-consistent* (weak, default ~70%) +
  *near-optimal* information-maximizing (strong, default ~30%). Blend tunable (§13).
- **Varied openers:** each game's first guess sampled from a small set of strong starters.
- Output: transcripts in the §5.2 format, the teacher's chosen guess as the target after `<GUESS>`.

### 5.5 Head-start training (`sft/`)
- Objective: next-token cross-entropy **masked to the guess-letter positions only** (the 5 letters
  after each `<GUESS>`); board + feedback are context, not scored. Imitate moves, not feedback.
- AdamW; defaults in §13. **Smoke test required** for the MPS AdamW bug (§12): assert optimizer
  state (`exp_avg_sq`) actually updates.
- **Outcome-based stop, capped:** train until the Phase-1 bar (§5.6) or **~15 min**, whichever
  first. If the cap hits before the bar, shrink the model or train set (never held-out).

### 5.6 Phase 1 — Definition of done
- **Valid-word rate ≥ 95%** over 1,000 sampled generations from random board states.
- **Clue-respecting bar (provisional, revise on first measurement):** over ≥500 sampled board
  states containing ≥1 known green, the next greedy guess **keeps every known green in place
  ≥ 80% of the time**. SFT stops when **both** bars hold or the 15-min cap hits.

---

## 6. Phase 2 — RL with GRPO (full clipped surrogate)

### 6.1 Policy, rollout, reference
- The policy **is** the model. A **rollout** = one full game (≤6 guesses) played by **sampling**
  from the policy (temperature §13).
- **`π_ref`** = the frozen Phase-1 SFT model (for the KL term). **Never updated.**
- **`π_θ_old`** = the policy weights snapshotted when the current rollout batch was sampled.
  Refreshed every rollout batch. (`π_ref` and `π_θ_old` are distinct policies.)

### 6.2 GRPO grouping & advantage
- A **group** = **G rollouts on the same secret word** (controls for word difficulty).
- **Trajectory return** `R_i` = sum of all shaped rewards in game *i* (§6.4).
- **Advantage** = group mean-centered, **no std-normalization** (avoids the Dr. GRPO
  difficulty-scaling bias): `A_i = R_i − mean(R_group)`. Broadcast to every guess-letter token
  in trajectory *i* (feedback tokens never receive gradient).
- **Zero-variance groups** (all G rollouts get identical reward — common on easy tiers, all-win,
  or all-loss) are **filtered out** (they carry no learning signal and destabilize training — StarPO-S).

### 6.3 GRPO loss (clipped surrogate, k3 KL)
On guess-letter tokens only. Let `r_t(θ) = π_θ(o_t|s_t) / π_θ_old(o_t|s_t)`:

```
L_clip = − (1/Σ|o_i|) Σ_i Σ_t  min( r_t · A_i ,  clip(r_t, 1−ε, 1+ε) · A_i )
```

KL penalty to the SFT reference via the **k3 unbiased estimator** (non-negative, low-variance):

```
KL_t = (π_ref/π_θ) − log(π_ref/π_θ) − 1
L_KL = β · (1/Σ|o_i|) Σ_i Σ_t KL_t
```

**Total:** `L = L_clip + L_KL`. No value network (advantage is group-relative). Discount γ = 1
(episodes ≤ 6 steps; shaped rewards give intermediate signal). **Inner epochs K** = gradient
steps per rollout batch before refreshing `π_θ_old` (default K=1; raise only if rollout cost
dominates — K>1 is exactly why the ratio + clip are required). **Gradient clipping** `max_grad_norm`
and **LR warmup** per §13.

### 6.4 Reward function (concrete v1)
Per game, summed over guesses *t = 1…T* (T ≤ 6).

**Letter progress** `a·(new_greens_t) + b·(new_yellows_t)`, against knowledge state carried
across the game (not per-turn feedback). Maintain, from empty: `green_known` (set of positions
known green) and `min_count` (letter → known minimum count in the answer).
1. `new_greens_t` = positions `i` that are GREEN this turn and `i ∉ green_known`; then add them.
2. For each letter `L` in the guess, `obs_L` = count of non-GRAY (green+yellow) positions showing
   `L`; `new_yellows_t += max(0, obs_L − min_count[L])`; then `min_count[L] = max(min_count[L], obs_L)`.

This gives: a position pays its green bonus **once**; a yellow→green upgrade pays `b` then `a`,
never double; a duplicate letter is credited only when it raises the known min-count;
re-confirming known constraints pays **0** (no farming).

**Other terms:**
- **Invalid-word penalty:** guess ∉ valid list → `− p_invalid`, no letter progress, **consumes the
  turn** (no retry).
- **Anti-stall / clue penalty (Phase 2, not deferred):** a guess that violates a confirmed
  constraint (drops a known green, reuses a known-gray letter) → `− q`, with **`q > b`** so any
  honest progress beats stalling. (GRPO will find the cheap −c stall before the +win otherwise —
  this is the loophole charbull hit.)
- **Step cost:** `− c` each guess.
- **Terminal:** win at guess *t* → `+ (W_base + W_speed·(6 − t))`; loss → `− L`.

**Dominance (must hold with the §13 coefficients):** `p_invalid > b` and `q > b`, and max
achievable progress in a slow game (`≈ 5·a + few·b`) < `W_base` so a win always dominates farming.
Verify in telemetry (§8); tune in Phase 3. **Information-gain reward is deferred to Phase 3.**

### 6.5 Curriculum + hard-word replay
- **Tiers** over the **train** answers: `200 → 500 → 1,000 → full (~1,852)`. Promote when win rate
  on the current tier crosses a threshold (§13).
- **Hard-word replay queue** (andrewkho's trick toward ~99%): a fixed-capacity FIFO; on each
  training loss, push the secret; when sampling the next secret, draw from the queue with
  probability ~10%, else from the current tier. Single sampler, one branch.
- The model is always **evaluated on the full immutable held-out set**, regardless of tier.

### 6.6 Budget model (ties GRPO to the ~45-min RL window)
```
rollout_batch = secrets_per_update × G
#updates ≈ (rollout_throughput[games/sec] × 45min×60 − eval_overhead) / rollout_batch
eval_overhead = full_heldout_games(≈463) × (#updates / full_eval_cadence)
```
Defaults (§13): `secrets_per_update = 8` (→ 128 rollouts/update at G=16). **Two-tier eval** to
keep eval cheap: a **128-secret held-out subsample every ~25 updates** for the learning curve;
the **full ~463 held-out every ~200 updates** for checkpoint selection. All starting points —
re-derive from the Phase-0 games/sec number before committing.

### 6.7 Evaluation, selection, gate
- **Generalization gap** = (fixed train *probe* set win rate − held-out win rate), measured on a
  **fixed probe set** (not the moving curriculum tier) so it's comparable across runs; report it
  signed (negative early is normal and fine). A persistently large positive gap = memorization → fix.
- **Model selection:** keep the checkpoint with the **highest full-held-out win rate**; also save final.
- **Pass/fail gate:** on held-out, win rate **clearly and steadily above the random floor, rising,
  with a small gap.** **Feasibility note (sourced):** comparable LLM-GRPO Wordle agents reach
  ~low-double-digits–30% win rate; ~99% has only been achieved by a restricted-action MLP trained
  on ~20M games with curriculum + replay (andrewkho). **≥80% is therefore a stretch hypothesis,
  gated on the Phase-0 throughput budget supporting curriculum + replay at scale — not a baseline
  expectation.** Hitting the gate (not 80%) is what declares this round a win.

---

## 7. Phases 3–4 (sketch) + named upgrade levers

- **Phase 3 — tune & understand:** add the information-gain reward; tune coefficients; inspect
  failures and reward exploits; require ≥1 change shown with before/after held-out numbers.
  **Upgrade levers (try if learning stalls):** (a) **turn-level advantage** instead of trajectory-level
  (better multi-turn credit assignment); (b) **constrained decoding to valid-and-consistent
  candidates** at eval (decouples "can spell" from "can strategize" — makes the strategy directly
  measurable and ≥80% more reachable; deliberately not the default because free-generation + board-only
  were chosen to make the model learn both); (c) an **entropy floor** if the policy collapses.
- **Echo Trap watch:** multi-turn RL characteristically improves then collapses (reward-variance
  cliff, entropy drop, gradient spikes). Monitor entropy, advantage/reward variance, and grad-norm
  as first-class signals (§8); best-checkpoint selection (§6.7) is the safety net.
- **Phase 4 (optional):** a CLI that plays one live terminal game; a public write-up.
- **Required throughout:** the **lab notebook** — decisions, each experiment as *hypothesis → result*,
  what moved the needle (PRD §15).

## 8. Telemetry (cross-cutting)

Per AGENTS.md, **every event/decision/outcome is logged.**
- **TensorBoard scalars:** held-out win rate (full + subsample), train-probe win rate,
  generalization gap, avg guesses, legal-word rate, each reward component (mean), GRPO `L_clip`,
  KL value, **grad-norm**, **policy entropy**, **reward/advantage variance** (Echo-Trap detection),
  **fraction of groups filtered** (zero-variance), current curriculum tier, replay-queue size,
  games/sec, steps/sec, eval-overhead %.
- **Structured JSON run log:** resolved config, seed, git SHA, all eval points, selected checkpoint,
  per-game transcripts for a sampled subset (secret, guesses, feedback, reward breakdown) — so any
  result is diagnosable from logs alone.

## 9. Repo layout

```
pyproject.toml            # uv-managed; deps: torch, numpy, tensorboard, pytest, ...
uv.lock
src/wordle_slm/
  engine/  data/  baselines/  teacher/  model/  sft/  rl/  eval/  telemetry/  config/
  cli.py                  # entrypoints: phase0, sft, rl, eval, play
tests/                    # pytest; engine scoring + reward v1 first
docs/design/wordle-slm.md
```

## 10. Configuration

Typed **dataclasses** (`ModelConfig`, `TokenizerConfig`, `RewardConfig`, `SFTConfig`, `GRPOConfig`,
`CurriculumConfig`, `EvalConfig`, `RunConfig`). A run loads a **preset** + accepts **CLI overrides**
(`--grpo.group-size 16`). The fully-resolved config is logged with the run.

## 11. Testing (pytest)

- **Engine scoring (priority)** — exact expected tuples, duplicate letters first:
  - `ALLEY → EARLY`, `SPEED → ERASE`, `LLAMA → BALSA → [Y,X,Y,X,G]` (one green, two yellows),
    exact match → all green, no overlap → all gray.
- **Reward v1:** first green pays `a` once / re-guess pays 0; yellow-then-green upgrade pays `a+b`
  (never 2·green); double-letter reveal pays exactly one `b`; all-known-constraints guess pays 0;
  invalid and clue-violation dominance (`p_invalid>b`, `q>b`).
- Validity lookup; seeded-split stability + **held-out immutability**; **zero-variance-group
  filtering path**; sequence-grammar round-trip (serialize → tokenize → parse); curriculum promotion;
  replay-queue sampling. **Every code path gets a test** (AGENTS.md).

## 12. Tooling & environment

- **uv** for env/deps/lock. Bootstrap `uv sync`.
- **PyTorch / MPS** (`device="mps"`), CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`). **MPS caveats:**
  (1) a known **AdamW silent-correctness bug** on non-contiguous tensors on older macOS — use macOS
  15+ and recent PyTorch, and **assert `exp_avg_sq` updates** in a smoke test (§5.5); (2) MPS is **not
  fully deterministic** even with a fixed seed — reproducibility is *approximate*, not bit-exact;
  (3) op-coverage gaps can force slow CPU round-trips — watch in the Phase-0 benchmark. MLX is the
  contingency only if MPS blocks us.
- Lint/format: **ruff** (lint) + **ruff format**. **TensorBoard** for curves.
- **AGENTS.md / CLAUDE.md will be updated** to this stack (remove Rust/cargo invariants; add ruff +
  pytest gates, `src/` layout, uv, Python error-handling norms).

## 13. Hyperparameters (tag: **I**=invariant · **H**=hypothesis, expect to change · **R**=routine default)

These are **starting points, not commitments** — the H-tagged rows are exactly the open
experiments where the learning happens (route results through the lab notebook).

| Tag | Group | Param | Default | Note |
| --- | --- | --- | --- | --- |
| R | Model | d_model/layers/heads/d_ff | 256/4/8/1024 | ≈3.2M default (1–5M range): small 128/5/4/512 ≈1.0M, top 256/6/8/1024 ≈4.8M |
| R | Model | context / dropout | 128 / 0.1 | board ~66 tokens |
| I | Tokenizer | vocab | ~34 | 26 letters + 8 specials |
| R | SFT | optimizer / lr / wd | AdamW / 3e-4 / 0.01 | |
| H | SFT | teacher blend (weak/strong) | 70% / 30% | |
| I | SFT | stop | bar reached or 15 min | outcome-based, capped |
| H | SFT | clue-respect bar | ≥80% green-retention | provisional |
| I | GRPO | algorithm | clipped surrogate + k3 KL | full GRPO, no value net |
| R | GRPO | clip ε | 0.2 | |
| R | GRPO | inner epochs K | 1 | raise only if rollout cost dominates |
| H | GRPO | group size G | 16 | **pin via Phase-0 memory benchmark** |
| H | GRPO | secrets_per_update | 8 | → 128 rollouts/update |
| R | GRPO | lr / warmup | 1e-5 / 0.05 ratio | |
| R | GRPO | max_grad_norm | 1.0 | charbull: small models need aggressive clipping |
| H | GRPO | KL coef β | 0.01 | watch the KL curve; verl default 0.001, DeepSeekMath 0.04 |
| R | GRPO | advantage norm | mean-center, **no ÷std** | Dr. GRPO |
| R | GRPO | zero-variance groups | filtered | StarPO-S |
| R | GRPO | temperature | 1.0 train / greedy eval | |
| I | GRPO | discount γ | 1.0 | short episodes |
| H | Reward | a (green) / b (yellow) | 0.1 / 0.05 | |
| H | Reward | p_invalid / q (clue) / step c | 0.5 / 0.5 / 0.02 | needs `p_invalid>b`, `q>b` |
| H | Reward | W_base / W_speed / L | 1.0 / 0.2 / 0.5 | |
| H | Curriculum | tiers / promote threshold | 200→500→1000→full / 0.6 | |
| H | Curriculum | replay capacity / prob | 256 / 0.10 | hard-word replay |
| R | Eval | curve subsample / cadence | 128 held-out / ~25 updates | |
| R | Eval | full held-out cadence | ~200 updates | checkpoint selection |
| I | Budget | SFT cap / RL remainder | ~15 min / ~45 min | ~1-hr cycle |
| I | Repro | seed | 0 | MPS only approx-reproducible |

## 14. Open items (the H-tagged knobs above + these — discovering them is the learning)

- The exact letter-progress shaping ("new info" rule may need iteration; §6.4 is v1).
- Curriculum tier sizes / promotion threshold / replay rate (tune to observed learning speed).
- Final model size within 1–5M, pinned to the measured speed + memory budget.
- Whether information-gain reward / turn-level advantage / constrained-candidate decoding help (Phase 3).
- The clue-respect threshold (§5.6) and KL β — set/tune on first measurement.

## 15. Build order

1. **Phase 0:** `engine` (+tests) → `data` (+split, held-out immutable) → `baselines` → telemetry +
   engine speed + model-rollout/memory micro-benchmark harness → record floor & yardstick. *(§4.6.)*
2. **Phase 1:** tokenizer → §5.2 serialization → model → teacher (SFT data) → SFT (masked loss,
   AdamW smoke test) → measure valid-word + clue-respect bars. *(§5.6.)*
3. **Phase 2:** reward v1 (+tests) → GRPO trainer (§6.3, with `π_θ_old`/`π_ref`, clip, k3 KL,
   grad-clip, zero-variance filtering) → budget-sized rollout loop → curriculum + replay → two-tier
   held-out eval → first learning curve clearing the gate.
