# Back to Free Generation — Architecture Migration (v3 → A)

**Status:** Proposed
**Author:** Wordle SLM team
**Package:** wordle_slm

## Version History
| Version | Date | Summary |
| --- | --- | --- |
| 1.0 | 2026-06-02 | Revert the v3 restricted-action-space pivot; restore free char-generation (spec §5/§6) as the architecture of record. |
| 1.1 | 2026-06-02 | Incorporated design-review feedback: specified the teacher-forced log-prob recomputation + `π_θ_old` storage (C1); committed to the 26-letter action-space mask at both generation & recomputation (C2); explicit `RewardConfig` fields + `gray_known` state (H1/M4); precise `generate()` signature + CPU-generator guard (H4/M2); disambiguated `π_ref` for V vs Q + made SFT non-optional (H3); invalid-turn prompt encoding (M1); mask-helper ownership (H2); rollout secret guard + what the tracer drops (L1/L2); solo-sequencing note (M3). |

## 1. Problem Statement

The v3 pivot (spec §1.5) made the model a **ranker over a pre-filtered list of still-consistent
words**. It hits ~99% win at random init — but the model never learns words: validity and
consistency are computed by classical code (`engine/constraints.py` + the word list), and the model
only learns *which consistent word splits best*. The project's purpose is to **train an SLM that
genuinely understands/generates words** and learns the strategy ("the journey is the point"). By that
goal, handing the model the filtered candidate list is cheating.

Decision (user, 2026-06-02): **revert to free generation (architecture "A")** — the original spec
§5/§6. The model generates 5 letters from its own weights; the engine validates (a non-word wastes a
turn, exactly as for a human). No candidate list reaches the model.

## 2. Current vs target

- **Current (v3, in code):** `CandidateScorer` ranks candidates; `play_game` filters to consistent
  answers and selects; reward = info-gain over the consistent set; tracer replays selection.
- **Target (A, in spec §5/§6 — already fully specified there):** decoder-only char transformer that
  **generates**; rollout = prompt→generate-5→validate→score→append; reward = §6.4 shaped
  (validity + clue-respect + win, with the dominance ordering); GRPO over per-token generation
  log-probs, loss **masked to the 5 guess-letter positions** (feedback/board tokens never get
  gradient).

The **environment** keeps a dictionary (Wordle's own rule: it only accepts dictionary words and
draws secrets from a list). That is unavoidable and is *not* what changed — what changed is that the
**model** no longer consumes it.

## 3. Scope — KEEP / MODIFY / REPLACE

**KEEP (architecture-agnostic, already merged & reviewed):** `engine/` (scoring, game, constraints —
constraints still used by the InfoMax teacher), `data/`, `model/tokenizer.py`, `telemetry/`,
`config/resolve.py`, `rl/curriculum.py`, `baselines/policies.py` (now the **SFT teachers**),
`baselines/phase0.py` (floor/yardstick/budget; one stale comment to fix), `cli.py`, all stubs.

**MODIFY:** `config/__init__.py` (`RewardConfig` → §6.4 fields: `a, b, p_invalid, q, c, win_base,
win_speed, loss_penalty`), `rl/__init__.py` (re-export the new reward), `tests/test_smoke.py`
(reward-default assertion).

**REPLACE (the 5 v3 files + their tests):**

| v3 file | becomes | plan step |
| --- | --- | --- |
| `model/scorer.py` (CandidateScorer ranker) | `model/transformer.py` — decoder-only generator (forward + `generate`) | G |
| `model/serialization.py` (encode_board/encode_word for ranking) | §5.2 board prompt grammar with `<GUESS>` | F |
| `rl/rollout.py` (select-from-consistent) | `play_game`: prompt→generate-5→validate→score | Y |
| `rl/reward.py` (info-gain) | §6.4 shaped reward (`green_known`/`min_count` rule + dominance) | H |
| `rl/tracer.py` (replay selection) | GRPO tracer over per-token generation log-probs | V |

The **GRPO scaffolding** in the current tracer (`compute_group_advantages`: mean-center, no ÷std,
zero-variance filter; k3 KL; clipped surrogate; grad-clip; optimizer step; TensorBoard scalars) is
architecture-agnostic and is **salvaged**; only the per-step log-prob computation changes from
candidate-softmax to per-token generation log-probs (masked to guess-letter positions).

## 4. Component designs (detail lives in spec §5/§6; deltas here)

### 4.1 Serialization (F) — spec §5.2
Byte-exact grammar: `<BOS> {completed_turn} current_turn`, where
`completed_turn = 5 letters + 5 feedback tokens + <SEP>` and `current_turn = <GUESS> + (model emits
5 letters) + <EOS>`. The current `encode_board` already emits the completed-turn history; the delta
is appending `<GUESS>` for the turn being generated (it is **never stored in history**). Public
surface:
- `build_prompt(turns) -> list[int]` — board history + `<GUESS>` (the decode prompt).
- `guess_letter_positions(seq) -> list[int]` — the indices of the 5 letters after each `<GUESS>`;
  this **mask helper ships in F (PR-2)** so the rollout (Y) and tracer (V) both import it (no
  circular dep, no copy). (review H2)
- **Invalid turns must still appear in the prompt** (review M1): an invalid guess consumes a turn,
  so the model must see it or it loses its turn count. Decision: encode an invalid completed turn as
  its 5 guess letters + **5×`<gray>`** + `<SEP>` ("burned a turn, learned nothing"). This reuses the
  existing grammar (no new token, tokenizer untouched); the mild ambiguity with a real all-gray
  valid guess is acceptable (both mean "no usable info, turn spent"). The current `encode_board`
  *skips* invalid turns — that is a generation bug and must change. A serialization test covers the
  invalid-turn case.

### 4.2 Model (G) — spec §5.3
Decoder-only transformer: token + learned positional embeddings, pre-norm causal blocks, **weight-
tied** output head, char vocab (34). Default ~3.2M (1–5M range). Concrete interface (review H4):
- `forward(ids: Tensor[B, L]) -> logits[B, L, vocab]` (causal mask).
- `generate(prompt_ids: Tensor[L], *, sample: bool, generator: torch.Generator | None) ->
  Tensor[5]` returning **exactly 5 letter token ids**. The 26-letter logit mask is applied **inside**
  `generate` (the action space is the 26 letters — special tokens get `-inf`). If `generator` is
  given it **must be a CPU generator**; sampling (`multinomial`) runs on CPU (the PR-B MPS lesson),
  asserted at entry (review M2). Greedy when `sample=False`.
- **Action-space mask is the policy (review C2 — decided):** the policy is a distribution over the
  **26 letters**, not the full 34-token vocab. The *same* 26-letter mask applied at generation is
  applied when the tracer recomputes log-probs (§4.5). So `log π(letter) = log_softmax(logits[26
  letter ids])[letter]`. A ratio test asserts `π_θ/π_θ_old == 1` before any update.

### 4.3 Rollout (Y) — spec §6.1
`play_game(model, tokenizer, secret, *, sample, generator, device, max_guesses=6) -> Game`: build
the prompt (F), generate 5 letters → word, `game.guess(word)` (engine validates; invalid consumes
the turn), append the scored turn, repeat. **No consistency filter, no `pool` parameter.** Greedy
(argmax) for eval, sampled for training. Sampling on CPU (MPS lesson, PR-B). Entry guard: assert the
secret is a real answer (`is_valid(secret)` / in the answer list) — the generation replacement for
the v3 `secret_in_pool` guard (review L2). Save/restore `model.training` via try/finally (PR-B
lesson).

### 4.4 Reward (H) — spec §6.4
New `RewardConfig` (replaces the v3 `info_gain_weight`/`win_speed` fields; review H1):
`a` (green bonus), `b` (yellow bonus), `p_invalid` (invalid-word penalty), `q` (clue-violation
penalty), `c` (step cost), `win_base`, `win_speed`, `loss_penalty` — 8 fields. Dominance asserts use
these exact names.
Per-guess shaped reward over **per-game knowledge state local to `compute_reward`** (not threaded
out): `green_known` (positions), `min_count` (letter→known min), and `gray_known` (letters confirmed
absent — needed to detect "reuses a known-gray letter"; review M4):
- letter progress `a·new_greens + b·new_yellows` (paid once; yellow→green pays `b` then `a`; dups
  only when raising known min-count; re-confirming pays 0),
- invalid-word penalty `−p_invalid` (consumes the turn, no letter progress),
- clue-violation penalty `−q` (drops a known green / reuses a known gray), with `q > b`,
- step cost `−c`, terminal `+(W_base + W_speed·(6−t))` on win / `−L` on loss.
- **Dominance asserts:** `p_invalid > b`, `q > b`, and `max farming (≈5a + few·b) < W_base` — a win
  always beats stalling/farming.

### 4.5 Tracer / GRPO (V) — spec §6.2-6.3
Per-token GRPO. Trajectory return `R_i = Σ` shaped rewards; advantage `A_i = R_i − mean(R_group)`
(no ÷std), broadcast to the guess-letter tokens of trajectory `i`; zero-variance groups filtered.
Loss = clipped surrogate (`r_t = π_θ/π_θ_old`) **summed over guess-letter tokens only** + `β`·k3-KL
vs frozen `π_ref`. The **defining test**: after one update, gradient is **non-zero on guess-letter
token logits** and **exactly zero on context/feedback/board token positions** (the loss mask is
correct).

**Log-prob recomputation — the riskiest detail, specified (review C1/C2):** the v3 "re-score the
same candidate tensor" trick does NOT transfer. `play_game` samples letters under `no_grad`; to get
a *differentiable* `log π_θ(letter_k | prefix)` we do a **teacher-forced causal forward** over the
realized sequence `[prompt ++ letter_1..letter_5 ++ <EOS>]`, read the logits at positions
`prompt_len + k` for k=0..4, apply the **26-letter action mask** (§4.2, Option A — same mask as
generation), `log_softmax`, and index the realized letter. Sum those 5 (per guess) → the trajectory
log-prob; the loss mask is exactly these positions.
- **`π_θ_old`:** for the tracer (K=1) it is the **detached log-probs** captured from this same forward
  before the optimizer step (so `r_t = exp(logp − logp.detach()) = 1` in value, carries `∇logp`).
  Note explicitly: the full trainer Q (K>1 inner epochs) instead snapshots `θ_old` **weights**
  (deepcopy) and recomputes — V's structure must not preclude that.
- **`π_ref` (review H3):** for the **tracer V**, `π_ref = π_θ` at step 0 (shared random init → KL
  starts at 0; the test is purely mechanical). For the **real RL run (Q)**, `π_ref` is the **frozen
  SFT checkpoint — SFT is not optional** (§6); anchoring KL to a random init is meaningless.

**What is NOT salvaged from the current tracer (review L1):** drop the `filter_consistent` import,
the `pool`/`candidates` narrowing, and `candidates.index(turn.guess)`. **Salvaged:**
`compute_group_advantages` (mean-center, no ÷std, zero-variance filter), the k3-KL math, the clipped
surrogate + grad-clip + optimizer-step + TensorBoard-scalar scaffolding, and the
`group_size>=2` / non-finite guards.

## 5. Test contracts (design → tests → code)

- **Serialization (F):** the §5.2 example round-trips byte-exact; `<GUESS>` only on the current turn,
  never in history; completed turns carry 5 letters + 5 feedback tokens + `<SEP>`; the guess-letter
  position mask selects exactly the 5 letters after each `<GUESS>`.
- **Model (G):** param count ∈ [1M, 5M]; `forward` output shape `[B, L, vocab]`; `generate` emits
  **exactly 5 letter tokens** (never a special token); greedy is deterministic; runs on MPS.
- **Reward (H):** exact values — first green paid once; yellow→green = `a+b` across turns; duplicate
  letter single `b`; re-confirm = 0; invalid = `−p_invalid` and consumes the turn; clue-violation =
  `−q`; the three dominance inequalities hold with the §13 coefficients.
- **Rollout (Y):** a full game round-trips & matches §5.2; greedy + sampled paths; an invalid
  generation consumes a turn and cannot win; ≥2 callers (no private rollout copies).
- **Tracer (V):** one update runs on MPS without shape/NaN errors; **gradient non-zero on guess-letter
  tokens, zero on context/feedback tokens**; advantage non-zero on ≥1 group; KL ≥ 0; π_θ moves off
  π_ref; reward/L_clip/KL/entropy → TensorBoard; does **not** assert reward rises (random model).

## 6. SFT / "learn to spell" (consequence of from-scratch generation)

A from-scratch 1–5M model starts with **zero** English knowledge. Free-generation RL from pure noise
spends its whole budget learning to spell. So the head-start (spec §5.4-5.5) is now load-bearing,
**not** optional (this reverses the v3-era "drop SFT" lean):
- **Teacher data (M):** transcripts in §5.2 format from the `ConsistentGuesser`/`InfoMaxGuesser` we
  already built (~70/30 blend), varied openers, target = the teacher's guess letters.
- **SFT (N):** next-token CE **masked to guess-letter positions**, AdamW (MPS `exp_avg_sq` smoke
  test), outcome-based stop (cap ~15 min); save a reloadable checkpoint → init the policy + freeze
  `π_ref`.
- Optional cheap **spell warm-up**: a few hundred steps of plain LM over the word list so letter
  statistics exist before imitation. (Decide after the SFT-overfit gate W.)

## 7. Success bar reset (PRD §6)

Free generation from scratch in ~1 hour will **not** reach 98.9%. Reset to the project's original
primary goal (spec §1 / PRD §10 already state it): **held-out win rate clearly and steadily above
the random floor, with a small practiced-vs-held-out gap and an explainable learning curve;
≥80% is the aspirational stretch.** The v3 consistent-over-answers number (~98.9%) becomes a *ceiling
reference* (what consistency-only play achieves), not the model's target.

## 8. Migration PR order (streamlined gauntlet: 1 /code-review + 1 /adversarial each)

1. **PR-1 (docs):** demote spec §1.5 → "considered & rejected (doesn't teach word understanding)";
   restore §5/§6 as authoritative; reset PRD §6; fix the phase0 stale comment + the `test_phase0`
   "v3 yardstick" naming. *(this doc + reconciliation)*
2. **PR-2 (env primitives) ∥:** §5.2 serialization (F) + §6.4 reward (H) — both leaf, torch-free,
   independent.
3. **PR-3 (model):** the decoder-only generator (G). Independent of PR-2; can run in parallel.
4. **PR-4 (rollout Y):** depends on F (PR-2) + G (PR-3).
5. **PR-5 (tracer V):** depends on G + H + F + Y.
6. **Later waves (unchanged plan):** Q (GRPO trainer), M/N (SFT data + training) + W (overfit gate),
   O (model-rollout benchmark), P (Phase-1 eval), then U (first RL run).

Each REPLACE deletes the v3 file + its test in the same PR that adds the generation replacement, so
the suite is never red. **Solo-sequencing note (review M3):** when doing PR-2/PR-3 one at a time,
prefer **G (model) first**, since Y (PR-4) is blocked on G's `generate()` interface — don't leave F
merged and idle while G is the long pole.

## 9. Risks

- **Won't hit high win rates in 1hr** — accepted; the success bar is reset (§7). The deliverable is a
  *learning curve above the floor*, the genuine SLM+RL learning experience.
- **Rework cost** — 5 files + 5 tests rewritten. Bounded; foundations (engine/data/tokenizer/
  telemetry/config/curriculum/baselines/phase0) and the plan stand. GRPO scaffolding is salvaged.
- **From-scratch spelling is hard** — mitigated by the SFT head-start (now load-bearing, §6) and the
  W overfit gate before the real run.
- **MPS reproducibility** — already handled (CPU sampling; fixed seeds; approximate determinism doc).
