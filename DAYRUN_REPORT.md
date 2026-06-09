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

## 🌙 Overnight #4 (2026-06-07→08, unattended): build the best honest model

**Mandate:** work all night, try new experiments/algos, change any params/algos, **no cheating** (no
dictionary at inference; non-words counted-not-fed-back; clue logic + dict are TRAINING-only; train-only
secrets; clean-protocol free-gen TEST on disjoint held[96:]), update README + map, don't make mistakes.

**Honest scoreboard at lights-out:** best honest model = **validity-max v2 = 0.302** clean-protocol TEST
(stage-1 = 0.243). Lever that works = in-weights validity (spelling); plateauing ~0.78 validity via aux.
Walls: spelling (vocabulary, in-weights) dominant (~97% of losses), then deduction (generalization, data-
bound ~1,852 secrets). Ceiling if validity→1.0 = 0.436 (constrained-mask, dict — not allowed as a model).

**Queue (one-at-a-time background jobs; capture each → README+map+push; adapt on results):**
1. v3 — validity push aux8 → **clean 0.264** (validity 0.767). Within noise of v2's 0.302 → **the aux
   validity lever has PLATEAUED** (validity caps ~0.78, clean win ~0.28–0.30; aux 6→8 = no real gain).
   Best honest model stays validity-max v2 (~0.30, robustly ~0.28). Aux-cranking is exhausted.
2. **Infill no-CoT** (`IF_THINK=0`) → **clean 0.049** · **Infill with-CoT** → **clean 0.000**. Both
   FAILED (near-all non-words; clean-VAL stayed ~0 all of training). **Diagnosis = config mistake, not the
   idea:** I used aux **λ=6** (copied from validity-max, which was a *warm-start fine-tune*); on a
   **scratch** model λ6 lets the per-position validity aux dominate the imitation loss → the model
   optimizes per-position valid-letter mass instead of composing coherent words → non-words. (stage-1's
   scratch run used λ=1.) The with-CoT 0.000 also says think+template+commit is too hard to learn from
   scratch. ⇒ retried no-CoT gentle aux λ=1 → **also failed (clean 0.014)**, so it wasn't the aux.
   **Opener diagnosis is decisive:** the model spells a **valid opener (`slate`)** but produces non-words
   on later turns *once the template has greens/yellows* — i.e. the **explicit template is net-negative**:
   it disrupts constrained generation rather than helping. The model learns clue-conditioning **better
   implicitly from the board** (validity-max) than from an explicit template (the "structured-context
   hurt" lesson, again). **VERDICT: the infill/template idea does NOT work** (fair test: ablation + 2 aux
   configs + opener diagnosis). Pivoted. validity-max v2 (0.302) stays the best honest model.
4. **validity-max v4** — aux6 + constrained self-distill on **ALL 1852 train secrets** → **clean TEST
   0.332** (+0.030 over v2). **New best honest model**, plausibly real: clean-VAL 0.333 ≈ clean-TEST 0.332
   across 463 independent secrets (neither selected-on), and a clear mechanism — validity held ~0.76, so
   the gain is **fewer deduction (ran-out) losses from more in-distribution data**. The data lever is now
   MAXED (all train secrets used). Caveat: +0.03 still warrants a different-seed confirm; flagged.
5. **ensemble committee** (new) — 5-model majority vote, no dict → **TEST 0.283**. NULL: below v4 (0.332)
   and v2 (0.302); the committee averages toward weaker members (stage-1 0.243 drags it) and the models
   aren't diverse enough (all validity-recipe variants) for voting to surface more real words.
6. **v4 verification** (seed 1) → **clean TEST 0.335** ≈ v4's 0.332. **CONFIRMED** — the data-lever gain
   is robust across two seeds, not a lucky draw. Honest free-gen held-out = **0.243 → 0.333** (verified).
7. **v5** — dropout 0.2 on the v4 recipe → **clean 0.283**. NULL/negative: more regularization HURT
   (the model needs the capacity; dropout 0.1 is right). Generalization-via-regularization doesn't help.
8. **v6** — path diversity: 6 teacher passes (more game-paths per secret) on the v4 recipe. Last
   in-distribution data lever. (running)

### 🐞 Bug (user-found on the live dashboard): the all-gray poisoning was NOT fixed in the training path

I claimed the clean protocol fixed the non-word poisoning; the user caught (on the live dashboard) that it
did **not**. The fix lived only in the separate `clean_eval.py` (which computes the headline number);
the **training scripts' own `play()` still did `board_only(g.turns)`** — feeding invalid guesses back as
fabricated **all-gray**, which contradicts real clues. This poisons (a) the **live dashboard** and (b) the
**checkpoint selection** (best epoch chosen on the poisoned metric). Vivid proof: `fluke` — after `GUILE`
(U/L yellow, E green), two non-words (`LUNCE`,`PHEOE`) fed all-gray → the model dropped U, L, *and the
green E* → `FIXIE`,`FIZZY` (valid words, but inconsistent — they abandon every clue).
**Scope:** the reported clean numbers (v4 0.333 etc.) came from `clean_eval` and ARE clean; but the
dashboard + selection were poisoned. **Fix:** `validity_max.play()` now conditions on VALID turns only
(non-words counted, not fed back). **v7** re-runs the v4 recipe with the fixed clean pipeline (clean
dashboard + clean selection) → the proper canonical best model.

**Overnight verdict (forming):** best honest model = **validity-max v4 = 0.333** (verified 2 seeds);
**v7 (fully-clean pipeline: clean play() + clean selection) = clean TEST 0.338** — confirms it, clean
selection edged it up slightly. Honest best ≈ **0.337**.

### 🔎 Investigated a colleague's "90% without cheating" repo (rynowak/mm)

Went deep on `/Users/pedram/repo/mm` (V2 = dense constraint-state encoding). Verdict:
- **Decode is HONEST** — V2 generates free char-by-char, `letter_mask` only masks to the 26 letters (no
  dictionary, no answer-trie, no consistency filter). (My first read wrongly flagged V1's answer-trie; the
  V2/90% path doesn't use it — corrected after the user pushed.)
- **The 90% is train-test CONTAMINATION** — no held-out by answer. Pretrain (`transcripts.py:73
  random.choice(answers)`) + RL (`finetune.py:193 random.choices(answers)`) train on the FULL 2,315 answer
  set; eval (`finetune.py:171 random.sample(answers)`) is a random sample of the SAME set; the only split
  is per-*example* after a shuffle (`data.py:126`). So 90% = memorizing the answer set, not generalizing
  (same class as oreo's rejected 0.89). Their honest analog is their own V1 *unconstrained* ~30% ≈ our 0.33.
- **The genuinely good honest learning:** their **dense constraint-state encoding** — REPLACE raw history
  with a clean summary (greens by pos, yellows w/ excluded pos, grays w/ EXACT counts). Cleaner + richer
  than my failed infill (which *appended* a template to history). Porting it honestly → `dense_encode.py`.

### dense-encode (the honest learning) — DONE: **0.065 TEST** (the encoding *memorizes*, doesn't deduce)

`scripts/dense_encode.py`: V2-style dense constraint state as the ONLY input, free char-gen (letter-mask,
no dict), aux-validity, dense-format spell warm-up, clean protocol, disjoint TEST. Bar to beat: **0.338**.

**Result: clean-protocol TEST win = 0.065 (24/367)**, valid 0.776, avg 3.67 (best VAL 0.094 @ ep35). That's
**~5× WORSE than validity-max (0.338)** — the honest port of the colleague's "90%" architecture lands at
~0.07 held-out. This is the clean confirmation that **their 90% was the train-test contamination, not the
encoding**: reproduced honestly (train-only secrets, disjoint TEST) the dense encoding barely generalizes.

**"Is there a bug?" — investigated deeply, NO bug.** The discriminator: **TRAIN-secret win 0.317 vs
HELD-OUT 0.075**. If the encoding/mask/eval were broken it couldn't win 31.7% on trained secrets — it can,
so the pipeline (`clue_state → encode_constraint → generate-5 → validate → win`) is correct. The gap *is*
the finding: the dense encoding learns `constraint→answer` for **seen** answers (memorizes) and ignores the
clue facts on unseen ones. Receipts: (1) grays ARE encoded and grow (salsa `enclen 7→9→11→13`, grays
`[]→{e,t}→{e,o,t,v}→{e,l,o,t,v,y}`) — no "gray dropped" bug; (2) the model *ignores* them — guessed `salvo`
with `v,o` already gray; (3) that causes the dashboard loop: replaying a word that adds no new gray →
**byte-identical state → deterministic greedy → same guess forever** (turn4≡turn5). Raw-history
(validity-max) can't loop this way because the board always grows — a structural reason the dense state is
*worse* honestly, not just under-trained. **Verdict: dense encoding rejected** (honest TEST 0.065 ≪ 0.338).

### honest free-gen GRPO on the 50M base — DONE: **NULL 0.332 TEST** (the RL question, answered)

User asked: *any RL techniques to push the win rate higher?* Built the first **clean-lineage** RL run
(`scripts/grpo_freegen.py`): correct free-gen GRPO on `validity_max_v4` — full CoT-bearing rollout, reward
= win-dominance + non-word penalty + per-valid partial credit, group-relative advantage (mean-center, NO
/std), zero-variance filter, clipped surrogate, k3 KL to a frozen ref. Honest: reward is engine-outcome
only (no solver/dict), train-only secrets, eval = greedy CoT free-gen (zero rules) on disjoint TEST.

- **Load-bearing CoT (new finding):** forcing immediate commit (no CoT) collapses win **0.333 → 0.073** at
  equal validity (0.866 → 0.847). The ephemeral "think" does the deduction — so RL shapes real reasoning,
  not just spelling. The RL action = the full free generation (CoT + `<guess>` + 5 letters).
- **A real bug, found + fixed:** first attempt *degraded* (reward −0.13, VAL drifting down). Cause:
  `old_logp` computed in eval mode (dropout off) but `new_logp` in train mode (dropout ON) → the surrogate
  ratio `exp(new−old)` was corrupted by dropout noise. Fix: keep the policy in eval mode through the update
  (gradients still flow via `enable_grad`). After the fix, RL behaved correctly.
- **Fixed RL works — on TRAIN.** Over 50 iters: train rollout win **0.41 → 0.60** (+19 pts), train
  non-words **0.57 → 0.36**. It genuinely learns to deduce + spell the train answers better.
- **…and transfers nothing.** Held-out **TEST 0.332 (122/367) valid 0.855 avg 4.30 ≈ base 0.338**. The
  per-eval VAL oscillated 0.333–0.354 (noise); the 0.354 "best" collapsed to 0.332 on the 4×-larger TEST.
- **Verdict: RL does not raise the honest win rate.** A *correctly implemented* GRPO improves train by ~50%
  relative and transfers zero to held-out — the cleanest demonstration yet that the wall is
  deduction-**generalization** (data-bound), not RL mechanics. No RL knob (temp, KL, reward shaping, lr)
  fixes a generalization wall. **RL closed (honestly), at 0.33.**

### Different encodings/architectures (the "go bigger on arch" push) — all NULL, but they explain the wall

After RL, tried two new architectures to attack 0.34 from the representation/computation side (not data/RL):

- **Reasoning-CoT** (`scripts/reason_cot.py`, ≈50M, vocab 270): condition on RAW history (the grounded
  input that generalizes), then DERIVE + write out the constraint state (greens-by-pos / yellows-excluded /
  grays-with-counts) as the CoT, then guess. Teacher supervises the derivation at TRAIN time (allowed); at
  inference the model emits its own reasoning + guess (structured decode, ephemeral, no engine/dict). Bet:
  externalizing the deduction is a scaffold the guess respects, and deriving constraints is answer-agnostic
  so it generalizes. **Result: NULL** (stopped ep20, win plateau ~0.04 ≪ 0.338). The derivation **collapsed
  to a constant** — `derive-acc` pinned at exactly 0.679 (all-BLANK green rows, over-baked by the
  empty-constraint warmup), while win still crept up from the model guessing straight from history. *Lesson:*
  the constraint state is a **deterministic function of the board**, so writing it out adds NO information —
  the model correctly ignores it. CoT only helps when it does search/computation hard in one pass; restating
  givens is not that.

- **Iterative Refine** (`scripts/iter_refine.py`, ≈50M, lean **36-vocab** to keep spelling strong): a learned
  EDIT operator — condition on history + a DRAFT word, output a better word; at play, draft starts BLANK →
  g0, then feed each guess back K passes (full-word lookahead), commit g_K. Trained on near-miss / random /
  identity drafts. Bet: verify-and-fix a concrete candidate is easier than generate-from-scratch, and it does
  genuine cross-pass computation. **Result: NULL, TEST 0.098** — and the pass-ablation is decisive:
  **passes 0 / 1 / 3 are byte-identical (VAL 0.104)**. The refinement is **identity** — the model hits a
  fixed point on pass 0 and ignores the draft. (Also spelled weaker, 0.73, from the draft-conditioning mix.)

**The unifying principle (now 3 architectures, 1 failure mode):**
| approach | mechanism | result |
| --- | --- | --- |
| dense-encode | **bottleneck** guess through explicit constraint state | MEMORIZES (train 0.32 / held 0.07) |
| reasoning-CoT | auxiliary **reasoning** channel | ROUTED AROUND → no-op (~0.04) |
| iter-refine | auxiliary **draft** channel | ROUTED AROUND → identity (0.098) |
| validity-max | **raw history only** | best generalization (**0.338**) |

Any auxiliary channel the model can route around collapses to a no-op (the guess attends to raw history and
the channel carries no info it can't compute). Bottlenecking instead memorizes. Raw history alone
generalizes best. *That tension is why ~0.34 resists a single forward pass.* The one untried mechanism that
escapes route-around: a **separate** consistency network fused **multiplicatively at the logits**
(product-of-experts) — both experts must agree, so it can't be ignored. That's the next candidate if pursued.

The
honest free-gen ceiling is **data-bound at ~0.33** (the ~1,852-secret answer set): more secrets helped
(0.30→0.33, maxed), but every other lever is null/negative — aux plateau (v3), dropout (v5), infill
(template net-negative), ensemble (0.283), RFT/STaR, DPO, scale (turns over). The remaining wall is
deduction-generalization, and it's a *data* limit, not a method we're missing.
4. **Best-recipe combine** — fold the winner (infill?) with constrained self-distill + strong aux.
5. **Longer/stronger pretrain** (better in-weights vocabulary) on the best recipe.
6. **Word-level validity** (new algo) — push spelling past the per-position-aux plateau, if time.
7. Iterate whatever wins; keep TEST-not-VAL discipline (small gains <0.03 = selection noise; verify).

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
| STaR iter-2 (RFT from the 0.319 ckpt) | null | 0.71 | **did NOT compound** — re-warmed from VAL 0.365 but settled VAL 0.31-0.33 |
| **teacher-only control** (no on-policy/constrained, aux 1.0) | **0.259** | 0.656 | **BUSTS the gains**: same VAL-0.365 selection (by chance, ep14) → TEST 0.259, *below* stage-1 |

### Verdict: the win "gains" were selection noise; only validity is robustly movable

The control is decisive. Selecting best-of-16-epochs on a **96-secret VAL** yields TEST anywhere in
**[0.26, 0.32] regardless of method** — the control (a plain teacher-only re-train, no special ingredients)
hit VAL 0.365 by chance and scored TEST **0.259**, *below* stage-1. So **RFT's 0.319 and validity-max's
0.305 are noise, retracted.** Free-gen held-out win is **~0.28–0.30 across every method**, statistically
indistinguishable from stage-1's 0.281.

What *is* real and method-driven: **validity-max raised in-weights validity 0.662 → 0.712** (the control
stayed 0.656) — the cranked aux + constrained-decode self-distillation genuinely improve spelling. **But
it bought no win.** This is the **answer to the composition question**: pushing in-weights spelling does
*not* lift win, because spelling was never the win bottleneck — **deduction-generalization is the wall**,
and it is data-bound (only ~1,852 in-distribution train secrets; held-out answers are off-limits).

**Tally: 9 training approaches now confirm the ~0.28–0.30 free-gen ceiling** (RL ×11, DPO ×3, DAgger ×2,
info-gain, distill, RFT/STaR, validity-max). The only honest way past it is **test-time compute + the
public dictionary** (beam-over-trie 0.55 deterministic; best-of-N 0.72) — the model generates freely,
the dictionary only spell-checks (never clue-filters). That is the real "best model."

### Generation-order flaw (user-found): late greens break L→R spelling; invalid guesses poison context

Two real mechanism bugs surfaced by inspecting lost games on the dashboard:

**1. Invalid guesses are mishandled.** `Game.guess` appends an invalid (non-word) guess as a turn with
`feedback=None` that *still counts toward the 6-guess limit*; and `board_only`/`_fb` encode it into the
next inference context as **all-gray** — a fabrication that says "these letters are absent," often
contradicting greens the model already earned. Worse, **training never contains invalid-word contexts**
(teacher games are always valid), so after its first non-word the model is in an **out-of-distribution,
self-contradictory context** — the likely cause of the non-word *cascades* (`manim→bando→pangy→hanja`).

**2. Left-to-right generation fails on late constraints.** The order diagnostic (free-gen, TEST) is
decisive:

| greens to satisfy | invalid-guess rate | green-violation rate |
|---|---|---|
| none | 6.3% | 0.0% |
| early (pos 1-3) | 37.8% | 4.3% |
| **late (pos 4-5)** | **52.4%** | **19.5%** |

Unconstrained, the model spells fine (6% non-words). With a green near the **end**, it emits a non-word
**52%** of the time and **ignores its own green 1-in-5**. L→R autoregression commits the early letters,
then can't compose a valid word that lands the required late letter. ⇒ generation *order* is a real lever.

**Fixes under test:** (a) **green-anchored decoding** — force-copy known greens, generate only the blanks
(running); (b) **recovery-augmentation SFT** — train on histories containing the model's own non-words so
it stops being OOD and learns to keep greens / not cascade. Both target composition, so capped ~0.55 by
the deduction wall, but they attack genuine mechanism bugs (not noise).

### ⭐ Honesty tightened → the validity lever WORKS (the first robust honest gain)

**New honesty rules (user, stricter and correct):** *no dictionary/trie at inference, ever* (retires
beam-over-trie 0.55 / best-of-N 0.72 / constrained-decode 0.44 as "the model"); and a non-word guess is
*counted as a turn* but *not fed back* to the model (kills the all-gray poisoning; also fixes the
train/inference mismatch — clean context = valid-only = what training looked like). The honest metric is
now the **clean protocol** (`clean_eval.py`): pure free-gen, non-words fatal.

Under it, the validity lever — which looked *net-negative* on the (now-retired) dictionary tier — is
**monotonic, mechanistic, and real**:

| model | in-weights validity | **clean TEST win** | non-word losses |
|---|---|---|---|
| stage-1 | 0.66 | **0.243** | 269 |
| validity-max (aux 3) | 0.71 | **0.281** | 250 |
| validity-max v2 (aux 6) | 0.76 | **0.302** | 227 |

+22 wins over stage-1 on 367 disjoint TEST (~2.5σ), and the mechanism is clear (more in-weights validity
→ fewer non-word losses → more wins). **This is the first robust honest improvement of the project.** The
old eval's 0.281 was inflated ~+0.04 by the all-gray "lottery"; the true stage-1 honest number is **0.243**.
Headroom: the validity=1.0 ceiling (constrained-mask, dict) is 0.436, so pushing in-weights validity should
keep climbing toward it. v3 (aux 8, 1500 secrets) running; then the green-conditioned infill (combines).

### [SUPERSEDED by the clean protocol above] Best model = stage-1; validity net-negative *on the dict tier*

*(This held only for the dictionary-aided inference tier, which the user has since ruled out. Kept for the
record.)* Profiling the higher-validity weights (`validity_max`, free-gen valid 0.712) on the honest-spelling tier:
**beam-16 0.529 / best-of-64 0.659** — *worse* than stage-1's **beam 0.55 / best-of-64 0.703**. The
cranked-aux **sharpened the distribution** (more confident per letter), which raises free-gen validity but
*reduces sample diversity* — and beam/best-of-N depend on diversity to surface the answer. So the one
robustly-movable knob (in-weights validity) is **net-negative** for the best honest model. **`stage-1`
(`cot_eph_aux_fair.pt`) is the definitive best model on every honest metric:** free-gen **0.281** ·
beam-over-trie **0.55** · best-of-64 **0.703** · best-of-128 **0.719** (model generates, dict spell-checks
only). Next: testing the generation-order hypothesis (infill/backwards) — capped ~0.55 by deduction, but
the cleanest "rework the model" experiment.

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
