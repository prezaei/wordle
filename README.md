# wordle-slm

Train a small language model **from scratch** and teach it to play Wordle with reinforcement
learning (GRPO). A learning-first project; runs locally on Apple Silicon (PyTorch / MPS).

- **Why & what:** [`PRD.md`](./PRD.md)
- **How (spec):** [`docs/design/wordle-slm.md`](./docs/design/wordle-slm.md)
- **Build plan:** [`docs/design/wordle-slm-plan.md`](./docs/design/wordle-slm-plan.md)
- **Working agreement:** [`AGENTS.md`](./AGENTS.md)

## Setup

```bash
uv sync
uv run pytest
uv run wordle-slm --help
```

## Status

S0 (scaffold) in place. See the build plan for the wave-by-wave roadmap.

## Experiment Log

The honest headline metric is the **held-out win rate** on the immutable 463-word split
(`data/wordlists.split` — train/held disjoint, held never trained on), greedy, free-generation,
**no inference-time rules** (no dictionary, no consistency filter, no candidate list). Numbers that
are *not* honest-greedy-held-out (seen/train probes, beam+dict decoding, leaked CoT) are labeled as
such. The whole thread runs 2026-06-02 → 06-04 on the M5 Max (MPS). All experiment drivers live in
[`scripts/`](./scripts/) (uncommitted; one script == one experiment, docstring at top states the test).

### Results leaderboard

Honest held-out only (greedy, free-gen, no rules), best-first. Yardsticks and the inference-aided
high-water mark are listed separately at the bottom — they are **not** comparable honest-greedy numbers.

| Approach | Size | Held-out win | Valid-rate | Avg guesses | Notes |
| --- | --- | --- | --- | --- | --- |
| **DPO commit-sharpening** 🏆 | 50M | **0.631** | 0.779 | 3.89 | **Honest best.** `dpo.pt` (`dpo_commit.py`); DPO on self-play win/loss commit pairs atop the model below; first preference method to move held-out (GRPO was flat) |
| ephemeral-CoT + aux trie-validity | 50M | 0.616 | 0.788 | 3.87 | `cot_eph_aux.pt` (`cot_ephemeral_aux.py`); the two honest SFT levers stack **super-additively**; the DPO base |
| char + aux trie-validity loss | 50M | 0.436 | 0.675 | 3.62 | `sft_aux.pt` (`train_auxvalid.py`); the spelling lever alone |
| ephemeral-CoT (honest scratchpad) | 50M | 0.430 | 0.671 | 3.69 | `cot_eph.pt` (`cot_ephemeral.py`); plain-CE, the search lever alone |
| char SFT, deep + converged | 50M | 0.402 | 0.664 | 3.58 | `sft_deep.pt` (`train_deep.py`); the matched no-CoT/no-aux plain-CE baseline |
| char SFT, strong InfoMax teacher | 25M | 0.391 | 0.66 | 3.52 | `sft_xl.pt` (`train_path_a.py`); big memorization gap but best generalizer at the time |
| char SFT | 4.8M | 0.300 | 0.61 | 3.66 | `sft_big.pt` (`train_sft_big.py`) |
| BPE+TinyStories recipe (honest split) | 11M | 0.257 | 0.96 | 3.24 | `oreo_recipe.py`; great spelling, memorizes seen, weak on novel |
| char SFT | 3.2M | 0.205 | 0.54 | 3.66 | `train_run.py` (`sft_strong.pt`) |
| char SFT, diverse secret set | 25M | 0.220 | 0.525 | 3.48 | `train_path_a_div.py`; diversity killed the gap but hurt win |
| BPE-on-flat-wordlist | 12M | 0.212 | 0.849 | 3.27 | `bpe_wordle.py`; valid words, bad strategy |
| char + diverse curriculum (co-design) | 50M | 0.188 | 0.526 | 3.63 | `train_codesign.py`; rare-word dilution hurt answers-only eval |
| BPE+TinyStories recipe (honest split) | 50M | 0.190 | 0.956 | 3.10 | `oreo_recipe.py` @50M; overfits early, generalizes worse than char |
| BPE-on-flat-wordlist | 50M | 0.188 | 0.806 | 3.05 | `bpe_wordle.py` @50M; win flat vs 12M → flat-wordlist is the limiter, not size |
| CoT-50M (honest self-context) | 50M | ~0.192 | 0.662 | 3.39 | corrected number; the 0.456 was a leak (see retraction below) |
| RL / GRPO (8 formulations) | 4.8M–50M | ≤ base (0.436) | — | — | all flat or degrading on held-out; dead end |
| — *yardsticks & inference-aided (NOT honest-greedy)* — | | | | | |
| InfoMax teacher (live consistency filter) | — | 0.99 | 1.00 | 3.55 | strategy ceiling; not a learned free-gen model |
| Consistent teacher (plays a real word each turn) | — | 0.967 | 1.00 | 4.46 | weaker strategy yardstick |
| oreo-ai recipe, **SEEN/train** secrets | 11M | 0.87 (≈ oreo 0.89) | 0.96 | 3.24 | **train/test contamination** — evaluated on trained secrets |
| beam+dict decoding (sft_xl) | 25M | 0.580 | 1.00 | 3.70 | inference-aided high-water mark (dictionary trie at decode) |
| beam+dict+norepeat decoding (sft_xl) | 25M | 0.596 | 1.00 | 3.77 | same, never re-emit a prior guess |
| Random baseline / Consistent floor | — | floor | — | — | floor/yardstick references |

### Chronological log

#### 2026-06-02 — scale ladder, teacher choice, first RL

| Time | Experiment (script) | What it tested | Config | Result (held-out) | Takeaway |
| --- | --- | --- | --- | --- | --- |
| 16:30 | 3.2M SFT (`train_run.py` → `sft_strong.pt`) | baseline free-gen SFT | 3.2M char, pretrain + InfoMax teacher | win **0.205**, valid 0.54, avg 3.66 | floor of the scale ladder |
| 16:48 | 4.8M SFT (`train_sft_big.py` → `sft_big.pt`) | scale up + deeper pretrain | 4.8M char, cosine decay | win **0.300**, valid 0.61, avg 3.77 | scale lifts win; valid-rate still the bottleneck |
| 17:27 | GRPO on 4.8M base (`train_grpo_run.py`) | does RL move greedy play | lr 8e-5, loose KL, full train set | win ~0.29 (no gain over base), probe gap +0.38 | RL moves but memorizes train; no held-out gain |
| 17:44 | GRPO, diverse 14k secrets (`train_grpo_run.py`) | RL on a non-memorizable secret pool | replay off, 14,392 RL secrets | win ~0.27, **reward negative** | rare-word secrets too hard; gap halved but reward went negative |
| 20:15 | 25M SFT, strong teacher (`train_path_a.py` → `sft_xl.pt`) | how far pure imitation climbs | 25M, 5-pass InfoMax-on-answers | win **0.391**, valid 0.66, avg 3.52 | **best of the day**; large memorization gap (probe 0.77 vs held 0.39) but best generalizer |

#### 2026-06-03 — diversity, self-distill, decoding probes, the 8-formulation RL sweep, model+curriculum redesign, aux-validity (the best)

| Time | Experiment (script) | What it tested | Config | Result (held-out) | Takeaway |
| --- | --- | --- | --- | --- | --- |
| 08:20 | 25M SFT diverse (`train_path_a_div.py`) | break memorization w/ full valid-word secrets | InfoMax-answers + Consistent-on-valid | win **0.220**, valid 0.525, gap collapsed to +0.05 | diversity killed the gap but **hurt win**; weak Consistent teacher dominated corpus |
| 08:31 | beam+dict decoding (`beam_eval.py`, sft_xl) | how much win greedy leaves behind | beam width 12, ±dict trie | greedy 0.392 / beam 0.392 / **beam+dict 0.580** | dictionary at decode banks +19pts spelling → 58% (inference-aided ceiling) |
| 09:21 | self-distillation (`self_distill.py`) | bank the beam+dict spelling gain into greedy | SFT on model's own beam+dict games | greedy **0.384** (was 0.391), valid 0.719 | spelling improved but win didn't transfer; gibberish→valid not enough |
| 09:39 | no-repeat decoding (`norepeat_eval.py`, sft_xl) | forbid duplicate guesses | beam w10, ±dict ±norepeat | beam+dict+norepeat **0.596** | small decode gain; new inference-aided high-water mark |
| 09:56 | RL validity+consistency reward (`rl_consistency.py`) | RL #3: reward legality, not just win | diverse secrets, +0.1 consistent / −1 invalid | win **0.384** (base 0.422), reward negative | no gain over base |
| 10:01 | turn-budget probe (`turnbudget.log`) | do more guesses recover wins | sft_xl, max 6/8/10 | win 0.392 flat (>6 adds gibberish) | extra turns don't help; >10 overflows context_len 128 |
| 10:19 | per-guess GRPO (`rl_perguess.py`) | RL #4: clean per-guess credit | single-guess episodes, mean-centered | win **0.384** (base 0.422) | cleanest credit, still no gain |
| 10:46 | dict-in-the-loop RL (`rl_dict.py`) | RL #5: trie surfaces words in training | trie-sampled candidates, free-gen eval | win **0.384**, solved/board 0.04 | trie favors common words, rarely surfaces the answer |
| 10:59 | consistency-constrained RL (`rl_constrained.py`) | RL #6 (decisive): sample from still-consistent set | answer surfaced ~22% boards | win **0.389** (base 0.422), consistency 0.82→0.83 | **answer surfaced and reinforced, win still flat** → barrier is generalization/capacity, not signal |
| 11:49 | 99M scale test (`train_scale.py`) | does scale lift the ceiling | 99M, old answer-only data | plateaued (under-converged) | scale alone on old data doesn't help |
| 12:38 | co-design 50M + diverse (`train_codesign.py`) | redesigned model + diverse curriculum | `large` 50M, InfoMax + Consistent-rare | win **0.188**, valid 0.526, gap +0.09 | diversity-closing-the-gap was illusory; rare-word dilution hurt answers-only eval |
| 13:50 | deep 50M, converged (`train_deep.py` → `sft_deep.pt`) | isolate the **model** redesign | `large` 512×16 ~50M, dropout 0.15, 5-pass InfoMax | win **0.402**, valid 0.664, consistency 0.88 | depth+dropout+convergence = small real win (0.391→0.402) |
| 15:36 | **aux trie-validity loss (`train_auxvalid.py` → `sft_aux.pt`)** ⭐ | bake the dictionary into the weights | train_deep + λ·(−log P next-letter ∈ trie); **no trie at inference** | win **0.436** (202/463), valid 0.675, consistency 0.901, avg 3.62 | **HONEST BEST.** Clean +3.4pts over train_deep; lifted win more than valid-rate |
| 15:54 | RL polish on best base (`rl_polish.py`) | RL #7: squeeze points on sft_aux | GRPO, validity+consistency reward | win **0.436** (unchanged), reward negative | RL can't improve even the strongest base |
| 17:35 | info-gain RL (`rl_infogain.py`) | RL #8: add the missing info-gain term + 12 train guesses | +β·log(\|C_before\|/\|C_after\|) on sft_aux | win **0.436** (unchanged) | the missing reward term was info-gain; adding it still no gain → RL closed conclusively |

> **RL verdict (8 formulations):** trajectory lr1e-5/8e-5 · diverse-secrets · validity+consistency · per-guess clean-credit · more-rounds · dict-in-loop · consistency-constrained · info-gain. **All flat or degrading on held-out win.** The decisive consistency-constrained run surfaced and reinforced the answer yet held-out stayed ~0.39 → the wall is **generalization/capacity, not signal/sampling/reward.**

#### 2026-06-04 — BPE/tokenization, the oreo contamination teardown, context management, pass@N, and the CoT thread (+ retraction)

| Time | Experiment (script) | What it tested | Config | Result (held-out) | Takeaway |
| --- | --- | --- | --- | --- | --- |
| 09:34 | BPE-on-wordlist 12M (`bpe_wordle.py`) | does subword tokenization fix validity | from-scratch BPE on valid list (vocab 433), 12M | win **0.212**, **valid 0.66→0.849** | BPE is THE validity lever, but 12M too small → weak strategy (cycling/repeats) |
| 10:35 | BPE-on-wordlist 50M (`bpe_wordle.py`) | is it a capacity problem | same recipe @50M | win **0.188**, valid 0.806 | win flat at 12M & 50M → **flat wordlist** (no frequency signal) is the limiter, not size |
| 12:19–14:02 | oreo recipe iterations (`oreo_recipe.py`, recipe.log…recipe5.log) | replicate oreo-ai (TinyStories BPE pretrain + SFT) | byte-BPE on TinyStories → pretrain → SFT, 11M | win climbs 0.000→0.132→**0.261/0.257**; **SEEN/train 0.870** | reproduced oreo's ~0.87-0.89 **on seen secrets** = train/test contamination (no held-out split). Honest held-out only 0.257 |
| 15:33 | oreo recipe @50M, honest split (`oreo_recipe.py`) | the recipe at scale, strict held-out | TinyStories BPE pretrain + SFT, ~50M | held-out **0.190**, seen 0.67, valid 0.956 | overfits early (peak ~epoch 9), generalizes WORSE than char. Final honest ranking: char-50M+aux 0.436 > recipe-11M 0.257 > recipe-50M 0.190 |
| 16:26 | structured-context A/B (`structured_context.py`) | does explicit derived-state help | raw board vs board+greens/present/absent block, 14M aux-SFT, held 200 | raw **0.260** vs +state **0.170** (Δ −0.090) | explicit state **hurts** (redundant + longer seq); context mgmt is a non-lever |
| 16:31 | **pass@N probe (`passk.log`, on sft_aux)** | is the wall capacity or decoding | 150 held-out, sample N games | greedy 0.453 · pass@1 0.353 · pass@5 0.720 · **pass@10 0.787** | **MAJOR CORRECTION: it's a decoding/search gap, not a capacity wall** — knowledge generalizes, greedy just doesn't find the line |
| 17:23 | CoT A/B 14M (`cot.py`) | does reasoning surface the latent pass@10 | no-CoT vs `<think>`-cands-then-guess, held 200 | no-CoT 0.155 vs **CoT 0.415** (Δ +0.26) | looked like the path to a new best (later RETRACTED — leak) |
| 19:33 | CoT-50M (`cot_50m.py` → `cot_50m.pt`) | scale the winning CoT | 50M, teacher reasoning traces | reported win **0.456** | beat 0.436 — **but this number was inflated by an inference-time leak** |
| 20:37 | CoT-50M + aux (`cot_50m_aux.py`) | stack the two honest levers | CoT + aux trie loss, 50M | incomplete (killed at epoch 24, subsample 0.406; no full-463 milestone) | superseded by the leak finding before completion |
| 20:37 | **CoT integrity teardown (`cot_show.py` → cotshow.log)** | are the CoT numbers honest | A/B on the SAME 0.456 model, held 120 | teacher-context (past `<think>` via consistency filter) **0.450** vs honest self-context **0.192** (Δ −0.258) | ⚠️ **RETRACTION:** CoT numbers were leaked — past `<think>` blocks were rebuilt at inference using the banned consistency filter. Honest CoT-50M ≈ **0.192**, far below 0.436 |
| 20:46–22:00 | **ephemeral-CoT (`cot_ephemeral.py` → cot_eph.pt)** | honest fix: throwaway scratchpad (history is board-only, regenerate think each turn, discard) | 50M, plain-CE, train==infer distribution, no filter at inference | **held-out 0.430** (199/463), valid 0.671, avg 3.69 (best ckpt e29; still climbing — subsample 0.469 at e29) | ✅ honest CoT **works**: +2.8pts over the matched no-CoT plain baseline (0.402), ≈ ties the 0.436 best, 2.2× the leaked-model honest 0.192 |
| **2026-06-05 22:27–00:40** | **ephemeral-CoT + aux 🏆 (`cot_ephemeral_aux.py` → cot_eph_aux.pt)** | stack the two honest levers, run long | 50M, CoT (ephemeral) + aux λ=0.5 **gated to current-turn**, 50 ep, cosine 4e-4→4e-5, 5 teacher passes | **held-out 0.616** (285/463), valid 0.788, avg 3.87 (best ckpt e45; curve 0.604) | 🏆 **NEW HONEST BEST.** Super-additive: 0.402 → +aux 0.436 → +CoT 0.430 → **+both 0.616** (+0.214). Honest-greedy now **beats** the inference-aided beam+dict mark (0.58–0.60). Wins sleek/surer/weedy; loses only the hard tail (joist `_oist`, salsa) |

> **CoT status (resolved → breakthrough):** Done honestly (ephemeral scratchpad, no filter at inference, train==infer), CoT works: 0.402 → 0.430 alone, and **stacked with aux-validity it reaches 0.616 honest held-out** — the two levers are super-additive (CoT enumerates candidates, aux makes the enumeration valid). The earlier 0.415/0.456 were a leak (past `<think>` rebuilt via the consistency filter; honest ≈ 0.192). The model genuinely reasons (traces: 💭candidate → 🎯GUESS) and **pass@10 = 0.787** holds (leak-free), so the decoding/search gap was real and reasoning closed it. The honest-greedy 0.616 now exceeds the inference-aided beam+dict mark (0.58–0.60). Remaining losses are the hard tail (joist `_oist`; salsa double-s/a).

#### 2026-06-05 — Wordle-rules audit, then RL on the 0.616 base (10-row), self-consistency, and DPO

Rules audit first: verified the engine against the official rules (`scoring.py` two-pass duplicate
handling exact — `lever`/`eaten`, `geese`/`these`; 5 letters; 6 guesses; win = all-green). One
deliberate deviation — the engine **accepts** a non-word and burns the turn (the app rejects it), which
makes our benchmark **stricter** than real Wordle. Hard mode (reuse hints) not enforced (standard mode).

| Time | Experiment (script) | What it tested | Result | Takeaway |
| --- | --- | --- | --- | --- |
| 08:00–09:08 | **expert-iteration / ReST (`rl_expert_10row.py` → `rl_expert.pt`)** | distill self-play wins, 10-row | held10 0.604→0.635→**0.646**→revert; full-463 ≈ 6-row 0.62 / **10-row 0.637** | **the RL that works** — taught the model to *use* rows 7–10 (base wasted them: held6==held10==0.604). +4pts on 10-row, ~flat on 6-row. Naive STaR on the model's own noisy think first **degraded** it (0.604→0.531) → fixed by clean teacher-think rebuild (RAFT) |
| 09:42–14:01 | **higher-ceiling / reachability (`rl_expert_tail.py`)** | full-coverage + tail-focus high-K | **solved 1843/1852 = 99.5%**; full-coverage SFT reverted (no gain) | **coverage is NOT the bottleneck** — sampling wins ~every train secret; the gap is commit/generalization, not reach. More expert-iter = dead end |
| 14:07–14:56 | **GRPO polish (`rl_grpo_polish.py`)** | token-level GRPO on the CoT policy, 10-row | full-463: 6-row 0.622 / 10-row 0.637 (flat) | **9th GRPO confirmation: flat.** First attempt blew up (KL 12→294, degraded) from a dropout-in-forward bug; fixed (eval-mode forward + KL 0.05 + lr 5e-6) → stable but barely moves |
| 14:56–15:08 | **self-consistency probe (`self_consistency.py`)** | vote vs pass@N (held-out 150, 6-row) | greedy 0.607 · **vote@12 0.627 (+0.02)** · **pass@12 0.953** | PIVOTAL: pass@12=0.95 (huge latent ceiling) but **voting barely helps** — the winning line is a *minority*; the lever is **selection**, not voting |
| 15:11–15:32 | **DPO, noisy pairs (`dpo_commit.py` → `dpo.pt`)** | DPO on first-divergence win/loss pairs | full-463 6-row **0.631** (+1.5 over 0.616); pref_acc 0.60→0.73 | **first preference method to move held-out.** Capped by noisy credit (outcome ≠ caused by the first divergent guess) |
| 15:34–17:20 | **DPO, decisive-board (`dpo_decisive.py`)** | clean pairs: commit-the-secret vs commit-a-wrong-consistent-word at the same reachable board (6519 pairs) | **flat — reverted to 0.616** (all 5 epochs regressed) | clean credit didn't help: DPO logp over the whole `<think>`+guess let the **long think dilute/hijack** the 5-token commit (loss fell but held6 dropped). Fix = score **guess-tokens only** |

> **RL/technique verdict (2026-06-05):** The bottleneck is the **commit gap**, not knowledge — reachability is 99.5% and pass@12 = 0.95, yet greedy commits wrong. Methods that *reweight outcomes* — GRPO (flat, 9×), self-consistency voting (+2pts) — barely move it. The ones that *sharpen the commit with clean training signal* do: expert-iteration unlocked the extra rows (10-row 0.637), and **DPO is the first to lift honest 6-row greedy (0.616 → 0.631)**, limited by credit-assignment noise — which the decisive-board variant targets.

### Algorithm reference (exact)

The exact per-run algorithm. Every driver is `scripts/<name>.py`; the canonical pieces they call
live in `src/wordle_slm/`. Stated values are the real knobs read from source; "(library default)"
means the run didn't set it and inherited `src/wordle_slm/config/__init__.py`.

#### Shared machinery

Inherited by every run unless its row says otherwise.

- **Engine scoring** (`engine/scoring.py`): two-pass color scoring — pass 1 marks GREEN where
  `guess[i]==answer[i]` and builds a remaining-letter multiset; pass 2 marks YELLOW for a non-green
  position only while that letter has remaining count (decrement on use), else GRAY. An invalid guess
  still consumes a turn (`feedback=None`, rendered as 5×`<gray>`).
- **§5.2 board grammar** (`model/serialization.py`): char vocab = **34** (`<PAD> <BOS> <EOS> <SEP>
  <GUESS> <green> <yellow> <gray>` + a–z; `tokenizer.py`). A completed turn = `<GUESS>` + 5 letters
  + 5 feedback + `<SEP>` (12 tokens; `<GUESS>` IS kept in history — deliberate departure from §5.2 so
  generation and log-prob recompute share an identical context). Prompt = `<BOS> (turn)* <GUESS>`;
  finished game = `<BOS> (turn)* <EOS>`. context_len = **128** (`ModelConfig` default).
- **Model** (`model/transformer.py`): decoder-only **pre-norm** (`norm_first=True`) causal
  transformer, GELU, learned token+positional embeddings, **weight-tied** output head, dropout per
  preset. The action space is the **26 letters only** — generation/log-probs always `log_softmax`
  over the 26 letter logits (a special token can never be emitted). `MODEL_PRESETS`
  (`config/__init__.py`), tuple = d_model × n_layers × n_heads, d_ff, dropout:

  | preset | d_model | layers | heads | d_ff | dropout | ≈params (vocab 34) |
  | --- | --- | --- | --- | --- | --- | --- |
  | `tiny` | 128 | 6 | 4 | 512 | 0.10 | ~1.2M |
  | `base` | 320 | 10 | 8 | 1280 | 0.10 | ~12M |
  | `large` | 512 | 16 | 8 | 2048 | **0.15** | ~50M |
  | `xl` | 640 | 20 | 10 | 2560 | 0.15 | ~98M |

  Several pre-`large`-preset runs hand-build a `ModelConfig` instead (sizes given per row).
- **Spell warm-up** (`sft/pretrain.py`): masked-letter LM over every valid guess (each word →
  `<BOS> <GUESS> w0..w4`), same masked loss as SFT, AdamW. Run-specified epochs/batch/lr.
- **Teacher data** (`teacher/transcripts.py`): plays the **train** secrets with a blend —
  `weak_frac` via `ConsistentGuesser` (opener then a uniform still-consistent word from the **valid**
  list, ~96.7%/4.46) and `1−weak_frac` via `InfoMaxGuesser` (opener then the candidate minimizing
  expected remaining consistent answers over the **answer** pool, ~99%/3.55; `baselines/policies.py`).
  Openers default `("slate","crane","trace","stare","raise","crate")`; most ≥25M runs override with
  `OPENERS=("salet","crane","slate","trace","stare","raise","crate")`. N "passes" = N reseeded
  replays of the train set.
- **SFT objective** (`sft/train.py:sft_loss`): masked next-token NLL over the 26-letter space at the
  5 guess-letter positions after each `<GUESS>` only — `imit = Σ(nll·mask)/Σmask`. With the
  **aux-validity** lever, `loss = imit + λ·aux` where `aux = mean over masked positions of
  −log Σ_letter (softmax · trie_valid_mask)` and `trie_valid_mask` (`valid_continuation_mask` +
  `_valid_trie`) is the set of dictionary-valid next letters given the realized prefix. `λ`
  (`aux_validity_lambda`) **default 0.5**; trie is a training signal only — **inference is never
  trie-aided**. Optimizer AdamW (`weight_decay` default 0.01); best-by-held-out checkpoint kept.
- **GRPO objective** (`rl/grpo.py`, `rl/tracer.py`): group = G same-secret rollouts (sampled
  free-gen). Advantage `A_i = r_i − mean(r_group)`, **no ÷std** (Dr. GRPO `advantage_norm="mean_center"`);
  **zero-variance groups filtered** (`filter_zero_variance=True`). Trajectory log-prob = Σ log p(letter)
  over the guess-letter positions (teacher-forced; logit q−1 predicts letter q). Clipped surrogate
  `min(ratio·A, clip(ratio,1−ε,1+ε)·A)` with **ε=0.2**; ratio uses a frozen θ_old per batch (≡1 at
  inner-epoch 0; K = `inner_epochs` default 1). KL = **k3** `exp(Δ)−Δ−1` to a **frozen π_ref** (the
  SFT checkpoint), Δ = `ref_logp − cur_logp`, **β=0.01** (`kl_beta`); `loss = −surrogate/|tok| +
  β·KL/|tok|`, grad-clip 1.0, linear LR warmup (`warmup_ratio` 0.05), γ=1. Defaults: G=16,
  secrets/update=8, lr=1e-5.
- **Reward** (`rl/reward.py`, `RewardConfig`): per game, knowledge-state carried across turns —
  new-green `a=0.2` (once/pos), new-yellow `b=0.1` (only when it raises a known min-count), invalid
  `−p_invalid=0.5`, clue-violation `−q=0.5` (drops a known green / reuses a known gray), step
  `−c=0.02`, terminal **win** `+(win_base 3.0 + win_speed 0.5·(max_guesses − t))`, **loss**
  `−loss_penalty 1.0`. Dominance held: `p_invalid>b`, `q>b`, max farmable < win_base. Several RL runs
  **replace** this with their own reward (noted per row).
- **Rollout / decode** (`rl/rollout.py`): `play_game` generates each guess letter-by-letter and the
  engine validates it (no candidate list, no consistency filter). Eval = **greedy** argmax,
  free-generation, 6 guesses, on the **463 held-out** (or a stated subsample); training samples
  multinomial. Split (`data/wordlists.py`): seed 0, 80/20 → **1,852 train / 463 held-out**; valid
  list 14,855; answers 2,315; `train_probe` = a fixed train subset for the memorization gap.

#### Per-run algorithm & deltas

Each row gives only what differs from Shared machinery; numbers are exact. "Δ vs …" is the
algorithmic change relative to the named run.

**2026-06-02**

- **3.2M SFT — `scripts/train_run.py`** (`sft_strong.pt`). Model = `ModelConfig()` default
  256×4×8, d_ff 1024 (~3.2M), dropout 0.10. Pretrain 4 ep, batch 512, lr 1e-3. Teacher **3 passes,
  weak_frac 0.5** (default openers). SFT plain NLL (**no aux**), AdamW lr **5e-4**, 60 ep, batch 96,
  grad-clip 1.0, eval every 3 ep on held[:100], best-by-curve. *Baseline run.*
- **4.8M SFT — `scripts/train_sft_big.py`** (`sft_big.pt`). Model 256×**6**×8, d_ff 1024 (~4.8M).
  Pretrain **8 ep**. Teacher **4 passes, weak_frac 0.45**. SFT lr **6e-4** with **CosineAnnealingLR**
  (η_min 6e-5), 70 ep, batch 96. Δ vs train_run: +2 layers, deeper pretrain, +1 teacher pass, more
  InfoMax (weak 0.5→0.45), cosine decay.
- **GRPO on 4.8M — `scripts/train_grpo_run.py`**. Loads `sft_big.pt`; π_ref = frozen copy. Library
  reward + GRPO. **G=16, secrets/update=8, lr 8e-5, kl_beta 0.005**, 200 updates. Secret pool =
  **full valid list minus held-out (~14,392)**, single tier `(None,)`, **replay OFF**; best-by-held-out
  only overwrites if it beats the SFT base. Δ vs the library RL defaults: lr 1e-5→**8e-5**, β
  0.01→**0.005**, non-memorizable 14k pool, no replay.
- **GRPO diverse-14k — `scripts/train_grpo_run.py`** (same script, the 14k-pool/replay-off path
  above). Δ vs "GRPO on 4.8M": identical knobs; logged separately as the diverse-secret RL test
  (reward went negative on rare-word secrets).
- **25M SFT strong teacher — `scripts/train_path_a.py`** (`sft_xl.pt`). Model **512×8×8, d_ff
  2048 (~25M)**. Pretrain **12 ep**. Teacher **5 passes, weak_frac 0.2** (80% InfoMax), `salet`-led
  openers. SFT manual **warmup(300 steps)+cosine**, peak 4e-4 → floor 4e-5, 100 ep, batch 128;
  tracks the probe/held gap. Δ vs train_sft_big: ~5× params, 80% InfoMax (weak 0.45→0.2), 5 passes,
  warmup+cosine schedule, larger batch.

**2026-06-03**

- **25M SFT diverse — `scripts/train_path_a_div.py`** (`sft_div.pt`). Same 25M model; pretrain 10
  ep. Data = **2 passes InfoMax-on-answers (weak_frac 0.3)** + **9,000 `ConsistentGuesser`-on-valid
  games** (random valid secrets minus held-out) for late-game spelling breadth. SFT peak 4e-4 cosine,
  50 ep. Δ vs train_path_a: replaces 3 of the 5 InfoMax passes with 9k Consistent-on-rare-valid games
  (kills the memorization gap, hurts win).
- **beam+dict decode — `scripts/beam_eval.py`** (on `sft_xl.pt`, 250 held). No training. **Beam
  width 12** over the model's own letter distribution; `beam+dict` additionally constrains each beam
  step to a **valid-word trie** (cumulative log-prob; emits the top valid word). Δ vs greedy eval:
  beam search ± dictionary-trie constraint at decode (inference-aided).
- **self-distillation — `scripts/self_distill.py`** (`sft_distill.pt`). Warm-start `sft_xl.pt`;
  generate **1,200 beam+dict (width 8) self-play games** on train secrets + **2 InfoMax passes
  (weak 0.2)**; SFT on the mix, AdamW cosine **1.5e-4→1.5e-5**, 30 ep, best-by-**greedy**-held. Δ vs
  train_path_a: targets are the model's own always-valid beam+dict words (bank spelling into greedy);
  trie touches training data only.
- **no-repeat decode — `scripts/norepeat_eval.py`** (on `sft_xl.pt`, 250 held). **Beam width 10**;
  4 conditions = beam ±dict-trie ±no-repeat (skip any word already guessed this game, else fall back
  to the top beam). Δ vs beam_eval: width 12→10, adds the never-re-emit-a-prior-guess rule.
- **RL #3 validity+consistency — `scripts/rl_consistency.py`** (`rl_cons.pt`). Base
  `sft_distill.pt`. **Monkeypatches `compute_reward`**: per turn invalid −1.0, valid-but-inconsistent
  (via `is_consistent`) −1.0, valid+consistent +0.1, step −0.02, win +3.0+0.5·(6−t), loss −1.0.
  GRPO **G=8, secrets/update=4, lr 5e-5, kl_beta 0.01**, 80 updates, secrets = full-valid-minus-held,
  warmup 8. Δ vs library reward: replaces shaped letter-progress with a flat validity/consistency
  reward (no green/yellow shaping); smaller G and lr.
- **RL #4 per-guess — `scripts/rl_perguess.py`** (`rl_pg.pt`). Base `sft_distill.pt`. **Episode =
  one guess**: at each on-policy board sample **G=8** candidate guesses, reward each (invalid −1,
  inconsistent −1, else 0.2 + 0.2·greens + 2.0 if solves), **mean-center per board**, clipped
  surrogate (ε 0.2) + k3 KL (β 0.01) on the 5-letter per-position log-probs. 5 rollouts/update, 70
  updates, lr 5e-5. Δ vs rl_consistency: trajectory credit → clean per-guess advantage; reward adds
  greens + solve bonus.
- **RL #5 dict-in-the-loop — `scripts/rl_dict.py`** (`rl_dict.pt`). Base `sft_distill.pt`. Behavior
  policy samples guesses **trie-constrained** (`trie_sample`); reward per word = −0.5 if inconsistent
  else 0.1 + 0.15·greens + 1.0 if solves; advantage = mean-centered, clamped ±1.5; loss pushes
  **free-gen** log-prob toward high-advantage words (advantage-weighted, KL-anchored β 0.02), **G=6**,
  5 rollouts, lr 3e-5, 70 updates. Δ vs rl_perguess: candidates drawn from the dictionary trie (not
  free-gen), surrogate is advantage-weighted free-gen log-prob (not a clipped ratio).
- **RL #6 consistency-constrained — `scripts/rl_constrained.py`** (`rl_constr.pt`). Base
  `sft_distill.pt`. Behavior policy samples from the **still-consistent candidate set** (filtered each
  turn, capped 48/board, answer kept reachable); reward 0.1 + 0.15·greens + 1.0 if solves;
  advantage-weighted free-gen push, KL β 0.02, **G=8**, 5 rollouts, lr 3e-5, 70 updates. Δ vs rl_dict:
  candidate pool = consistent set (not full trie) → surfaces the answer ~22% of boards.
- **99M scale — `scripts/train_scale.py`** (`sft_xxl.pt`). Model **768×14×12, d_ff 3072 (~99M)**.
  Pretrain 12 ep (batch 256). Teacher **5 passes weak_frac 0.2** (the train_path_a recipe, old
  answer-only data). SFT peak **2.5e-4** warmup(400)+cosine, 45 ep, batch 64. Δ vs train_path_a:
  only the model size (25M→99M); data/recipe held fixed.
- **co-design 50M + diverse — `scripts/train_codesign.py`** (`sft_codesign.pt`). Model =
  `large` preset (~50M, dropout 0.15). Data = `build_curriculum_pool` (≈14k, difficulty-ordered):
  **3 InfoMax passes (weak 0.2) on the answer secrets** + **5,500 `ConsistentGuesser` games on rarer
  valid words**. SFT peak 3e-4 warmup(400)+cosine, 45 ep, batch 96. Δ vs train_scale: 99M→`large`
  50M, answer-only data → curriculum-pool diverse data.
- **deep 50M converged — `scripts/train_deep.py`** (`sft_deep.pt`). Model = `large` preset (~50M,
  dropout 0.15). Pretrain 12 ep. Teacher **5 passes weak_frac 0.2** (same as train_path_a).
  SFT peak 3e-4 → floor **2e-5**, warmup **500**, **90 ep** (long convergence), batch 96. Δ vs
  train_codesign: drops the diverse Consistent-on-rare data back to pure InfoMax-on-answers (5
  passes), longer training — isolates the model redesign.
- **⭐ aux trie-validity — `scripts/train_auxvalid.py`** (`sft_aux.pt`, **honest best 0.436**).
  Identical to train_deep (50M `large`, pretrain 12 ep, 5 InfoMax passes weak 0.2, peak 3e-4
  warmup(500)+cosine, 80 ep, batch 96) **plus** the aux-validity term in-line: `loss = imit + λ·aux`,
  **λ=0.5**, `aux = −log P(next letter ∈ trie continuations)` at every guess-letter position
  (precomputed per-game trie masks). No trie at inference. Δ vs train_deep: adds the λ=0.5
  aux-validity loss (the only change).
- **RL #7 polish — `scripts/rl_polish.py`** (`rl_polish.pt`). Base **`sft_aux.pt`** (0.436).
  Same monkeypatched validity+consistency reward as rl_consistency. GRPO **G=8, secrets/update=4, lr
  4e-5, kl_beta 0.01**, ≤80 updates / ~1100 s cap, best-checkpoint seeded at the base (cannot
  regress). Δ vs rl_consistency: base is the strongest model (sft_aux), lr 5e-5→4e-5, time-capped.
- **RL #8 info-gain — `scripts/rl_infogain.py`** (`rl_infogain.pt`). Base `sft_aux.pt`. Reward =
  validity/consistency **plus** `+0.1 + β·log(|C_before|/|C_after|)` per valid+consistent guess
  (**β=0.2**, candidate pool = full valid list), step −0.02, win/loss as before. **Rollouts use 10
  guesses** (`G.play_game` monkeypatched, max that fits context_len 128); **eval stays at 6**. GRPO
  G=8, secrets/update=4, lr 4e-5, β 0.01, ≤120 updates / ~1300 s. Δ vs rl_polish: adds the
  information-gain shaping term + 10-guess training rollouts.

**2026-06-04**

- **BPE-on-wordlist 12M — `scripts/bpe_wordle.py`**. **From-scratch BPE on the flat valid list,
  400 merges** → vocab = 7 specials + occurring chunks (~2 tokens/word); guesses generated as subword
  chunks. ~12M model (earlier `base`-class config). Pretrain = word-list LM 8 ep (lr 8e-4); SFT on
  **5 InfoMax passes (weak 0.2)** action-masked, lr 4e-4, 60 ep. Δ vs char SFT: char-34 tokenizer →
  BPE-on-wordlist subwords (the validity lever); guess = chunk sequence.
- **BPE-on-wordlist 50M — `scripts/bpe_wordle.py`** (current on-disk config). Same recipe, model =
  `large` preset (~50M). Δ vs BPE-12M: only model size (win flat → flat-wordlist, not size, is the
  limiter).
- **oreo recipe 11M (honest split) — `scripts/oreo_recipe.py`** (recipe…recipe5.log). **Byte-level
  BPE on 10k TinyStories docs, vocab 2048**; pretrain the transformer on the TinyStories token stream
  (2,000 steps, block 256, lr 6e-4, batch 32), then SFT on InfoMax-teacher games as an **oreo-style
  text transcript** (`guess <word> fb GYBB… win/lose`), action-masked, lr 4e-4 warmup(400)+cosine.
  ~11.5M model, **6 teacher passes (weak 0.2)**. Reports **SEEN/train (0.870)** vs honest held-out
  (0.257). Δ vs char SFT: real-text BPE pretrain instead of spell warm-up + char tokenizer; text
  transcript serialization.
- **oreo recipe 50M (honest split) — `scripts/oreo_recipe.py`** (recipe50.log). Same recipe, model
  **512×16, d_ff 2048, context_len 256 (~50M), 10 teacher passes**, 28 SFT ep, best-by-held-out. Δ vs
  oreo-11M: ~11.5M→50M, 6→10 passes (overfits earlier, generalizes worse).
- **structured-context A/B — `scripts/structured_context.py`**. Model 384×8×6, d_ff 1536,
  **context_len 256** (~14M). Shared spell warm-up; **5 InfoMax passes (weak 0.2)** with **aux λ=0.5**.
  Two SFTs from the same init (25 ep, lr 4e-4): **raw board** vs **board + a derived-state block**
  (`<green>` slots / present / absent letters inserted before each guess). Held 200. Δ vs aux SFT:
  smaller model, +explicit derived-state tokens in the context (the tested lever).
- **pass@N probe — `passk.log`** (on `sft_aux.pt`). No training: **multinomial-sample** N full games
  per secret on 150 held-out and count any-win. greedy 0.453 · pass@1 0.353 · pass@5 0.720 ·
  **pass@10 0.787**. Δ vs greedy eval: sampled decoding, N tries (measures the search gap).
- **CoT A/B 14M — `scripts/cot.py`**. Vocab **35** (adds `<think>`); model 384×8×6, d_ff 1536,
  context_len 256 (~14M). Shared spell warm-up; **5 InfoMax passes (weak 0.2)**. Plain CE (no aux) over
  the loss-masked CoT target = `<think>` + each of **K=3** candidates (1 = the teacher guess + 2 random
  still-consistent answers, shuffled, via `consistent_candidates`) then `<GUESS>` + the guess. A/B from
  the same init: no-CoT (board→guess) vs CoT. Held 200. Δ vs char SFT: +`<think>` token, K=3 candidate
  reasoning block before the guess. ⚠️ later RETRACTED as leaked (cot_show).
- **CoT-50M — `scripts/cot_50m.py`** (`cot_50m.pt`). Same CoT serialization (K=3), model
  **512×16, d_ff 2048, context_len 256, vocab 35 (~50M)**, pretrain 10 ep, **5 InfoMax passes (weak
  0.2)**, 32 SFT ep, lr 4e-4. Δ vs CoT-14M: 14M→50M, otherwise identical CoT recipe. Reported 0.456 —
  inflated by the leak.
- **CoT-50M + aux — `scripts/cot_50m_aux.py`** (`cot_50m_aux.pt`). CoT-50M **plus** the aux-validity
  term at **every** word-letter position (`<think>` candidates AND the committed guess), **λ=0.5**
  (`cot_valid_mask`). Δ vs CoT-50M: adds the aux trie loss (stacks the two honest levers). Killed at
  epoch 24 (subsample 0.406); superseded by the leak finding before a full-463 milestone.
- **CoT integrity teardown — `scripts/cot_show.py`** (cotshow.log). No training; loads `cot_50m.pt`.
  A/B on the SAME model over held 120: **teacher-context** (`play_teacher` rebuilds each past turn's
  `<think>` block with the consistency filter via `cot_prompt` — the leak) **0.450** vs **honest
  self-context** (`play_honest` carries only the model's OWN generated `<think>` forward, board+real
  feedback, filter never called) **0.192**. Δ vs cot_50m eval: removes the consistency-filter
  reconstruction of past reasoning → exposes the −0.258 leak.
- **ephemeral-CoT — `scripts/cot_ephemeral.py`** (`cot_eph.pt`, coteph.log). **One training example
  per turn**: history is **board-only** (`<GUESS>` guess + feedback + `<SEP>`, **no `<think>`**), target
  = fresh `<think>`(K=3) + `<GUESS>` + guess. At inference the prompt is board-only, the model
  regenerates reasoning each turn and **discards** it (never enters later context) — train and inference
  distributions are now identical, filter never called. Model = 50M CoT config (vocab 35); **4 teacher
  passes (weak 0.2)**; 30 ep, batch 128, lr 4e-4. Δ vs cot_50m: past-turn `<think>` removed from
  context (ephemeral scratchpad) — removes both the leak and the train/infer shift. **Full held-out 463
  = 0.430** (best ckpt e29); +2.8pts over the matched no-CoT baseline (0.402).
- **ephemeral-CoT + aux — `scripts/cot_ephemeral_aux.py`** (`cot_eph_aux.pt`). Ephemeral-CoT **plus** the
  aux-validity term, **λ=0.5 GATED to the current-turn supervised positions** (past guesses live in the
  board-only history → must not get aux; `aux_pos = (vmask>0) * loss_mask`). 50M, **50 ep, cosine LR
  4e-4→4e-5, 5 teacher passes**, batch 128. Δ vs ephemeral-CoT: adds gated aux + longer cosine schedule.
  **Full held-out 463 = 0.616** (285/463), valid 0.788 — super-additive with CoT; the prior honest best
  and the base for all 2026-06-05 RL/DPO runs.

**2026-06-05** (RL on the 0.616 base + DPO; shared: batched multi-game roller — sample ~80 games in
parallel, right-pad causal, per-seq finish; `play` = greedy ephemeral-CoT; honest = TRAIN-secret labels
only, held-out greedy eval, no inference rules)

- **expert-iteration — `scripts/rl_expert_10row.py`** (`rl_expert.pt`). Per iter: sample **K=12**
  rollouts/secret (10-row, temp 0.9, 400 secrets/iter), keep ≤2 shortest wins (≤8 turns), **rebuild with
  CLEAN teacher think + the model's winning guesses** (RAFT — *not* the noisy sampled think, which
  degraded it), SFT (aux λ=0.5 + teacher-mix), lr 3e-5, 3 ep, **revert-on-regress**. Δ vs SFT: RL via
  self-play win distillation at 10 rows. held10 0.604→**0.646** (full-463 ≈ 6-row 0.62 / 10-row 0.637).
- **higher-ceiling / reachability — `scripts/rl_expert_tail.py`** (`rl_expert_tail.pt`). Pass 0 = **all
  1852 secrets, K=10, temp 1.0** (coverage); passes 1–3 = unsolved tail only, **K=24/32/48, temp
  1.1/1.2/1.3**; accumulate the union of clean-think wins, SFT, revert-on-regress. Δ vs expert-iter:
  full coverage + tail-focused high-K. **Reachability = 1843/1852 = 99.5%**; full-coverage SFT reverted
  (no gain) → coverage is not the bottleneck.
- **GRPO polish — `scripts/rl_grpo_polish.py`** (`rl_grpo.pt`). Token-level GRPO on the CoT policy:
  **G=8, 8 secrets/update**, advantage `A=r−mean(group)` (no ÷std), clip **ε=0.2**, **k3 KL to frozen
  ref β=0.05**, reward `win·(2+0.25·(10−t)) − 0.1·invalid`, lr **5e-6**, **eval-mode policy forward (no
  dropout)**. Δ vs the failed first try: eval-mode forward + β 0.01→0.05 + lr 1e-5→5e-6 (the first
  attempt's dropout-on forward made KL explode 12→294 and degraded). Result: stable but **flat** (6-row
  0.622, KL ~0.005 — policy barely moves).
- **self-consistency probe — `scripts/self_consistency.py`** (no training). Per turn, sample **N=12**
  traces (temp 0.9), commit the **majority-voted** guess (pure vote — no dict/filter); also pass@N (any
  of N sampled games wins). held 150, 6-row: greedy 0.607 · vote **0.627** · **pass@12 0.953**.
- **DPO, noisy pairs — `scripts/dpo_commit.py`** (`dpo.pt`). Sample N=8 rollouts/secret; pair at the
  **first win/loss divergence board** (chosen=winning think+guess, rejected=losing). DPO `−logσ(β·((logπ−logπ_ref)_chosen − (…)_rejected))`, **β=0.1**, lr 5e-6, ref=frozen base, 4 ep, response-token
  logp, revert-on-regress. Δ vs SFT: preference loss on commits. **Full-463 6-row 0.631** (+1.5);
  pref_acc 0.60→0.73 — capped by noisy credit.
- **DPO, decisive-board — `scripts/dpo_decisive.py`** (`dpo_decisive.pt`). Find decisive boards (last 2
  turns of winning rollouts, secret reachable), **resample M=14 responses/board**, pair **chosen =
  commits the secret** vs **rejected = valid + consistent + ≠ secret** at the same board (clean label).
  Same DPO (β=0.1, lr 5e-6, 5 ep, revert-on-regress). Δ vs dpo_commit: clean credit (decisive-board
  re-sampling) instead of noisy first-divergence pairs. **In progress.**

### Current standing

The honest best (6-row greedy, free-generation, no inference rules) is **DPO commit-sharpening = 0.631 held-out**
(`runs/dpo.pt`, `scripts/dpo_commit.py`), built on top of **ephemeral-CoT + aux-validity = 0.616**
(`runs/cot_eph_aux.pt`). The SFT base stacks two **orthogonal honest levers** super-additively
(no-CoT/no-aux 0.402 → +aux 0.436 → +CoT 0.430 → **+both 0.616**): the *ephemeral CoT scratchpad* (enumerate
candidates, commit, discard the reasoning — filter never at inference) for **search/strategy**, and the
*aux-validity trie loss* (dictionary baked into the weights; no trie at inference) for **spelling**. That
already **exceeds** the inference-aided beam+dict mark (0.58–0.60), honestly.

The remaining bottleneck is the **commit gap**: reachability is **99.5%** (sampling wins almost every train
secret) and **pass@12 = 0.95**, yet greedy commits wrong — so the knowledge is there; the model just doesn't
output it. Methods that *reweight sampled outcomes* can't move it: **GRPO is flat** (9 formulations now; the
stabilized run barely moves), and **self-consistency voting adds only +2pts** (the winning line is a minority,
not the mode). What helps is *sharpening the commit with clean training signal*: **expert-iteration** unlocked
the extra rows (10-row 0.604→0.637), and **DPO** is the first method to lift honest 6-row greedy
(0.616→0.631), currently limited by credit-assignment noise — which the **decisive-board DPO** run targets.

Dead ends remain dead: **BPE/real-text only wins by memorizing the answer set** (oreo-ai's 0.89 was train/test
contamination — reproduced as SEEN 0.87 / honest held-out 0.257), and **context management is a non-lever**
(explicit state hurt; length neutral). The standing **honesty rule**: answer-derived signals are fine in
*training* (teacher, engine-labeled wins, aux trie, DPO preference labels — all on TRAIN secrets); inference is
always **greedy on the strict held-out split with zero rules** — no dictionary, filter, candidate list, or
verifier.
