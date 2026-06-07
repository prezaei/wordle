# Experiment map — STYLE SAMPLE (2 example nodes + a fork)

This is a small sample to confirm the format before I build the full map. Each **box** = one experiment
(name · script · config · ─── · result · verdict). **Arrows** = forks (what was built *from* what).
Color = outcome. (Renders on GitHub.)

```mermaid
graph TD
  S1["<b>stage-1 · fair SFT</b> — cot_eph_aux_fair.pt<br/>50M · pretrain-30 + ephemeral-CoT + aux λ=1.0<br/>dictionary pools · answer-hood train-only · disjoint VAL/TEST<br/>━━━━━━━━━━<br/><b>TEST 0.281</b> win · 0.662 valid · avg 4.33<br/>📍 the honest base of everything"]:::base

  DPO["<b>DPO commit-sharpening</b> — dpo_fair.py<br/>from stage-1 · β=0.1 · 1,900 win/loss pairs · 4 epochs<br/>━━━━━━━━━━<br/>TEST 0.281 · valid 0.666<br/>❌ <b>NULL</b> — every epoch regressed → reverted to base"]:::null

  CD["<b>constrained-decode diagnostic</b> — constrained_decode_eval.py<br/>SAME stage-1 weights · greedy masked to the valid-word trie<br/>spelling-only (the model still does all the deduction)<br/>━━━━━━━━━━<br/><b>TEST 0.436</b> · valid 1.000<br/>🔍 <b>AIDED</b> — proves the model knows the words"]:::aided

  S1 -->|"preference RL"| DPO
  S1 -->|"inference-time aid"| CD

  classDef base  fill:#26323d,stroke:#7ab6c0,color:#e6f3f6,stroke-width:2px
  classDef null  fill:#3a2326,stroke:#b06b6b,color:#f6dada
  classDef aided fill:#22303c,stroke:#6699bb,color:#dcebf6
```

**Legend (outcome colors):** 🟦 baseline/SFT · 🟥 null (no gain) · 🟦(blue) aided/diagnostic · (full map
will add 🟩 genuine improvement). Emoji verdicts: ❌ null · 🔍 aided · 🎯 win · ⚠️ regressed/contaminated.

## What the full map will contain
~30–40 nodes across the whole thread, grouped into lanes: **foundations** (pretrain → char-SFT → CoT →
aux → DPO) · **honesty audits** (contamination → clean re-run → fair recipe) · **RL** (GRPO variants) ·
**validity push** (DAgger, distillation, info-gain XIT) · **scale sweep** (tiny/base/large/xl) ·
**inference** (constrained-decode, best-of-N N=16/64/128, beam). Every fork edge labeled with the
*decision* that spawned it (often a user steer, e.g. "make it genuinely generate words").
