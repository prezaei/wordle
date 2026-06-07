"""Generate the full experiment-lineage Graphviz map (data-driven).

Edit the DATA below; run `uv run python scripts/build_experiment_map.py` to emit EXPERIMENTS.dot,
then `dot -Tpng -Gdpi=140 EXPERIMENTS.dot -o EXPERIMENTS.png`. Boxes = experiments (title · model
size · recipe · result); colour = era/type; edges = the decision that forked each. No filenames, concise.
"""

from __future__ import annotations

# era/type -> header colour (the result row is always dark with the number + a tag)
ERA = {
    "base": "#cfcfcf",       # baseline / floor
    "cont": "#d9c89a",       # contaminated-lineage (numbers inflated by held-out leakage)
    "clean": "#a9d3d9",      # clean / leak-free
    "head": "#7ec8d0",       # clean headline
    "audit": "#e8c98a",      # audit / pivot
    "aided": "#aac4dd",      # inference-time aided (on the SAME weights, no training)
    "best": "#7fa8cf",       # best aided
    "run": "#d0d0d0",        # running / pending
}
# nodes: id -> (lane, title, size, recipe, result, era)
N = {
  # ---- foundations: SFT scale ladder ----
  "sft_3m":   ("found", "char model, plain SFT", "≈3M", "writes a guess letter-by-letter; spell warm-up, then imitates a near-optimal solver", "0.205 · valid 0.54", "base"),
  "sft_5m":   ("found", "deeper SFT + cosine", "≈5M", "deeper net, deeper warm-up, more teacher data", "0.300 · valid 0.61", "cont"),
  "sft_25m":  ("found", "strong-teacher SFT", "≈25M", "5x bigger, 80% near-optimal teacher imitation", "0.391 · valid 0.66", "cont"),
  "sft_25m_div": ("found", "diverse-secret SFT", "≈25M", "train on full valid-word secrets (not just answers) to stop memorizing", "0.220 (gap collapsed)", "cont"),
  "scale_99m": ("found", "pure-scale test", "99M", "the 25M recipe scaled up, old answer-only data", "plateaued / under-converged", "cont"),
  "codesign_50m": ("found", "co-design + curriculum", "≈50M", "redesigned net + difficulty-ordered diverse curriculum", "0.188 (rare-word dilution)", "cont"),
  "sft_deep": ("found", "deep SFT, converged", "≈50M", "redesigned net on strong-teacher answers, trained to convergence; no CoT/aux", "0.402 · valid 0.664", "cont"),
  "sft_aux":  ("found", "+ spelling helper (aux)", "≈50M", "training-only loss pushing each letter toward real-word paths; no dict at play", "0.436 · valid 0.675", "cont"),
  "sft_aux_xl": ("found", "scale + aux", "98M", "scale and aux stacked, wall-clock capped", "superseded (no milestone)", "cont"),
  # ---- foundations: CoT thread ----
  "cot_14m":  ("cot", "CoT prototype", "≈14M", "writes a short <think> candidate list, then commits", "no-CoT 0.155 vs CoT 0.415", "cont"),
  "cot_50m":  ("cot", "CoT scaled", "≈50M", "scale the winning CoT with teacher reasoning traces", "0.456 (later → ~0.192)", "cont"),
  "cot_50m_aux": ("cot", "CoT + aux", "≈50M", "stack spelling helper on CoT", "incomplete / superseded", "cont"),
  "cot_show": ("cot", "CoT integrity teardown", "≈50M", "A/B: teacher-context (rebuilds past think) vs self-context", "0.450 vs 0.192 — LEAK found", "audit"),
  "cot_eph":  ("cot", "ephemeral CoT", "≈50M", "throwaway scratchpad: think regenerated each turn, discarded at play; board-only history", "0.430 · valid 0.671", "cont"),
  "cot_eph_aux": ("cot", "ephemeral CoT + aux", "≈50M", "stack search (CoT) + spelling (aux); long schedule", "0.616 · valid 0.788 (era best)", "cont"),
  # ---- validity / tokenization probes ----
  "bpe_12m":  ("val", "BPE tokenizer", "≈12M", "from-scratch subword tokenizer on the word list; guesses as letter-chunks", "0.212 · valid 0.66→0.85", "cont"),
  "bpe_50m":  ("val", "BPE tokenizer, scaled", "≈50M", "same BPE recipe, bigger", "0.188 · valid 0.806 (win flat)", "cont"),
  "oreo_11m": ("val", "real-text pretrain", "≈11M", "TinyStories byte-BPE pretrain, then SFT on game transcripts", "0.257 / seen 0.87", "cont"),
  "oreo_50m": ("val", "real-text pretrain, scaled", "≈50M", "same recipe, bigger, more passes", "0.190 (overfits)", "cont"),
  "structured_ctx": ("val", "structured-context A/B", "≈50M", "raw board vs board + explicit greens/present/absent block", "raw 0.260 vs +state 0.170", "cont"),
  # ---- RL family (contaminated bases) ----
  "grpo_5m":  ("rl", "GRPO #1", "4.8M", "RL over the train set, loose KL, to move greedy play", "~0.29 null (memorizes train)", "cont"),
  "grpo_5m_div": ("rl", "GRPO #2 diverse", "≈5M", "GRPO over 14k non-memorizable secrets", "~0.27, reward negative", "cont"),
  "self_distill_25m": ("rl", "self-distill beam+dict→greedy", "≈25M", "SFT on its own always-valid beam+dict games", "0.384 (spelling up, win flat)", "cont"),
  "rl_consistency": ("rl", "GRPO #3 legality reward", "≈25M", "flat reward for legal/consistent guesses", "0.384 null", "cont"),
  "rl_perguess": ("rl", "GRPO #4 per-guess", "≈25M", "episode = one guess, clean per-guess credit", "0.384 null", "cont"),
  "rl_dict":  ("rl", "GRPO #5 dict-in-loop", "≈25M", "behavior policy samples trie-valid words, push free-gen toward high-adv", "0.384 null", "cont"),
  "rl_constrained": ("rl", "GRPO #6 consistency (decisive)", "≈25M", "behavior samples the still-consistent set (answer surfaced ~22%)", "0.389 null — barrier is generalization", "cont"),
  "rl_polish": ("rl", "GRPO #7 polish", "≈50M", "validity+consistency reward on the 0.436 base, revert-on-regress", "0.436 unchanged", "cont"),
  "rl_infogain": ("rl", "GRPO #8 info-gain", "≈50M", "add the information-gain reward term + 12 train guesses", "0.436 unchanged (RL closed)", "cont"),
  "rl_expert_10row": ("rl", "expert-iteration (10-row)", "≈50M", "keep winning rollouts, rebuild with clean think, SFT; teach rows 7–10", "0.604→0.646 (the RL that works)", "cont"),
  "rl_expert_tail": ("rl", "reachability expert-iter", "≈50M", "full coverage + tail high-K/high-temp sampling", "solved 99.5%; SFT reverted", "cont"),
  "rl_grpo_polish": ("rl", "GRPO #9 token-level", "≈50M", "stabilized token-GRPO on the CoT policy (eval-mode forward, k3 KL)", "6r 0.622 / 10r 0.637 (flat)", "cont"),
  "dpo_commit": ("rl", "DPO commit-sharpening", "≈50M", "DPO on win/loss first-divergence commit pairs; raise P(winning commit)", "0.631 (era best, +1.5)", "cont"),
  "dpo_decisive": ("rl", "DPO decisive-board", "≈50M", "clean labels: secret-commit vs wrong consistent word at the same board", "flat → reverted (think dilutes)", "cont"),
  "dpo_guessonly": ("rl", "DPO guess-only", "≈50M", "score preference on only the 5 committed letters", "knife-edge: flat or collapses", "cont"),
  "constraint_aux": ("rl", "constraint-aux fine-tune", "≈50M", "extra aux: keep greens / reuse yellows / no repeats", "no-op (zero gradient, OOD) → reverted", "cont"),
  "dagger_v1": ("rl", "DAgger v1", "≈50M", "relabel its bad boards with teacher's word, SFT (corrections ~1:6)", "reverted (under-weighted)", "cont"),
  "rl_grpo_reward": ("rl", "GRPO #10 + shaped reward", "≈50M", "token-GRPO with the full updated reward (repeat/drop-present)", "greedy DECLINED 0.615→0.583 (proxy-hack)", "cont"),
  "rl_grpo_guessonly": ("rl", "GRPO guess-only credit", "≈50M", "credit only the 5 guess letters", "KL EXPLODED 0.01→12.7", "cont"),
  "dagger_v2": ("rl", "DAgger v2 (full coverage)", "≈50M", "all secrets, corrections upweighted ×4", "null (later clean → 0.14–0.17)", "cont"),
  # ---- inference probes (no training) ----
  "beam_dict": ("inf", "beam + dictionary decode", "≈25M", "beam search ± constrained to real words via trie (no training)", "greedy 0.392 → beam+dict 0.580", "aided"),
  "norepeat_decode": ("inf", "beam+dict + no-repeat", "≈25M", "beam(10) + dict-trie + never re-emit a prior guess", "0.596", "aided"),
  "turn_budget": ("inf", "turn-budget probe", "≈25M", "same weights, allow 6/8/10 guesses", "0.392 flat", "aided"),
  "passk": ("inf", "pass@N probe", "≈50M", "sample N full games/secret, count any win", "greedy 0.453 → pass@10 0.787", "aided"),
  "self_consistency": ("inf", "self-consistency vote", "≈50M", "sample 12 traces/turn, commit the majority vote (no filter)", "vote 0.627 vs pass@12 0.953", "aided"),
  # ---- audit / pivot ----
  "audit": ("audit", "4-agent adversarial audit", "—", "Lead/Code/Telemetry/Devil's-Advocate, every claim a file:line receipt", "library OK; pipeline contaminated", "audit"),
  "clean_rerun": ("audit", "clean re-run (leak-free)", "≈50M", "the 0.616 recipe with train-only pools, disjoint VAL/TEST", "0.616 → 0.166 — OVERTURNS", "clean"),
  "overnight_clean_sft": ("audit", "overnight clean SFT", "≈50M", "the clean ephemeral-CoT+aux base (3-seed)", "0.166 (strict held-out)", "clean"),
  "overnight_clean_dpo": ("audit", "clean DPO", "≈50M", "DPO on the clean base", "0.166 null (gain was contamination)", "clean"),
  "overnight_clean_grpo": ("audit", "clean GRPO", "≈50M", "stabilized GRPO + full reward on clean base", "0.166 null (11th GRPO)", "clean"),
  "overnight_clean_dagger": ("audit", "clean DAgger", "≈50M", "failure-state relabel on clean base", "0.144 (hurts)", "clean"),
  "overnight_clean_dagger2": ("audit", "clean DAgger ×2", "≈50M", "full-coverage DAgger ×4 corrections", "0.169", "clean"),
  # ---- scale / fair re-build ----
  "fair_stage1": ("scale", "FAIR model · STAGE-1", "≈50M", "candidate/teacher pools = full dictionary (knows spelling, not answer-hood); pretrain 30 + aux λ=1.0", "0.281 · valid 0.662 · avg 4.33", "head"),
  "distill_constrained": ("scale", "constrained self-distillation", "≈50M", "imitate its own dictionary-constrained rollouts (+aux), eval free-gen", "valid 0.62→0.80, win flat (traded)", "clean"),
  "infogain_xit_c": ("scale", "info-gain XIT (constrained)", "≈50M", "constrained rollouts (wheel), keep high-info-gain turns, SFT free-gen", "0.281 null", "clean"),
  "infogain_xit_f": ("scale", "info-gain XIT (free / STaR)", "≈50M", "same, free-gen rollouts (no wheel)", "0.281 null", "clean"),
  "dpo_fair": ("scale", "DPO on fair base", "≈50M", "the contaminated-winning DPO recipe, clean base", "0.281 null (every epoch reverted)", "clean"),
  "grpo_full_fair": ("scale", "stage-4 long GRPO", "≈50M", "stabilized GRPO + full reward from the distilled base, 150 updates", "flat ~0.33 null", "clean"),
  "scale_tiny": ("scale", "scale — tiny", "1.2M", "fair recipe, smallest net", "0.163 (underfits)", "clean"),
  "scale_base": ("scale", "scale — base", "12M", "fair recipe, mid net", "0.251 (gap grows)", "clean"),
  "scale_xl": ("scale", "scale — xl", "99M", "fair recipe, largest net", "0.270 · valid 0.591 — turns over (< 50M)", "clean"),
  # ---- inference on the clean fair weights ----
  "constrained_decode": ("inf2", "constrained-decode diagnostic", "≈50M", "greedy masked to real-word spellings; model still deduces", "0.281 → 0.436 · valid 1.0 — KNOWS the words", "aided"),
  "bestof16": ("inf2", "best-of-16 (valid vote)", "≈50M", "sample 16, keep real words, majority vote", "0.632 · valid 0.925", "aided"),
  "bestof16_nodict": ("inf2", "best-of-16 NO dict", "≈50M", "same, but keep non-words too", "0.243 — compute alone HURTS", "aided"),
  "bestof64": ("inf2", "best-of-64", "≈50M", "N=64", "0.703", "aided"),
  "bestof128": ("inf2", "best-of-128 — BEST", "≈50M", "N=128", "0.719 · valid 0.979 (plateau)", "best"),
  "beam_trie": ("inf2", "beam over real-word trie", "≈50M", "sequence-level argmax over valid words", "queued", "run"),
  # ---- deployed ----
  "deployed": ("dep", "deployed real-Wordle player", "≈50M", "trained on all 2,315 known answers (fixed set = the real game)", "~0.62 (legit deployed, framing)", "cont"),
}
LANES = {
  "found": "FOUNDATIONS — SFT scale ladder (contaminated lineage)",
  "cot":   "FOUNDATIONS — chain-of-thought thread",
  "val":   "TOKENIZER / REPRESENTATION PROBES",
  "rl":    "REINFORCEMENT LEARNING (×10 GRPO + DPO + DAgger, on contaminated bases)",
  "audit": "AUDIT → CLEAN RE-RUN (the honesty pivot)",
  "scale": "FAIR RE-BUILD + SCALE SWEEP (clean)",
  "inf":   "INFERENCE PROBES (decoding, contaminated bases)",
  "inf2":  "INFERENCE on the CLEAN fair weights (aided)",
  "dep":   "DEPLOYED FRAMING",
}
EDGES = [
  ("sft_3m","sft_5m","scale+depth"),("sft_5m","sft_25m","5x params"),("sft_25m","sft_25m_div","diverse secrets"),
  ("sft_25m","scale_99m","pure scale"),("scale_99m","codesign_50m","redesign+curriculum"),("codesign_50m","sft_deep","isolate redesign, converge"),
  ("sft_deep","sft_aux","add spelling loss"),("sft_aux","sft_aux_xl","scale aux"),
  ("sft_3m","bpe_12m","swap to BPE"),("bpe_12m","bpe_50m","capacity test"),("oreo_11m","oreo_50m","scale"),("sft_aux","structured_ctx","add state block"),
  ("sft_aux","passk","pass@N probe"),("sft_25m","beam_dict","beam+dict"),("sft_25m","norepeat_decode","+no-repeat"),("sft_25m","turn_budget","more guesses"),
  ("passk","cot_14m","reason to surface latent"),("cot_14m","cot_50m","scale CoT"),("cot_50m","cot_50m_aux","+aux"),("cot_50m","cot_show","integrity A/B"),
  ("cot_show","cot_eph","throwaway scratchpad (fix leak)"),("sft_deep","cot_eph","matched baseline"),("cot_eph","cot_eph_aux","stack +aux"),("sft_aux","cot_eph_aux","the spelling lever"),
  ("sft_5m","grpo_5m","RL?"),("grpo_5m","grpo_5m_div","diverse"),("sft_25m","self_distill_25m","bank spelling"),
  ("self_distill_25m","rl_consistency","legality reward"),("self_distill_25m","rl_perguess","per-guess credit"),("self_distill_25m","rl_dict","dict-in-loop"),("self_distill_25m","rl_constrained","surface the answer"),
  ("sft_aux","rl_polish","polish best"),("sft_aux","rl_infogain","info-gain term"),
  ("cot_eph_aux","rl_expert_10row","expert-iter"),("rl_expert_10row","rl_expert_tail","reachability"),("rl_expert_10row","rl_grpo_polish","token-GRPO"),
  ("cot_eph_aux","self_consistency","vote vs pass@N"),("cot_eph_aux","dpo_commit","DPO commit"),("cot_eph_aux","dpo_decisive","clean pairs"),("dpo_commit","dpo_guessonly","guess-only"),
  ("cot_eph_aux","constraint_aux","train-in reward"),("dpo_commit","dagger_v1","failure relabel"),("dpo_commit","rl_grpo_reward","GRPO+reward"),("dpo_commit","rl_grpo_guessonly","guess-only credit"),("dagger_v1","dagger_v2","full coverage"),
  ("cot_eph_aux","audit","is it correct?"),("dpo_commit","audit","audit 0.631 too"),("cot_eph_aux","clean_rerun","re-run leak-free"),
  ("clean_rerun","overnight_clean_sft","clean base"),("overnight_clean_sft","overnight_clean_dpo","DPO"),("overnight_clean_sft","overnight_clean_grpo","GRPO"),("overnight_clean_sft","overnight_clean_dagger","DAgger"),("overnight_clean_dagger","overnight_clean_dagger2","×2"),
  ("overnight_clean_sft","fair_stage1","know dict, hide answers"),
  ("fair_stage1","distill_constrained","push validity"),("fair_stage1","infogain_xit_c","info-gain XIT"),("fair_stage1","infogain_xit_f","free / STaR"),("fair_stage1","dpo_fair","retry DPO clean"),("distill_constrained","grpo_full_fair","long GRPO"),
  ("fair_stage1","scale_tiny","shrink"),("fair_stage1","scale_base","mid"),("fair_stage1","scale_xl","grow"),
  ("fair_stage1","constrained_decode","mask spelling (diagnostic)"),("fair_stage1","bestof16","test-time compute"),("fair_stage1","bestof16_nodict","no-dict ablation"),("bestof16","bestof64","N=64"),("bestof64","bestof128","N=128"),("fair_stage1","beam_trie","beam"),
  ("cot_eph_aux","deployed","deployed framing"),("dpo_commit","deployed","best deployed"),
]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def node_dot(nid, title, size, recipe, result, era):
    head = ERA[era]
    rcol = "#1f5e66" if era in ("head", "best") else "#4a4a4a"
    size_disp = "method — no model" if size == "—" else f"{esc(size)} params"
    return (f'  {nid} [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">'
            f'<TR><TD BGCOLOR="{head}"><B>{esc(title)}</B></TD></TR>'
            f'<TR><TD BGCOLOR="#efefef"><FONT POINT-SIZE="9" COLOR="#222222"><B>{size_disp}</B></FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">{esc(recipe)}</FONT></TD></TR>'
            f'<TR><TD BGCOLOR="{rcol}"><FONT COLOR="white" POINT-SIZE="10"><B>{esc(result)}</B></FONT></TD></TR>'
            f'</TABLE>>];')


def main():
    out = ['digraph experiments {', '  rankdir=TB; bgcolor="white";',
           '  graph [fontname="Helvetica", ranksep=1.2, nodesep=0.65, fontsize=13, '
           'concentrate=true, splines=spline, pad=0.4];',
           '  node [shape=plaintext, fontname="Helvetica"];',
           '  edge [fontname="Helvetica", fontsize=8, color="#888888", penwidth=1.1, arrowsize=0.7];']
    for lane, lname in LANES.items():
        out.append(f'  subgraph cluster_{lane} {{ label=<<B>{esc(lname)}</B>>; fontsize=13; '
                   f'color="#cccccc"; style="rounded"; bgcolor="#fbfbfb";')
        for nid, (ln, title, size, recipe, result, era) in N.items():
            if ln == lane:
                out.append("  " + node_dot(nid, title, size, recipe, result, era))
        out.append("  }")
    for a, b, lbl in EDGES:
        out.append(f'  {a} -> {b} [label="{esc(lbl)}"];')
    out.append("}")
    with open("EXPERIMENTS.dot", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote EXPERIMENTS.dot · {len(N)} nodes · {len(EDGES)} edges")


if __name__ == "__main__":
    main()
