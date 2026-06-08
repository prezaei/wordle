🧩 Wordle SLM — week experiment

Trained a ~50M transformer + RL (GRPO) from scratch to play Wordle, all local on an M5 Max. Wordle doesn't actually need an LLM — a few lines of information theory solves it optimally — so the transformer is deliberately the "wrong" tool. The point was hands-on reps with the real model-building stack (from-scratch training, SFT→RL, GRPO, reward shaping, honest held-out eval) on a small, fully-verifiable puzzle.

Best part was the process: an AI agent drove the research loop and ran adversarial self-audits on its own conclusions (every claim citing a file:line / tool receipt). It caught its own mistakes — found 4 data-leak channels and overturned its own "these are harmless" verdict when a clean re-run dropped the score 0.62 → 0.17. Effectively recursive self-improvement with a human in the loop: the AI reasons and self-critiques; I steer with the values and the hard questions.

Full write-up 👉 https://github.com/prezaei/wordle
