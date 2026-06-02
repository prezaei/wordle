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
