# AGENTS.md

Project context and working agreement for AI coding agents. Keep this file small — it loads into every agent's context on every task.

> **Stack:** Python 3.12 + PyTorch (Apple-Silicon **MPS** backend). This repo trains a small
> language model from scratch and uses RL (GRPO) to play Wordle. Runs **locally** on the M5 Max;
> no external services. See `PRD.md` (product) and `docs/design/wordle-slm.md` (spec).

## Working Style

- **Operate autonomously.** Prefer making reasonable choices over stopping to ask.
- **Fix ALL failures, not just the first.** Tests, lint, type checks — fix everything. Iterate until green.
- **Stay focused** on the current component. Reference other components for context only.
- **Cross-component work**: only when explicitly asked.

## Operating Guidelines

| Situation | Approach |
| --------- | -------- |
| Normal | Answer first, explain after |
| Uncertain | Confidence → best guess → what resolves it |
| Stuck | State it → what tried → what needed |
| Disagree | Directly with alternatives |

**Avoid:** apologetic hedging, asking permission for trivial actions, silent failures.

**Autonomy boundaries:** Autonomy applies to local, reversible actions (reading code, running tests, editing files, pushing to your own branch). These actions **always require explicit user instruction**:

- Merging PRs
- Force-pushing or deleting shared branches
- Deploying, releasing, or promoting builds
- Any action that could reach users or break the team's workflow

**Prohibitions:** never commit secrets, bypass tests, force push to shared branches, hide failures.

**Refactoring bias:** deletion and compaction worth 2x additions (if no data loss). Prefer removing code over adding.

## Project

- **Host:** github.com (private)
- **CI gating:** all CI checks must pass before merge
- **Tool:** `gh` CLI
- **Default branch:** `main`
- **Issues:** GitHub Issues
- **Merge strategy:** squash — single commit per PR
- **Branch naming:** feature branches off `main`
- **Package manager:** **uv** (with `uv.lock` committed)
- **Do NOT** depend on unreleased/experimental APIs without discussion

## Skills

- **Proactive routing.** Invoke skills on matching intent — the user doesn't have to type the slash command.
- **Design first.** Run `architect-and-design` before non-trivial feature work.

**Research first:** When debugging or implementing features that touch external dependencies, research their documentation BEFORE forming hypotheses.

**Execution:** Run independent ops in parallel; sequential only when output depends on prior.

## Repo Invariants

- `ruff check` must pass (lint clean — warnings treated as errors)
- `ruff format --check` must pass (formatting enforced)
- Full **type hints** on public functions/classes
- **No swallowed exceptions** — raise or handle explicitly; never `except:`/silent failure
- **No `unsafe` numerics silently** — no NaN/inf hidden; assert on critical tensor state
- Layout: `src/wordle_slm/` package; `tests/` mirrors it
- **Determinism:** fixed seeds for runs; note MPS is only *approximately* reproducible (document, don't pretend bit-exact)

## Prerequisites

| Need | Why |
|------|-----|
| Python 3.12+ and `uv` | Build, deps, run |
| Apple-Silicon Mac (MPS) | Train on the GPU |
| `gh` CLI authenticated to your git host | PRs, issues |

Bootstrap: `uv sync && uv run pytest`

## PR Quality Checklist

```bash
# Run everything
uv run ruff format --check && uv run ruff check && uv run pytest

# Individual gates
uv run ruff format --check   # formatting
uv run ruff check            # lint
uv run pytest                # tests
```

Every PR must also pass:
- **Telemetry review:** every new function, decision, and error path has info-level logging.
- **Test coverage:** every behavior in the design has a test. New code without tests is incomplete code.

## Testing Guardrails

- Tests required for new code paths.
- No hardcoded ports/paths — use temp dirs / dynamic allocation.
- No global mutable state — tests run in parallel by default.
- Scoped test runs: `uv run pytest tests/<area>` or `pytest -k <pattern>`.
- **Exact expected values.** Engine color-scoring (duplicate letters) and the reward function assert the exact output, not a paraphrase.
- **Every enum variant through every entry point.** If an API accepts an enum (e.g. `Color`), every variant gets a test through every path. Not one variant that happens to work.
- **Test names are contracts.** A mismatch between test name and test body is a bug.
- **No optimizations.** An optimization is a second codepath. A second codepath hides bugs. One path.
- **Design → tests → code.** Always in that order. Tests define the contract. Code fulfills it.

## Telemetry

**Every activity in the system is logged. Every event, every outcome, every decision. No exceptions.** If something happens and there's no log, that's a critical gap. If you can't diagnose a problem from the logs alone, the telemetry is insufficient. (This project: TensorBoard scalars + a structured JSON run log; see `docs/design/wordle-slm.md` §8.)

## Keeping this file small

This file is loaded into every AI agent's context on every task. Every line is a tax.

- Invariants live inline (this file). Patterns and reference material live in `docs/*`.
- Component-specific context lives in `<component>/AGENTS.md`.
- No enumeration of components, design docs, or subsystem files.
- **Size budget:** root ≤ 200 lines (warn at 180).
