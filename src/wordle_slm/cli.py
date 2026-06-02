"""Command-line entrypoints for wordle-slm.

Subcommands are stubs in S0; each is implemented in its build-plan wave
(docs/design/wordle-slm-plan.md).
"""

from __future__ import annotations

import argparse

_COMMANDS: dict[str, str] = {
    "phase0": "Phase 0: engine, data, baselines, floor/yardstick + speed budget",
    "sft": "Phase 1: imitation head-start (SFT) training",
    "rl": "Phase 2: GRPO reinforcement learning",
    "eval": "Evaluate a checkpoint on the held-out set",
    "play": "Play one live game in the terminal (Phase 4)",
}


def _not_implemented(name: str) -> int:
    print(f"`wordle-slm {name}` is not implemented yet — see docs/design/wordle-slm-plan.md")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wordle-slm", description="Wordle SLM toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in _COMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
