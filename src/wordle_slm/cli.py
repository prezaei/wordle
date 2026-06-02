"""Command-line entrypoints for wordle-slm.

Each subcommand resolves a config (preset + ``--set`` overrides, Plan: K) and logs it; the
per-phase logic is implemented in its build-plan wave (docs/design/wordle-slm-plan.md).
"""

from __future__ import annotations

import argparse
import json
import logging

from wordle_slm.config.resolve import resolve, to_dict

logger = logging.getLogger(__name__)

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
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--preset", default="default", help="config preset name")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config field, e.g. --set grpo.group_size=16 (repeatable)",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = resolve(preset=args.preset, overrides=args.overrides)
    except (KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))  # clean usage error + exit 2, not a raw traceback
    logger.info(
        "resolved config for %s: %s", args.command, json.dumps(to_dict(cfg), sort_keys=True)
    )
    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
