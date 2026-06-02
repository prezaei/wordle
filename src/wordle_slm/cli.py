"""Command-line entrypoints for wordle-slm.

Each subcommand resolves a config (preset + ``--set`` overrides, Plan: K) and logs it; the
per-phase logic is implemented in its build-plan wave (docs/design/wordle-slm-plan.md).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from wordle_slm.config import RunConfig
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


def _run_phase0(cfg: RunConfig) -> int:
    """Phase-0 run (Plan: L): floor + yardsticks + engine throughput + budget over held-out."""
    from wordle_slm.baselines.phase0 import run_phase0
    from wordle_slm.data import load_answers, load_valid_guesses, split
    from wordle_slm.telemetry.run_log import RunLog

    answers = load_answers()
    valid = load_valid_guesses()
    _, heldout = split(seed=cfg.data.split_seed, train_frac=cfg.data.train_frac)
    with RunLog(Path(cfg.run_dir) / "phase0", config=to_dict(cfg), seed=cfg.seed) as run_log:
        report = run_phase0(
            heldout,
            answers,
            valid,
            grpo=cfg.grpo,
            eval_cfg=cfg.eval,
            run_log=run_log,
            seed=cfg.seed,
        )
    b = report.budget
    print(
        f"Phase 0 over {len(heldout)} held-out secrets:\n"
        f"  floor (answers): {report.floor_answers.win_rate * 100:.3f}%   "
        f"floor (valid): {report.floor_valid.win_rate * 100:.3f}%\n"
        f"  yardstick (valid, §4.3): {report.yardstick_valid.win_rate * 100:.2f}%   "
        f"yardstick (answers, v3 floor): {report.yardstick_answers.win_rate * 100:.2f}% "
        f"(avg {report.yardstick_answers.avg_guesses_on_wins:.3f} guesses)\n"
        f"  engine: {report.games_per_sec:.0f} games/sec   "
        f"budget: ~{b.n_updates:.0f} updates fit {b.rl_seconds / 60:.0f}-min RL at G="
        f"{cfg.grpo.group_size} (fits={b.fits}, max G={b.fitting_group_size}; engine is an "
        f"optimistic upper bound — model-rollout benchmark pins it in Plan O)"
    )
    return 0


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
    if args.command == "phase0":
        return _run_phase0(cfg)
    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
