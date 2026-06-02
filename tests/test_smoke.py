"""S0 smoke tests: the package imports, configs instantiate, the CLI parses.

These are scaffolding checks only — real behavior is tested per the build plan.
"""

from __future__ import annotations

import pytest

from wordle_slm import cli
from wordle_slm.config import RunConfig


def test_package_imports() -> None:
    import wordle_slm

    assert wordle_slm.__version__


def test_run_config_defaults() -> None:
    cfg = RunConfig()
    # spot-check defaults mirror spec §13
    assert cfg.model.d_model == 256
    assert cfg.grpo.group_size == 16
    assert cfg.grpo.kl_estimator == "k3"
    assert cfg.grpo.advantage_norm == "mean_center"  # Dr. GRPO: no /std
    assert cfg.curriculum.tiers[-1] is None  # final tier = full train set


def test_reward_dominance_preconditions() -> None:
    # These must hold for the reward not to be gameable (asserted again in step H).
    r = RunConfig().reward
    assert r.invalid_penalty > r.yellow
    assert r.clue_penalty > r.yellow


@pytest.mark.parametrize("command", ["phase0", "sft", "rl", "eval", "play"])
def test_cli_subcommands_parse(command: str) -> None:
    ns = cli.build_parser().parse_args([command])
    assert ns.command == command
