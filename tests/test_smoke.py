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
    assert cfg.model.n_layers == 4
    assert 1_000_000 <= cfg.model.estimated_params() <= 5_000_000  # spec: 1–5M range
    assert cfg.grpo.group_size == 16
    assert cfg.grpo.kl_estimator == "k3"
    assert cfg.grpo.advantage_norm == "mean_center"  # Dr. GRPO: no /std
    assert cfg.curriculum.tiers[-1] is None  # final tier = full train set


def test_reward_defaults_are_sane() -> None:
    # §6.4 shaped reward: the dominance inequalities must hold (invalid/clue beat honest progress;
    # a win beats the most you can farm).
    r = RunConfig().reward
    assert r.p_invalid > r.b and r.q > r.b
    assert 5 * r.a + 5 * r.b < r.win_base


@pytest.mark.parametrize("command", ["phase0", "sft", "rl", "eval", "play"])
def test_cli_subcommands_parse(command: str) -> None:
    ns = cli.build_parser().parse_args([command])
    assert ns.command == command


def test_cli_bad_preset_exits_cleanly() -> None:
    # A bad --preset must be a clean CLI error (SystemExit), not a raw traceback.
    with pytest.raises(SystemExit):
        cli.main(["phase0", "--preset", "does-not-exist"])
