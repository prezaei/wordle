"""End-to-end CLI run tests (Plan: U) — sft → rl wired through `main`, tiny model on CPU."""

from __future__ import annotations

from wordle_slm import cli

# A tiny model + CPU so the end-to-end run is fast.
_TINY = [
    "--device",
    "cpu",
    "--set",
    "model.d_model=64",
    "--set",
    "model.n_layers=2",
    "--set",
    "model.n_heads=4",
    "--set",
    "model.d_ff=256",
]


def test_sft_then_rl_end_to_end(tmp_path) -> None:
    common = [*_TINY, "--set", f"run_dir={tmp_path}"]
    assert cli.main(["sft", "--limit", "6", "--epochs", "2", *common]) == 0
    assert (tmp_path / "sft.pt").exists()  # the checkpoint RL will load
    assert (tmp_path / "sft" / "run.json").exists()

    rl_args = [
        "rl",
        "--limit",
        "6",
        "--updates",
        "2",
        "--set",
        "grpo.group_size=4",
        "--set",
        "grpo.secrets_per_update=2",
        "--set",
        "eval.curve_subsample=6",
        "--set",
        "eval.curve_cadence=2",
        *common,
    ]
    assert cli.main(rl_args) == 0
    assert (tmp_path / "rl" / "run.json").exists()

    # the Phase-1 readiness gate runs and returns a pass/fail exit code (tiny model → likely fail)
    assert cli.main(["eval", "--limit", "5", *common]) in (0, 1)


def test_rl_without_a_checkpoint_errors_cleanly(tmp_path) -> None:
    # No SFT checkpoint present -> exit 1 with a helpful message, no traceback.
    assert cli.main(["rl", "--device", "cpu", "--set", f"run_dir={tmp_path}"]) == 1
