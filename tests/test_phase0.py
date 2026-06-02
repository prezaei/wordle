"""Phase-0 run + budget-gate tests (Plan: L ★).

Contracts: baselines tally wins/distribution correctly; the v3 (answer-pool) yardstick beats the
spec §4.3 (valid-pool) yardstick and both beat the floor; engine throughput is positive; the §6.6
budget formula resolves to the hand-computed #updates and the largest group size that fits.
"""

from __future__ import annotations

import json

import pytest

from wordle_slm.baselines.phase0 import (
    engine_games_per_sec,
    estimate_budget,
    measure_baseline,
    run_phase0,
)
from wordle_slm.baselines.policies import ConsistentGuesser, RandomGuesser
from wordle_slm.config import EvalConfig, GRPOConfig
from wordle_slm.data import load_answers, load_valid_guesses, split
from wordle_slm.telemetry.run_log import RunLog


def _heldout(n: int) -> tuple[str, ...]:
    _, heldout = split(seed=0)
    return heldout[:n]


# --- measure_baseline -------------------------------------------------------------------------


def test_measure_baseline_counts_a_guaranteed_win() -> None:
    secret = load_answers()[0]
    stats = measure_baseline("win", RandomGuesser(draw_pool=(secret,)), (secret,))
    assert stats.games == 1 and stats.wins == 1 and stats.win_rate == 1.0
    assert stats.win_distribution == {1: 1} and stats.avg_guesses_on_wins == 1.0


def test_measure_baseline_no_wins_has_none_average() -> None:
    answers = load_answers()
    secret = next(w for w in answers if w != "slate")  # 'slate' never solves it
    stats = measure_baseline("loss", RandomGuesser(draw_pool=("slate",)), (secret,))
    assert stats.wins == 0 and stats.win_rate == 0.0
    assert stats.avg_guesses_on_wins is None and stats.win_distribution == {}


def test_floor_win_rate_is_near_zero() -> None:
    secrets = _heldout(60)
    stats = measure_baseline("floor", RandomGuesser(draw_pool=load_answers(), seed=0), secrets)
    assert stats.win_rate <= 0.1  # the random floor is ~0.26%; a sample stays near zero


def test_v3_yardstick_beats_valid_yardstick_and_both_beat_floor() -> None:
    secrets = _heldout(60)
    answers, valid = load_answers(), load_valid_guesses()
    floor = measure_baseline("floor", RandomGuesser(draw_pool=answers, seed=0), secrets)
    y_valid = measure_baseline("y_valid", ConsistentGuesser(seed=0, pool=valid), secrets)
    y_answers = measure_baseline("y_answers", ConsistentGuesser(seed=0, pool=answers), secrets)
    # The v3 action space is consistent ANSWERS, so it wins more often than consistent-over-valid.
    assert y_answers.win_rate >= y_valid.win_rate > floor.win_rate


# --- engine_games_per_sec ---------------------------------------------------------------------


def test_engine_games_per_sec_is_positive() -> None:
    gps = engine_games_per_sec(_heldout(20), load_answers())
    assert gps > 0.0


def test_engine_games_per_sec_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one secret"):
        engine_games_per_sec((), load_answers())


# --- estimate_budget (the §6.6 formula) -------------------------------------------------------


def test_estimate_budget_matches_the_hand_computed_formula() -> None:
    budget = estimate_budget(100.0, grpo=GRPOConfig(), eval_cfg=EvalConfig(), full_heldout=463)
    # capacity = 100 * 45*60 = 270000; batch = 8*16 = 128;
    # eval/update = 463/200 + 128/25 = 7.435; n_updates = 270000 / 135.435.
    assert budget.capacity_games == pytest.approx(270000.0)
    assert budget.rollout_batch == 128
    assert budget.eval_games_per_update == pytest.approx(463 / 200 + 128 / 25)
    assert budget.n_updates == pytest.approx(270000.0 / (128 + 463 / 200 + 128 / 25))
    # max G with >=100 updates: floor((270000/100 - 7.435)/8) = 336.
    assert budget.fitting_group_size == 336
    assert budget.fits is True


def test_estimate_budget_flags_insufficient_throughput_with_a_real_shrink() -> None:
    # gps=1.0 -> capacity 2700, n_updates 2700/135.435 ≈ 19.9 (< 100): doesn't fit. The largest G
    # that WOULD fit is floor((2700/100 - 7.435)/8) = 2 — a concrete "shrink G from 16 to 2".
    budget = estimate_budget(1.0, grpo=GRPOConfig(), eval_cfg=EvalConfig(), full_heldout=463)
    assert budget.fits is False
    assert budget.n_updates < budget.min_updates
    assert budget.fitting_group_size == 2


def test_estimate_budget_rejects_nonpositive_cadence() -> None:
    with pytest.raises(ValueError, match="cadences must be positive"):
        estimate_budget(
            100.0, grpo=GRPOConfig(), eval_cfg=EvalConfig(full_cadence=0), full_heldout=463
        )


def test_estimate_budget_rejects_nonfinite_throughput() -> None:
    for bad in (float("inf"), float("nan")):
        with pytest.raises(FloatingPointError):
            estimate_budget(bad, grpo=GRPOConfig(), eval_cfg=EvalConfig(), full_heldout=463)


def test_estimate_budget_rejects_negative_full_heldout() -> None:
    with pytest.raises(ValueError, match="full_heldout must be non-negative"):
        estimate_budget(100.0, grpo=GRPOConfig(), eval_cfg=EvalConfig(), full_heldout=-1)


# --- run_phase0 -------------------------------------------------------------------------------


def test_run_phase0_returns_a_full_report() -> None:
    answers, valid = load_answers(), load_valid_guesses()
    report = run_phase0(
        _heldout(30), answers, valid, grpo=GRPOConfig(), eval_cfg=EvalConfig(), bench_games=15
    )
    assert report.floor_answers.games == 30 and report.floor_valid.games == 30
    assert report.yardstick_valid.games == 30 and report.yardstick_answers.games == 30
    assert report.games_per_sec > 0.0 and report.budget.n_updates > 0.0


def test_run_phase0_logs_scalars_and_a_report_record(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    answers, valid = load_answers(), load_valid_guesses()
    with RunLog(tmp_path / "p0", config={}, seed=0) as run_log:
        run_phase0(
            _heldout(20),
            answers,
            valid,
            grpo=GRPOConfig(),
            eval_cfg=EvalConfig(),
            run_log=run_log,
            bench_games=10,
        )
    acc = EventAccumulator(str(tmp_path / "p0" / "tb"))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    assert {
        "phase0/floor_answers_win_rate",
        "phase0/yardstick_valid_win_rate",
        "phase0/yardstick_answers_win_rate",
        "phase0/games_per_sec",
        "phase0/budget_n_updates",
    } <= tags
    lines = (tmp_path / "p0" / "transcripts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    report = records[-1]
    assert report["kind"] == "phase0_report"
    assert report["budget"]["rollout_batch"] == 128  # the report body serialized, not just the kind
    # win_distribution keys are deliberately stringified (JSON has no integer keys).
    assert all(isinstance(k, str) for k in report["yardstick_answers"]["win_distribution"])
    # Per-game transcripts are persisted (spec §4.4): a sampled subset, with guesses + feedback.
    games = [r for r in records if r["kind"] == "phase0_game"]
    assert games, "expected per-game transcripts"
    assert {r["baseline"] for r in games} == {
        "floor_answers",
        "floor_valid",
        "yardstick_valid",
        "yardstick_answers",
    }
    assert all("turns" in r and "guesses_used" in r for r in games)
