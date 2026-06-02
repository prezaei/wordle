"""Phase-0 run: baseline numbers + the speed/budget gate (spec §4.4-4.6, §6.6; Plan: L ★).

Measures, over the held-out set, the three calibration numbers the rest of the project is judged
against, plus the engine throughput that sizes the ~1-hour budget:

- **Random floor** over the answer pool (headline ~0.26% win) and over the full valid list (~0.05%).
- **Consistent yardstick** (~96-99% win) — the honest reference, not the pass bar.
- **Engine games/sec** (pure rollout, no model) — fed into the §6.6 budget formula to estimate how
  many GRPO updates fit the ~45-min RL window (and the largest group size G that still fits).

The budget estimate uses *engine* throughput (no model), an **optimistic upper bound** — the binding
number comes later from the model-rollout micro-benchmark (Plan: O). All numbers are logged.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass

from wordle_slm.baselines.policies import (
    DEFAULT_OPENER,
    ConsistentGuesser,
    RandomGuesser,
    play,
)
from wordle_slm.config import EvalConfig, GRPOConfig
from wordle_slm.telemetry.run_log import RunLog

logger = logging.getLogger(__name__)

# The §13 budget split: SFT caps at ~15 min, the RL window gets the rest of the ~1-hour cycle.
RL_BUDGET_MINUTES = 45.0
# A rough floor on the GRPO updates needed to show a learning curve (H-tagged judgment, §13);
# `fits` is informational — the DoD only requires #updates be computed and logged.
DEFAULT_MIN_UPDATES = 100


@dataclass(frozen=True)
class BaselineStats:
    """One baseline measured over a secret set."""

    name: str
    games: int
    wins: int
    win_rate: float
    avg_guesses_on_wins: float | None  # None when there are no wins
    win_distribution: dict[int, int]  # guesses_used -> count, over WON games only


@dataclass(frozen=True)
class BudgetEstimate:
    """Back-of-envelope #updates that fit the RL window (spec §6.6), at a given throughput."""

    games_per_sec: float
    rl_seconds: float
    rollout_batch: int  # secrets_per_update × G
    eval_games_per_update: float  # amortized two-tier eval overhead
    capacity_games: float  # throughput × rl_seconds
    n_updates: float
    min_updates: int
    fits: bool
    fitting_group_size: int  # the largest G that still yields >= min_updates


@dataclass(frozen=True)
class Phase0Report:
    floor_answers: BaselineStats
    floor_valid: BaselineStats
    yardstick_valid: BaselineStats  # spec §4.3: consistent over the valid list
    yardstick_answers: BaselineStats  # the v3 action-space floor: consistent over answers
    games_per_sec: float
    budget: BudgetEstimate


def measure_baseline(name: str, guesser, secrets: tuple[str, ...]) -> BaselineStats:
    """Play one game per secret with `guesser` (reused across the sweep) and tally the outcomes.

    The guesser is reused intentionally: a fresh seeded instance per game would replay the identical
    draw sequence and bias the estimate (see `RandomGuesser`). Each guesser plays over its own
    `default_pool`, so no pool argument is needed here.
    """
    wins = 0
    win_distribution: dict[int, int] = {}
    for secret in secrets:
        game = play(guesser, secret)
        if game.won:
            wins += 1
            win_distribution[game.guesses_used] = win_distribution.get(game.guesses_used, 0) + 1
    games = len(secrets)
    win_rate = wins / games if games else 0.0
    avg_guesses_on_wins = (
        sum(turn * count for turn, count in win_distribution.items()) / wins if wins else None
    )
    logger.info(
        "baseline %s: %d/%d wins (%.4f win rate), avg_guesses_on_wins=%s",
        name,
        wins,
        games,
        win_rate,
        f"{avg_guesses_on_wins:.4f}" if avg_guesses_on_wins is not None else "n/a",
    )
    return BaselineStats(name, games, wins, win_rate, avg_guesses_on_wins, win_distribution)


def engine_games_per_sec(
    secrets: tuple[str, ...], pool: tuple[str, ...], *, opener: str = DEFAULT_OPENER, seed: int = 0
) -> float:
    """Games/sec of a pure-engine rollout (consistent guesser over `pool`, no model).

    Times the consistent guesser — it exercises the same engine + incremental consistency filtering
    a real rollout does, minus the model scoring. `secrets` must be non-empty.
    """
    if not secrets:
        raise ValueError("need at least one secret to benchmark throughput")
    guesser = ConsistentGuesser(opener=opener, seed=seed, pool=pool)
    start = time.perf_counter()
    for secret in secrets:
        play(guesser, secret)
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        raise RuntimeError("benchmark elapsed time was non-positive; run more games")
    games_per_sec = len(secrets) / elapsed
    logger.info("engine throughput: %.1f games/sec over %d games", games_per_sec, len(secrets))
    return games_per_sec


def estimate_budget(
    games_per_sec: float,
    *,
    grpo: GRPOConfig,
    eval_cfg: EvalConfig,
    full_heldout: int,
    rl_minutes: float = RL_BUDGET_MINUTES,
    min_updates: int = DEFAULT_MIN_UPDATES,
) -> BudgetEstimate:
    """Solve the §6.6 budget model for #updates that fit the RL window, and the largest G that fits.

    §6.6 is circular (#updates ← eval_overhead ← #updates); it is linear, so it resolves directly:
    each update costs ``rollout_batch`` rollout games plus an amortized two-tier eval cost
    ``eval_per_update``, and ``n_updates = capacity / (rollout_batch + eval_per_update)``.
    """
    # Guard the denominators — a hostile override (e.g. --set eval.full_cadence=0) would otherwise
    # raise a bare ZeroDivisionError instead of a clear message.
    if eval_cfg.full_cadence <= 0 or eval_cfg.curve_cadence <= 0:
        raise ValueError("eval cadences must be positive")
    if grpo.secrets_per_update <= 0 or grpo.group_size <= 0:
        raise ValueError("grpo.secrets_per_update and group_size must be positive")
    if min_updates <= 0:
        raise ValueError("min_updates must be positive")
    rl_seconds = rl_minutes * 60.0
    rollout_batch = grpo.secrets_per_update * grpo.group_size
    # Amortized eval games per update: full held-out every full_cadence + 128-subsample every curve.
    eval_per_update = (
        full_heldout / eval_cfg.full_cadence + eval_cfg.curve_subsample / eval_cfg.curve_cadence
    )
    capacity_games = games_per_sec * rl_seconds
    n_updates = capacity_games / (rollout_batch + eval_per_update)
    if not math.isfinite(n_updates):
        raise FloatingPointError(f"non-finite #updates estimate: {n_updates}")
    # Largest G with n_updates >= min_updates: G ≤ (capacity/min_updates − eval_per_update)/secrets.
    max_batch = capacity_games / min_updates - eval_per_update
    fitting_group_size = max(0, math.floor(max_batch / grpo.secrets_per_update))
    estimate = BudgetEstimate(
        games_per_sec=games_per_sec,
        rl_seconds=rl_seconds,
        rollout_batch=rollout_batch,
        eval_games_per_update=eval_per_update,
        capacity_games=capacity_games,
        n_updates=n_updates,
        min_updates=min_updates,
        fits=n_updates >= min_updates,
        fitting_group_size=fitting_group_size,
    )
    logger.info(
        "budget @ %.1f games/sec: ~%.0f updates in %.0f min (batch=%d, eval/upd=%.1f) -> fits=%s "
        "(max G=%d for >=%d updates). NOTE engine throughput is an optimistic upper bound (model "
        "rollout is slower; see Plan O).",
        games_per_sec,
        n_updates,
        rl_minutes,
        rollout_batch,
        eval_per_update,
        estimate.fits,
        fitting_group_size,
        min_updates,
    )
    return estimate


def _log_report(run_log: RunLog, report: Phase0Report) -> None:
    run_log.log_scalar("phase0/floor_answers_win_rate", report.floor_answers.win_rate, 0)
    run_log.log_scalar("phase0/floor_valid_win_rate", report.floor_valid.win_rate, 0)
    run_log.log_scalar("phase0/yardstick_valid_win_rate", report.yardstick_valid.win_rate, 0)
    run_log.log_scalar("phase0/yardstick_answers_win_rate", report.yardstick_answers.win_rate, 0)
    avg = report.yardstick_answers.avg_guesses_on_wins
    run_log.log_scalar("phase0/yardstick_answers_avg_guesses", avg if avg is not None else 0.0, 0)
    run_log.log_scalar("phase0/games_per_sec", report.games_per_sec, 0)
    run_log.log_scalar("phase0/budget_n_updates", report.budget.n_updates, 0)
    run_log.log_scalar(
        "phase0/budget_fitting_group_size", float(report.budget.fitting_group_size), 0
    )
    run_log.log_transcript({"kind": "phase0_report", **_report_record(report)})


def _report_record(report: Phase0Report) -> dict:
    """JSON-ready report dict: dataclasses.asdict, with win_distribution keys stringified so the
    transcript schema is explicit (JSON has no integer keys — make the coercion deliberate)."""
    record = dataclasses.asdict(report)
    for key in ("floor_answers", "floor_valid", "yardstick_valid", "yardstick_answers"):
        dist = record[key]["win_distribution"]
        record[key]["win_distribution"] = {str(turn): count for turn, count in dist.items()}
    return record


def run_phase0(
    heldout: tuple[str, ...],
    answers: tuple[str, ...],
    valid: tuple[str, ...],
    *,
    grpo: GRPOConfig,
    eval_cfg: EvalConfig,
    run_log: RunLog | None = None,
    seed: int = 0,
    bench_games: int | None = None,
) -> Phase0Report:
    """Measure the floors, the yardstick, engine throughput, and the budget; log everything.

    `heldout` are the secrets to evaluate (spec §4.4 measures over held-out). `answers`/`valid` are
    the floor draw pools and the consistency universe. `bench_games` caps the throughput benchmark
    (defaults to the whole held-out set).
    """
    floor_answers = measure_baseline(
        "floor_answers", RandomGuesser(draw_pool=answers, seed=seed), heldout
    )
    floor_valid = measure_baseline(
        "floor_valid", RandomGuesser(draw_pool=valid, seed=seed), heldout
    )
    yardstick_valid = measure_baseline(
        "yardstick_valid", ConsistentGuesser(seed=seed, pool=valid), heldout
    )
    yardstick_answers = measure_baseline(
        "yardstick_answers", ConsistentGuesser(seed=seed, pool=answers), heldout
    )
    bench = heldout if bench_games is None else heldout[:bench_games]
    # Benchmark over the ANSWER pool: that is the v3 RL action space (the model rollout filters the
    # answers — spec §1.5), so this matches the engine cost of a real GRPO rollout.
    games_per_sec = engine_games_per_sec(bench, answers, seed=seed)
    budget = estimate_budget(games_per_sec, grpo=grpo, eval_cfg=eval_cfg, full_heldout=len(heldout))
    report = Phase0Report(
        floor_answers, floor_valid, yardstick_valid, yardstick_answers, games_per_sec, budget
    )
    if run_log is not None:
        _log_report(run_log, report)
    return report
