"""Model-rollout + update micro-benchmark — pin the real budget / group size (spec §4.5; Plan: O).

Phase-0 (`baselines.phase0`) sizes the budget from *engine* throughput and rollout games only — an
optimistic upper bound. This measures the **real cost on-device**: rollout games/sec, MPS memory,
and — crucially — the wall-time of one **full GRPO update** (rollout + teacher-forced recompute +
backward + step), which the §6.6 rollout-only formula misses but which dominates in practice. It
then projects #updates that fit the ~45-min RL window for each candidate group size G + recommends.

Our rollouts are sequential, so per-update time scales ~linearly with the rollout batch
(`secrets_per_update × G`); we time one reference update and scale. The benchmark runs on a *fresh*
model by default — generation/grad cost is weight-independent and a random model plays the maximum 6
guesses, giving a conservative (worst-case) estimate for safe budgeting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

import torch

from wordle_slm.baselines.phase0 import DEFAULT_MIN_UPDATES, RL_BUDGET_MINUTES
from wordle_slm.config import GRPOConfig, RewardConfig
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchRow:
    group_size: int
    rollout_games_per_sec: float
    peak_mem_mb: float
    seconds_per_update: float  # full update: rollout + recompute + backward + step
    n_updates: float  # that fit the RL window at this G
    fits: bool


def _on_mps(device: str) -> bool:
    return device == "mps" and torch.backends.mps.is_available()


def _sync(device: str) -> None:
    if _on_mps(device):
        torch.mps.synchronize()


def _mem_mb(device: str) -> float:
    return torch.mps.driver_allocated_memory() / 1e6 if _on_mps(device) else 0.0


def benchmark_rollout(
    model: WordleGenerator,
    tokenizer: Tokenizer,
    secrets: tuple[str, ...],
    *,
    n_rollouts: int = 16,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Time `n_rollouts` model rollouts → (games/sec, GPU-mem MB). Sequential play."""
    from wordle_slm.rl.rollout import play_game

    if not secrets:
        raise ValueError("need at least one secret to benchmark")
    model.eval()
    play_game(model, tokenizer, secrets[0], sample=True, generator=generator, device=device)  # warm
    _sync(device)
    start = time.perf_counter()
    for i in range(n_rollouts):
        play_game(
            model,
            tokenizer,
            secrets[i % len(secrets)],
            sample=True,
            generator=generator,
            device=device,
        )
    _sync(device)
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        raise RuntimeError("rollout benchmark elapsed time was non-positive; run more rollouts")
    return n_rollouts / elapsed, _mem_mb(device)


def benchmark_update(
    model: WordleGenerator,
    ref_model: WordleGenerator,
    tokenizer: Tokenizer,
    secrets: tuple[str, ...],
    *,
    grpo: GRPOConfig,
    reward: RewardConfig,
    ref_secrets_per_update: int = 2,
    ref_group_size: int = 4,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[float, int]:
    """Wall-time of ONE full GRPO update at a reference batch → (seconds, rollout_batch).

    `filter_zero_variance=False` forces the recompute+backward to run even on a fresh model (whose
    rollouts are all zero-variance), so the timing reflects the true update cost.
    """
    from wordle_slm.rl.grpo import grpo_update

    cfg = replace(
        grpo,
        group_size=ref_group_size,
        secrets_per_update=ref_secrets_per_update,
        filter_zero_variance=False,
    )
    sample = secrets[:ref_secrets_per_update]
    optimizer = torch.optim.AdamW(model.parameters(), lr=grpo.lr)
    kwargs = dict(
        grpo=cfg,
        reward=reward,
        optimizer=optimizer,
        group_size=ref_group_size,
        device=device,
        generator=generator,
    )
    grpo_update(model, ref_model, tokenizer, sample, **kwargs)  # warm
    _sync(device)
    start = time.perf_counter()
    grpo_update(model, ref_model, tokenizer, sample, **kwargs)
    _sync(device)
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        raise RuntimeError("update benchmark elapsed time was non-positive")
    return elapsed, ref_secrets_per_update * ref_group_size


def run_benchmark(
    model: WordleGenerator,
    ref_model: WordleGenerator,
    tokenizer: Tokenizer,
    secrets: tuple[str, ...],
    *,
    grpo: GRPOConfig,
    reward: RewardConfig,
    group_sizes: tuple[int, ...] = (4, 8, 16),
    device: str = "cpu",
    n_rollouts: int = 16,
    rl_minutes: float = RL_BUDGET_MINUTES,
    min_updates: int = DEFAULT_MIN_UPDATES,
) -> list[BenchRow]:
    """Measure rollout throughput + a real update wall-time; project honest #updates per G."""
    games_per_sec, mem = benchmark_rollout(
        model, tokenizer, secrets, n_rollouts=n_rollouts, device=device
    )
    ref_seconds, ref_batch = benchmark_update(
        model, ref_model, tokenizer, secrets, grpo=grpo, reward=reward, device=device
    )
    rl_seconds = rl_minutes * 60.0
    rows: list[BenchRow] = []
    for g in group_sizes:
        batch = grpo.secrets_per_update * g
        seconds_per_update = ref_seconds * batch / ref_batch  # ~linear in the rollout batch
        n_updates = rl_seconds / seconds_per_update
        rows.append(
            BenchRow(g, games_per_sec, mem, seconds_per_update, n_updates, n_updates >= min_updates)
        )
    logger.info(
        "benchmark: %.1f rollout games/sec, %.0f MB, %.2fs per update at batch %d",
        games_per_sec,
        mem,
        ref_seconds,
        ref_batch,
    )
    return rows


def recommend_group_size(rows: list[BenchRow]) -> int:
    """Largest G that fits ≥ min_updates (most rollouts/update); else the smallest tested G."""
    fitting = [r.group_size for r in rows if r.fits]
    return max(fitting) if fitting else min(r.group_size for r in rows)
