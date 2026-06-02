"""Model-rollout + update benchmark tests (Plan: O; spec §4.5)."""

from __future__ import annotations

import pytest
import torch

from wordle_slm.baselines.benchmark import (
    BenchRow,
    benchmark_rollout,
    benchmark_update,
    recommend_group_size,
    run_benchmark,
)
from wordle_slm.config import GRPOConfig, ModelConfig, RewardConfig
from wordle_slm.data import load_answers
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.rl.tracer import make_reference

_CFG = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)


def _model_ref_tok():
    torch.manual_seed(0)
    tok = Tokenizer()
    model = WordleGenerator(_CFG, tok.vocab_size)
    return model, make_reference(model), tok


def test_benchmark_rollout_positive_throughput() -> None:
    model, _, tok = _model_ref_tok()
    gps, mem = benchmark_rollout(model, tok, load_answers()[:4], n_rollouts=4, device="cpu")
    assert gps > 0.0 and mem == 0.0  # CPU: GPU mem not tracked


def test_benchmark_update_times_a_real_update() -> None:
    model, ref, tok = _model_ref_tok()
    seconds, batch = benchmark_update(
        model,
        ref,
        tok,
        load_answers()[:4],
        grpo=GRPOConfig(),
        reward=RewardConfig(),
        ref_secrets_per_update=2,
        ref_group_size=4,
        device="cpu",
    )
    assert seconds > 0.0 and batch == 8  # 2 secrets × group 4


def test_run_benchmark_scales_linearly_with_group_size() -> None:
    model, ref, tok = _model_ref_tok()
    rows = run_benchmark(
        model,
        ref,
        tok,
        load_answers()[:4],
        grpo=GRPOConfig(secrets_per_update=8),
        reward=RewardConfig(),
        group_sizes=(4, 8),
        device="cpu",
        n_rollouts=4,
    )
    assert [r.group_size for r in rows] == [4, 8]
    # per-update time is ~linear in the rollout batch (G=8 is 2× G=4): 2× time, half the updates
    assert rows[1].seconds_per_update == pytest.approx(2 * rows[0].seconds_per_update)
    assert rows[1].n_updates == pytest.approx(rows[0].n_updates / 2)
    assert all(r.rollout_games_per_sec > 0 for r in rows)


def test_recommend_group_size_picks_largest_fitting() -> None:
    rows = [
        BenchRow(4, 16.0, 50.0, 2.3, 1150, True),
        BenchRow(8, 16.0, 50.0, 4.7, 575, True),
        BenchRow(16, 16.0, 50.0, 9.4, 288, True),
    ]
    assert recommend_group_size(rows) == 16  # largest that fits


def test_recommend_group_size_falls_back_to_smallest_when_none_fit() -> None:
    rows = [
        BenchRow(4, 1.0, 50.0, 200.0, 13, False),
        BenchRow(8, 1.0, 50.0, 400.0, 6, False),
    ]
    assert recommend_group_size(rows) == 4  # shrink as far as the smallest tested G
