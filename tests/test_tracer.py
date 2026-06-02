"""GRPO tracer-bullet tests (Plan: V ★).

The v3-adapted Layer-1 mechanics check: one end-to-end GRPO update runs on a random model without
shape/NaN errors, the gradient reaches the scorer, the advantage has signal, KL ≥ 0, and the
update moves the policy off the frozen reference. It does NOT assert the reward rises (random).
"""

from __future__ import annotations

import pytest
import torch

from wordle_slm.config import GRPOConfig, ModelConfig, RewardConfig
from wordle_slm.data import load_answers
from wordle_slm.model import Tokenizer
from wordle_slm.model.scorer import CandidateScorer
from wordle_slm.rl.rollout import play_game
from wordle_slm.rl.tracer import (
    compute_group_advantages,
    grpo_tracer_step,
    make_reference,
    trajectory_surrogate,
)
from wordle_slm.telemetry.run_log import RunLog


def _setup(seed: int = 0) -> tuple[CandidateScorer, Tokenizer, tuple[str, ...], tuple[str, ...]]:
    torch.manual_seed(seed)
    tok = Tokenizer()
    model = CandidateScorer(ModelConfig(), tok.vocab_size)
    pool = load_answers()[:60]  # a small pool keeps the mechanics check fast
    return model, tok, pool, pool[:3]


def _step(model, tok, pool, secrets, *, seed=0, group_size=6, run_log=None):
    ref = make_reference(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    stats = grpo_tracer_step(
        model,
        ref,
        tok,
        secrets,
        pool,
        grpo=GRPOConfig(),
        reward=RewardConfig(),
        optimizer=optimizer,
        group_size=group_size,
        device="cpu",
        generator=generator,
        run_log=run_log,
    )
    return ref, stats


# --- compute_group_advantages -----------------------------------------------------------------


def test_advantages_are_mean_centered_without_dividing_by_std() -> None:
    adv = compute_group_advantages(torch.tensor([1.0, 2.0, 3.0]), filter_zero_variance=True)
    assert adv is not None
    # Dr. GRPO: raw deviations from the mean (NOT [-1.22, 0, 1.22] that ÷std would give).
    assert torch.allclose(adv, torch.tensor([-1.0, 0.0, 1.0]))


def test_advantages_filter_drops_a_zero_variance_group() -> None:
    assert (
        compute_group_advantages(torch.tensor([2.0, 2.0, 2.0]), filter_zero_variance=True) is None
    )


def test_advantages_filter_off_keeps_a_zero_variance_group() -> None:
    adv = compute_group_advantages(torch.tensor([2.0, 2.0, 2.0]), filter_zero_variance=False)
    assert adv is not None and torch.allclose(adv, torch.zeros(3))


# --- make_reference ---------------------------------------------------------------------------


def test_reference_is_a_frozen_detached_copy() -> None:
    model, tok, _, _ = _setup()
    ref = make_reference(model)
    assert ref is not model
    assert all(not p.requires_grad for p in ref.parameters())
    for (name, p), (rname, rp) in zip(
        model.named_parameters(), ref.named_parameters(), strict=True
    ):
        assert name == rname and torch.equal(p.detach(), rp)  # identical at creation


# --- grpo_tracer_step (the end-to-end slice) --------------------------------------------------


def test_tracer_step_produces_finite_stats() -> None:
    model, tok, pool, secrets = _setup()
    _, stats = _step(model, tok, pool, secrets)
    for value in (stats.reward_mean, stats.advantage_var, stats.loss, stats.kl, stats.entropy):
        assert torch.isfinite(torch.tensor(value))
    assert stats.kept_groups >= 1


def test_tracer_gradient_reaches_the_scorer() -> None:
    # The v3 analogue of "gradient non-zero on guess-letter tokens": after one update the scorer's
    # candidate-projection (and other params) carry non-zero gradient.
    model, tok, pool, secrets = _setup()
    _step(model, tok, pool, secrets)
    total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert total > 0.0
    assert model.cand_proj.weight.grad is not None
    assert float(model.cand_proj.weight.grad.abs().sum()) > 0.0


def test_tracer_moves_the_policy_off_the_reference() -> None:
    model, tok, pool, secrets = _setup()
    ref, _ = _step(model, tok, pool, secrets)
    moved = any(not torch.equal(p, ref.get_parameter(name)) for name, p in model.named_parameters())
    assert moved  # the optimizer step changed θ; π_θ ≠ π_ref afterward


def test_tracer_advantage_has_signal() -> None:
    model, tok, pool, secrets = _setup()
    _, stats = _step(model, tok, pool, secrets)
    assert stats.advantage_var > 0.0  # varied guess counts -> non-degenerate group


def test_tracer_kl_is_nonnegative_at_the_reference() -> None:
    model, tok, pool, secrets = _setup()
    _, stats = _step(model, tok, pool, secrets)
    # k3 is non-negative by construction; at θ = π_ref it is 0 up to float rounding.
    assert stats.kl >= -1e-6


def test_tracer_is_reproducible_with_fixed_seeds() -> None:
    a_model, tok, pool, secrets = _setup(seed=0)
    _, a = _step(a_model, tok, pool, secrets, seed=0)
    b_model, _, _, _ = _setup(seed=0)
    _, b = _step(b_model, tok, pool, secrets, seed=0)
    assert a.loss == b.loss and a.reward_mean == b.reward_mean


def test_tracer_raises_when_every_group_is_filtered() -> None:
    # A singleton pool: the only candidate is the secret, so every rollout wins in one guess with
    # an identical reward -> every group is zero-variance -> no learning signal.
    model, tok, pool, _ = _setup()
    secret = pool[0]
    with pytest.raises(RuntimeError, match="no learning signal"):
        _step(model, tok, (secret,), (secret,), group_size=4)


def test_tracer_writes_scalars_to_tensorboard(tmp_path) -> None:
    model, tok, pool, secrets = _setup()
    with RunLog(tmp_path / "run", config={}, seed=0) as run_log:
        _step(model, tok, pool, secrets, run_log=run_log)
    tb_files = list((tmp_path / "run" / "tb").glob("events.out.tfevents.*"))
    assert tb_files  # the SummaryWriter recorded the tracer scalars


# --- trajectory_surrogate (KL positivity on real divergence) ----------------------------------


def test_kl_is_strictly_positive_when_policy_differs_from_reference() -> None:
    # Proves the KL term measures real divergence (not just ~0 at θ = ref).
    model, tok, pool, _ = _setup(seed=0)
    torch.manual_seed(1)
    other = CandidateScorer(ModelConfig(), tok.vocab_size)  # genuinely different params
    ref = make_reference(other)
    model.eval()
    game = play_game(model, tok, pool[0], pool, sample=False)  # multi-turn greedy game
    _, kl, _, n_steps = trajectory_surrogate(
        model, ref, tok, game, pool, torch.tensor(1.0), clip_eps=0.2, device="cpu"
    )
    assert n_steps >= 1
    kl = kl.detach()
    assert float(kl) > 0.0 and torch.isfinite(kl)
