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
from wordle_slm.engine import filter_consistent
from wordle_slm.model import Tokenizer
from wordle_slm.model.scorer import CandidateScorer
from wordle_slm.model.serialization import encode_board, encode_word
from wordle_slm.rl.reward import compute_reward
from wordle_slm.rl.rollout import play_game
from wordle_slm.rl.tracer import (
    compute_group_advantages,
    grpo_tracer_step,
    make_reference,
    trajectory_surrogate,
)
from wordle_slm.telemetry.run_log import RunLog


def _sum_chosen_logp(model: CandidateScorer, tok: Tokenizer, game, pool: tuple[str, ...]) -> float:
    """Σ_t logπ(chosen guess | board_t), replayed independently of trajectory_surrogate, no grad."""
    pad_id = tok.pad_id
    candidates: tuple[str, ...] = tuple(pool)
    total = 0.0
    history: list = []
    with torch.no_grad():
        for turn in game.turns:
            board = torch.tensor(encode_board(history, tok)).unsqueeze(0)
            cand_ids = torch.tensor([encode_word(w, tok) for w in candidates])
            logprobs = torch.log_softmax(model.score(board, cand_ids, pad_id), dim=0)
            total += float(logprobs[candidates.index(turn.guess)])
            history.append(turn)
            candidates = filter_consistent(candidates, turn)
    return total


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


def test_tracer_update_ascends_the_advantage_weighted_objective() -> None:
    # The Layer-1 SIGN check: one update must move θ UPHILL on Σ_traj adv·Σ_t logπ(chosen) — i.e.
    # gradient ASCENT. "Gradient is non-zero" alone passes even with a flipped (descent) loss sign;
    # this fails on that bug. At θ=ref the KL gradient is 0, so the first step is pure surrogate.
    model, tok, pool, _ = _setup()
    secret = pool[0]
    ref = make_reference(model)

    # Reproduce the exact group grpo_tracer_step samples (same seed, pre-update model).
    model.eval()
    gen = torch.Generator().manual_seed(0)
    games = [play_game(model, tok, secret, pool, sample=True, generator=gen) for _ in range(6)]
    rewards = torch.tensor([compute_reward(g, RewardConfig(), pool).total for g in games])
    advantages = rewards - rewards.mean()
    assert not bool((advantages.abs() < 1e-9).all())  # the group must carry signal

    def objective() -> float:
        model.eval()
        return sum(
            float(a) * _sum_chosen_logp(model, tok, g, pool)
            for g, a in zip(games, advantages, strict=True)
        )

    before = objective()
    grpo_tracer_step(
        model,
        ref,
        tok,
        (secret,),
        pool,
        grpo=GRPOConfig(),
        reward=RewardConfig(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
        group_size=6,
        device="cpu",
        generator=torch.Generator().manual_seed(0),
    )
    assert objective() > before  # ascent on the same rollouts; a flipped loss sign would descend


def test_reference_receives_no_gradient_after_a_step() -> None:
    # The v3 analogue of the old "gradient is zero on context/feedback tokens": no gradient must
    # leak into the frozen π_ref (it is consulted only under no_grad for the KL penalty).
    model, tok, pool, secrets = _setup()
    ref, _ = _step(model, tok, pool, secrets)
    assert all(p.grad is None for p in ref.parameters())


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


def test_tracer_writes_the_expected_scalars_to_tensorboard(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    model, tok, pool, secrets = _setup()
    with RunLog(tmp_path / "run", config={}, seed=0) as run_log:
        _step(model, tok, pool, secrets, run_log=run_log)
    acc = EventAccumulator(str(tmp_path / "run" / "tb"))
    acc.Reload()
    # Assert the specific tags were written (an empty RunLog also creates the event file) — the
    # V DoD requires reward + L_clip(loss) + KL + entropy in TensorBoard.
    expected = {
        "tracer/reward_mean",
        "tracer/advantage_var",
        "tracer/loss",
        "tracer/kl",
        "tracer/entropy",
        "tracer/grad_norm",
        "tracer/kept_groups",
    }
    assert expected <= set(acc.Tags()["scalars"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
def test_tracer_step_runs_on_mps() -> None:
    # V DoD: one update runs on the project's target backend without shape/NaN errors.
    model, tok, pool, secrets = _setup()
    model = model.to("mps")
    ref = make_reference(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)  # CPU generator: multinomial samples on CPU
    stats = grpo_tracer_step(
        model,
        ref,
        tok,
        secrets,
        pool,
        grpo=GRPOConfig(),
        reward=RewardConfig(),
        optimizer=optimizer,
        group_size=4,
        device="mps",
        generator=generator,
    )
    assert torch.isfinite(torch.tensor(stats.loss))
    assert stats.kl >= -1e-6 and stats.kept_groups >= 1


def test_tracer_restores_caller_training_mode() -> None:
    model, tok, pool, secrets = _setup()
    model.train()  # caller expects train mode to survive the call
    _step(model, tok, pool, secrets)
    assert model.training is True


# --- trajectory_surrogate (surrogate value + KL positivity on real divergence) ----------------


def test_trajectory_surrogate_scales_with_advantage() -> None:
    # At θ_old = θ the per-step ratio is exactly 1 (clip inactive), so the trajectory surrogate is
    # Σ_t advantage = n_steps · advantage — for BOTH advantage signs.
    model, tok, pool, _ = _setup()
    ref = make_reference(model)
    model.eval()
    game = play_game(model, tok, pool[0], pool, sample=False)
    for advantage in (2.0, -2.0):
        surrogate, _, _, n_steps = trajectory_surrogate(
            model, ref, tok, game, pool, torch.tensor(advantage), clip_eps=0.2, device="cpu"
        )
        assert float(surrogate.detach()) == pytest.approx(advantage * n_steps)


def test_kl_is_strictly_positive_when_policy_differs_from_reference() -> None:
    # Proves the KL term measures real divergence (not just ~0 at θ = ref). The two seeds MUST
    # differ — identical seeds give identical nets and KL == 0 exactly.
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
