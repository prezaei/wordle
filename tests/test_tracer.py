"""GRPO tracer-bullet tests for the generation policy (Plan: V ★; spec §6).

The Layer-1 mechanics check: one end-to-end GRPO update runs without shape/NaN errors, the loss
flows ONLY to the guess-letter tokens (zero on context/feedback — the defining property), the
gradient reaches the model, KL ≥ 0, and the update moves θ off the frozen reference. It does NOT
assert the reward rises (the model is random; a random generator can't spell, so reward variance
must be constructed — see the monkeypatched group below).
"""

from __future__ import annotations

import pytest
import torch

from wordle_slm.config import GRPOConfig, ModelConfig, RewardConfig
from wordle_slm.engine import Game
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.model.serialization import encode_completed_game, guess_letter_target_positions
from wordle_slm.rl.rollout import letter_id_tensor
from wordle_slm.rl.tracer import (
    compute_group_advantages,
    grpo_tracer_step,
    make_reference,
    trajectory_terms,
)
from wordle_slm.telemetry.run_log import RunLog


def _model(seed: int = 0) -> tuple[WordleGenerator, Tokenizer]:
    torch.manual_seed(seed)
    tok = Tokenizer()
    return WordleGenerator(ModelConfig(), tok.vocab_size), tok


def _win_game(secret: str = "crane") -> Game:
    g = Game(secret)
    g.guess("slate")
    g.guess(secret)  # win in 2 (valid words, so the realized letters aren't the model's argmax)
    return g


def _lose_game(secret: str = "crane") -> Game:
    g = Game(secret)
    for _ in range(6):
        g.guess("slate")  # 6 valid-but-wrong guesses -> lose
    return g


# --- compute_group_advantages -----------------------------------------------------------------


def test_advantages_are_mean_centered_without_dividing_by_std() -> None:
    adv = compute_group_advantages(torch.tensor([1.0, 2.0, 3.0]), filter_zero_variance=True)
    assert adv is not None and torch.allclose(adv, torch.tensor([-1.0, 0.0, 1.0]))


def test_advantages_filter_drops_zero_variance() -> None:
    assert compute_group_advantages(torch.tensor([2.0, 2.0]), filter_zero_variance=True) is None


def test_advantages_filter_off_keeps_zero_variance() -> None:
    adv = compute_group_advantages(torch.tensor([2.0, 2.0]), filter_zero_variance=False)
    assert adv is not None and torch.allclose(adv, torch.zeros(2))


# --- make_reference ---------------------------------------------------------------------------


def test_reference_is_frozen_and_identical_at_creation() -> None:
    model, _ = _model()
    ref = make_reference(model)
    assert ref is not model and all(not p.requires_grad for p in ref.parameters())
    for (n, p), (rn, rp) in zip(model.named_parameters(), ref.named_parameters(), strict=True):
        assert n == rn and torch.equal(p.detach(), rp)


# --- trajectory_terms (per-token GRPO mechanics) ----------------------------------------------


def test_surrogate_equals_n_times_advantage_for_both_signs() -> None:
    model, tok = _model()
    ref = make_reference(model)
    letter_ids = letter_id_tensor(tok)
    model.eval()
    game = _win_game()
    for advantage in (2.0, -2.0):
        surrogate, _, _, n = trajectory_terms(
            model, ref, tok, game, torch.tensor(advantage), letter_ids, clip_eps=0.2, device="cpu"
        )
        # ratio == 1 at θ_old=θ, clip inactive -> Σ_t advantage = n * advantage.
        assert float(surrogate.detach()) == pytest.approx(advantage * n)


def test_kl_is_zero_at_reference_and_positive_when_policy_differs() -> None:
    model, tok = _model(seed=0)
    letter_ids = letter_id_tensor(tok)
    model.eval()
    game = _win_game()
    _, kl_same, _, _ = trajectory_terms(
        model,
        make_reference(model),
        tok,
        game,
        torch.tensor(1.0),
        letter_ids,
        clip_eps=0.2,
        device="cpu",
    )
    assert float(kl_same.detach()) == pytest.approx(0.0, abs=1e-5)  # θ == ref
    torch.manual_seed(1)
    other = WordleGenerator(ModelConfig(), tok.vocab_size)  # genuinely different params
    _, kl_diff, _, _ = trajectory_terms(
        model,
        make_reference(other),
        tok,
        game,
        torch.tensor(1.0),
        letter_ids,
        clip_eps=0.2,
        device="cpu",
    )
    assert float(kl_diff.detach()) > 0.0


def test_loss_flows_only_to_guess_letter_tokens() -> None:
    # The defining V property: gradient is non-zero ONLY on the guess-letter predict positions and
    # exactly zero on context/feedback/board tokens (the loss mask is correct).
    model, tok = _model()
    letter_ids = letter_id_tensor(tok)
    letter_lo = int(letter_ids.min())
    model.eval()
    game = _win_game()
    seq_list = encode_completed_game(game.turns, tok)
    targets = guess_letter_target_positions(seq_list, tok)
    predict = {q - 1 for q in targets}  # the logit positions the loss reads
    logits = model.forward(torch.tensor(seq_list).unsqueeze(0))
    logits.retain_grad()
    loss = torch.zeros(())
    for q in targets:
        loss = (
            loss - torch.log_softmax(logits[0, q - 1][letter_ids], dim=0)[seq_list[q] - letter_lo]
        )
    loss.backward()
    nonzero = {int(i) for i in (logits.grad[0].abs().sum(dim=1) > 0).nonzero().flatten().tolist()}
    assert nonzero, "gradient must reach the guess-letter tokens"
    assert nonzero <= predict  # never leaks to context/feedback/board positions


def test_update_ascends_the_log_prob_of_realized_guesses() -> None:
    # Sign check: a positive-advantage step must INCREASE Σ logπ(realized) — gradient ascent.
    model, tok = _model()
    ref = make_reference(model)
    letter_ids = letter_id_tensor(tok)
    game = _win_game()

    def sum_logp() -> float:
        model.eval()
        seq_list = encode_completed_game(game.turns, tok)
        targets = guess_letter_target_positions(seq_list, tok)
        lo = int(letter_ids.min())
        with torch.no_grad():
            logits = model.forward(torch.tensor(seq_list).unsqueeze(0))[0]
        return float(
            sum(torch.log_softmax(logits[q - 1][letter_ids], 0)[seq_list[q] - lo] for q in targets)
        )

    before = sum_logp()
    surrogate, _, _, _ = trajectory_terms(
        model, ref, tok, game, torch.tensor(1.0), letter_ids, clip_eps=0.2, device="cpu"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    (-surrogate).backward()  # the grpo loss for one positive-advantage trajectory (KL≈0 at ref)
    optimizer.step()
    assert sum_logp() > before


# --- grpo_tracer_step (end-to-end) ------------------------------------------------------------


def test_grpo_step_raises_when_every_group_is_zero_variance() -> None:
    # The natural random-model case: every rollout is all-invalid -> identical reward -> filtered.
    model, tok = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="no learning signal"):
        grpo_tracer_step(
            model,
            make_reference(model),
            tok,
            ("crane",),
            grpo=GRPOConfig(),
            reward=RewardConfig(),
            optimizer=optimizer,
            group_size=4,
            generator=torch.Generator().manual_seed(0),
        )


def test_grpo_step_runs_end_to_end_with_constructed_variance(monkeypatch) -> None:
    model, tok = _model()
    ref = make_reference(model)
    games = iter([_win_game(), _lose_game(), _lose_game(), _win_game()])  # a group with variance
    monkeypatch.setattr("wordle_slm.rl.tracer.play_game", lambda *a, **k: next(games))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = grpo_tracer_step(
        model,
        ref,
        tok,
        ("crane",),
        grpo=GRPOConfig(),
        reward=RewardConfig(),
        optimizer=optimizer,
        group_size=4,
    )
    assert stats.kept_groups == 1 and stats.advantage_var > 0.0
    for value in (stats.reward_mean, stats.loss, stats.kl, stats.entropy, stats.grad_norm):
        assert torch.isfinite(torch.tensor(value))
    assert stats.kl >= -1e-6
    grad_total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert grad_total > 0.0  # the gradient reached the model
    assert any(not torch.equal(p, ref.get_parameter(n)) for n, p in model.named_parameters())


def test_grpo_step_writes_scalars_to_tensorboard(monkeypatch, tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    model, tok = _model()
    games = iter([_win_game(), _lose_game(), _lose_game(), _win_game()])
    monkeypatch.setattr("wordle_slm.rl.tracer.play_game", lambda *a, **k: next(games))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with RunLog(tmp_path / "run", config={}, seed=0) as run_log:
        grpo_tracer_step(
            model,
            make_reference(model),
            tok,
            ("crane",),
            grpo=GRPOConfig(),
            reward=RewardConfig(),
            optimizer=optimizer,
            group_size=4,
            run_log=run_log,
        )
    acc = EventAccumulator(str(tmp_path / "run" / "tb"))
    acc.Reload()
    assert {"tracer/loss", "tracer/kl", "tracer/entropy", "tracer/reward_mean"} <= set(
        acc.Tags()["scalars"]
    )


def test_grpo_step_rejects_group_size_below_two() -> None:
    model, tok = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="group_size must be >= 2"):
        grpo_tracer_step(
            model,
            make_reference(model),
            tok,
            ("crane",),
            grpo=GRPOConfig(),
            reward=RewardConfig(),
            optimizer=optimizer,
            group_size=1,
        )
