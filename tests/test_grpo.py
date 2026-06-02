"""GRPO trainer tests (Plan: Q; spec §6.3).

A random generator can't spell, so natural groups are zero-variance (all-invalid, identical reward).
We construct reward variance by monkeypatching the rollout with win/lose games to exercise the real
update path (the per-token GRPO math itself is covered in test_tracer).
"""

from __future__ import annotations

import pytest
import torch

from wordle_slm.config import CurriculumConfig, GRPOConfig, ModelConfig, RewardConfig, SFTConfig
from wordle_slm.engine import Game
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.rl import grpo as G
from wordle_slm.rl.curriculum import Curriculum
from wordle_slm.telemetry.run_log import RunLog

_CFG = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)


def _model_and_ref(seed: int = 0) -> tuple[WordleGenerator, WordleGenerator, Tokenizer]:
    torch.manual_seed(seed)
    tok = Tokenizer()
    model = WordleGenerator(_CFG, tok.vocab_size)
    ref = WordleGenerator(_CFG, tok.vocab_size)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    ref.requires_grad_(False)
    return model, ref, tok


def _win(secret: str = "crane") -> Game:
    g = Game(secret)
    g.guess("slate")
    g.guess(secret)
    return g


def _lose(secret: str = "crane") -> Game:
    g = Game(secret)
    for _ in range(6):
        g.guess("slate")
    return g


def _variance_rollouts(monkeypatch) -> None:
    games = iter([_win(), _lose(), _win(), _lose()] * 500)
    monkeypatch.setattr("wordle_slm.rl.grpo.play_game", lambda *a, **k: next(games))


def _opt(model: WordleGenerator) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


# --- grpo_update ------------------------------------------------------------------------------


def test_update_steps_moves_theta_and_reports_hard_secrets(monkeypatch) -> None:
    model, ref, tok = _model_and_ref()
    _variance_rollouts(monkeypatch)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    stats = G.grpo_update(
        model,
        ref,
        tok,
        ("crane",),
        grpo=GRPOConfig(group_size=4, inner_epochs=1),
        reward=RewardConfig(),
        optimizer=_opt(model),
        group_size=4,
    )
    assert stats.stepped and stats.kept_secrets == 1
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters())  # θ moved
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
    assert "crane" in stats.hard_secrets  # the group had a loss -> queued for replay
    for v in (stats.reward_mean, stats.kl, stats.entropy, stats.grad_norm, stats.loss):
        assert torch.isfinite(torch.tensor(v))


def test_update_runs_multiple_inner_epochs(monkeypatch) -> None:
    # K>1 exercises a real θ_old (ratio ≠ 1 after the first inner step, so the clip can bite).
    model, ref, tok = _model_and_ref()
    _variance_rollouts(monkeypatch)
    stats = G.grpo_update(
        model,
        ref,
        tok,
        ("crane",),
        grpo=GRPOConfig(group_size=4, inner_epochs=3),
        reward=RewardConfig(),
        optimizer=_opt(model),
        group_size=4,
    )
    assert stats.stepped and torch.isfinite(torch.tensor(stats.loss))


def test_update_skips_when_all_groups_zero_variance() -> None:
    # Natural random-model case: all-invalid rollouts -> identical reward -> filtered -> no step.
    model, ref, tok = _model_and_ref()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    stats = G.grpo_update(
        model,
        ref,
        tok,
        ("crane",),
        grpo=GRPOConfig(group_size=4),
        reward=RewardConfig(),
        optimizer=_opt(model),
        group_size=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert stats.stepped is False and stats.kept_secrets == 0
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())  # no update applied


def test_update_rejects_group_size_below_two() -> None:
    model, ref, tok = _model_and_ref()
    with pytest.raises(ValueError, match="group_size must be >= 2"):
        G.grpo_update(
            model,
            ref,
            tok,
            ("crane",),
            grpo=GRPOConfig(),
            reward=RewardConfig(),
            optimizer=_opt(model),
            group_size=1,
        )


# --- eval_win_rate ----------------------------------------------------------------------------


def test_eval_win_rate(monkeypatch) -> None:
    model, _, tok = _model_and_ref()
    assert G.eval_win_rate(model, tok, (), device="cpu") == 0.0
    games = iter([_win(), _lose(), _win(), _win()])  # 3/4 win
    monkeypatch.setattr("wordle_slm.rl.grpo.play_game", lambda *a, **k: next(games))
    assert G.eval_win_rate(model, tok, ("a", "b", "c", "d")) == pytest.approx(0.75)


# --- train_grpo -------------------------------------------------------------------------------


def test_train_loop_steps_records_replay_and_warms_up_lr(monkeypatch, tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    model, ref, tok = _model_and_ref()
    _variance_rollouts(monkeypatch)
    curriculum = Curriculum(("crane", "slate", "money", "light"), CurriculumConfig(tiers=(2, None)))
    with RunLog(tmp_path / "run", config={}, seed=0) as run_log:
        history = G.train_grpo(
            model,
            ref,
            tok,
            curriculum,
            grpo=GRPOConfig(group_size=2, secrets_per_update=2, inner_epochs=1, lr=1e-3),
            reward=RewardConfig(),
            n_updates=5,
            warmup_updates=3,
            run_log=run_log,
        )
    assert len(history) == 5 and all(h.stepped for h in history)
    assert len(curriculum._replay) > 0  # hard secrets were queued for replay

    acc = EventAccumulator(str(tmp_path / "run" / "tb"))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    assert {"grpo/loss", "grpo/kl", "grpo/entropy", "grpo/grad_norm", "grpo/lr"} <= tags
    lrs = [e.value for e in acc.Scalars("grpo/lr")]
    assert lrs[0] < lrs[-1] and lrs[-1] == pytest.approx(1e-3)  # linear warmup to the base lr


def test_train_loop_evaluates_and_can_promote(monkeypatch) -> None:
    model, ref, tok = _model_and_ref()
    monkeypatch.setattr("wordle_slm.rl.grpo.play_game", lambda *a, **k: _win())  # always wins
    curriculum = Curriculum(
        ("crane", "slate", "money"), CurriculumConfig(tiers=(1, None), promote_threshold=0.5)
    )
    G.train_grpo(
        model,
        ref,
        tok,
        curriculum,
        grpo=GRPOConfig(group_size=2, secrets_per_update=2, inner_epochs=1),
        reward=RewardConfig(),
        n_updates=2,
        eval_secrets=("crane", "slate"),
        eval_every=1,
    )
    assert curriculum.tier_index == 1  # 100% eval win rate promoted past the first tier


# --- X: GRPO overfit-one-secret gate (Plan: X) ------------------------------------------------


def test_grpo_overfits_one_secret_reward_rises_and_solves() -> None:
    # The decisive pre-flight gate: warm-start a model that can spell a tiny vocab (so turn-1
    # rollouts are valid + varied → real reward variance), then GRPO on ONE secret must RAISE mean
    # reward and learn to solve it greedily — the go/no-go before a real run.
    from wordle_slm.data import load_answers
    from wordle_slm.rl.rollout import play_game
    from wordle_slm.sft.pretrain import pretrain_lm

    torch.manual_seed(0)
    tok = Tokenizer()
    model = WordleGenerator(_CFG, tok.vocab_size)
    vocab = load_answers()[:10]
    secret = vocab[0]
    pretrain_lm(model, vocab, tok, SFTConfig(lr=1e-3), epochs=120, batch_size=10, seed=0)

    from wordle_slm.rl.tracer import make_reference

    history = G.overfit_one_secret(
        model,
        make_reference(model),
        tok,
        secret,
        grpo=GRPOConfig(group_size=10, inner_epochs=2, kl_beta=0.01),
        reward=RewardConfig(),
        n_updates=25,
        lr=3e-4,
        generator=torch.Generator().manual_seed(0),
    )
    rewards = [h.reward_mean for h in history]
    assert sum(rewards[-5:]) > sum(rewards[:5])  # mean group reward rises — the loop learns
    assert all(torch.isfinite(torch.tensor([h.kl, h.grad_norm, h.entropy])).all() for h in history)
    model.eval()
    assert play_game(model, tok, secret, sample=False).won  # solves the secret greedily by the end
