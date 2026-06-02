"""Generalization-gap + best-checkpoint-selection tests (Plan: S; spec §6.7)."""

from __future__ import annotations

import torch

from wordle_slm.config import CurriculumConfig, GRPOConfig, ModelConfig, RewardConfig
from wordle_slm.engine import Game
from wordle_slm.eval.selection import GapReport, generalization_gap
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.rl import grpo as G
from wordle_slm.rl.curriculum import Curriculum
from wordle_slm.sft import load_checkpoint

_CFG = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)


def _model_ref_tok():
    torch.manual_seed(0)
    tok = Tokenizer()
    model = WordleGenerator(_CFG, tok.vocab_size)
    ref = WordleGenerator(_CFG, tok.vocab_size)
    ref.load_state_dict(model.state_dict())
    ref.requires_grad_(False)
    return model, ref, tok


def _win(secret="crane"):
    g = Game(secret)
    g.guess("slate")
    g.guess(secret)
    return g


def _lose(secret="crane"):
    g = Game(secret)
    for _ in range(6):
        g.guess("slate")
    return g


# --- generalization gap -----------------------------------------------------------------------


def test_generalization_gap_is_signed_probe_minus_heldout(monkeypatch) -> None:
    model, _, tok = _model_ref_tok()
    # probe: 2 secrets both win (1.0); held-out: 2 secrets, 1 win (0.5) -> gap = +0.5 (memorizing).
    games = iter([_win(), _win(), _win(), _lose()])
    monkeypatch.setattr("wordle_slm.rl.grpo.play_game", lambda *a, **k: next(games))
    report = generalization_gap(
        model, tok, probe_secrets=("a", "b"), heldout_secrets=("c", "d"), device="cpu"
    )
    assert report == GapReport(1.0, 0.5, 0.5)
    assert report.memorizing is True


def test_generalization_gap_negative_is_not_memorizing(monkeypatch) -> None:
    model, _, tok = _model_ref_tok()
    games = iter([_lose(), _win(), _win()])  # probe 0/1, held-out 2/2 -> gap = -1.0
    monkeypatch.setattr("wordle_slm.rl.grpo.play_game", lambda *a, **k: next(games))
    report = generalization_gap(
        model, tok, probe_secrets=("a",), heldout_secrets=("b", "c"), device="cpu"
    )
    assert report.gap == -1.0 and report.memorizing is False


# --- best-checkpoint-by-held-out --------------------------------------------------------------


def test_train_saves_the_best_held_out_checkpoint_not_the_last(monkeypatch, tmp_path) -> None:
    model, ref, tok = _model_ref_tok()
    # rollouts always have variance so every update steps (θ keeps changing).
    rollouts = iter([_win(), _lose()] * 10_000)
    monkeypatch.setattr("wordle_slm.rl.grpo.play_game", lambda *a, **k: next(rollouts))
    # held-out win rate: peaks at update 2 (1.0), then collapses at update 3 — best must be kept.
    eval_rates = iter([0.4, 1.0, 0.2])
    monkeypatch.setattr("wordle_slm.rl.grpo.eval_win_rate", lambda *a, **k: next(eval_rates))
    best_path = tmp_path / "best.pt"
    G.train_grpo(
        model,
        ref,
        tok,
        Curriculum(("crane", "slate"), CurriculumConfig(tiers=(2, None))),
        grpo=GRPOConfig(group_size=2, secrets_per_update=1, inner_epochs=1),
        reward=RewardConfig(),
        n_updates=3,
        eval_secrets=("crane",),
        eval_every=1,
        best_checkpoint=best_path,
    )
    assert best_path.exists()
    # the saved checkpoint is the peak (update 2, 0-indexed step 1), not the final collapsed one.
    fresh = WordleGenerator(_CFG, tok.vocab_size)
    ckpt = load_checkpoint(best_path, fresh)
    assert ckpt["step"] == 1  # update index 1 (the 1.0 peak), not 2 (the 0.2 collapse)


def test_train_logs_the_generalization_gap(monkeypatch, tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from wordle_slm.telemetry.run_log import RunLog

    model, ref, tok = _model_ref_tok()
    rollouts = iter([_win(), _lose()] * 10_000)
    monkeypatch.setattr("wordle_slm.rl.grpo.play_game", lambda *a, **k: next(rollouts))
    monkeypatch.setattr("wordle_slm.rl.grpo.eval_win_rate", lambda *a, **k: 0.5)
    with RunLog(tmp_path / "run", config={}, seed=0) as run_log:
        G.train_grpo(
            model,
            ref,
            tok,
            Curriculum(("crane", "slate"), CurriculumConfig(tiers=(2, None))),
            grpo=GRPOConfig(group_size=2, secrets_per_update=1, inner_epochs=1),
            reward=RewardConfig(),
            n_updates=2,
            eval_secrets=("crane",),
            probe_secrets=("slate",),
            eval_every=1,
            run_log=run_log,
        )
    acc = EventAccumulator(str(tmp_path / "run" / "tb"))
    acc.Reload()
    assert {"grpo/gen_gap", "grpo/probe_win_rate"} <= set(acc.Tags()["scalars"])
