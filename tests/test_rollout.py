"""Generation rollout tests (Plan: Y; spec §6.1)."""

from __future__ import annotations

import pytest
import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, load_answers
from wordle_slm.engine import Status
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.rl.rollout import play_game


def _setup(seed: int = 0) -> tuple[WordleGenerator, Tokenizer, str]:
    torch.manual_seed(seed)
    tok = Tokenizer()
    return WordleGenerator(ModelConfig(), tok.vocab_size), tok, load_answers()[0]


def test_play_game_finishes_with_five_letter_guesses() -> None:
    model, tok, secret = _setup()
    game = play_game(model, tok, secret, sample=False)
    assert game.status in (Status.WIN, Status.LOSE)
    assert all(len(turn.guess) == 5 for turn in game.turns)  # the model emits exactly 5 letters


def test_greedy_play_is_deterministic() -> None:
    model, tok, secret = _setup()
    g1 = play_game(model, tok, secret, sample=False)
    g2 = play_game(model, tok, secret, sample=False)
    assert [t.guess for t in g1.turns] == [t.guess for t in g2.turns]


def test_sampling_is_reproducible_with_a_seed() -> None:
    model, tok, secret = _setup()
    g1 = play_game(model, tok, secret, sample=True, generator=torch.Generator().manual_seed(7))
    g2 = play_game(model, tok, secret, sample=True, generator=torch.Generator().manual_seed(7))
    assert [t.guess for t in g1.turns] == [t.guess for t in g2.turns]


def test_invalid_secret_raises() -> None:
    model, tok, _ = _setup()
    assert not is_valid("zzzzz")
    with pytest.raises(ValueError, match="must be a valid word"):
        play_game(model, tok, "zzzzz")


def test_random_model_loses_within_the_guess_cap() -> None:
    # A from-scratch generator can't spell, so it should lose (all-invalid) within max_guesses.
    model, tok, secret = _setup()
    game = play_game(model, tok, secret, sample=False)
    assert game.status is Status.LOSE
    assert game.guesses_used == 6
    assert sum(t.valid for t in game.turns) == 0  # nothing valid yet (random init)


def test_max_guesses_is_respected() -> None:
    model, tok, secret = _setup()
    game = play_game(model, tok, secret, sample=False, max_guesses=2)
    assert game.guesses_used <= 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
def test_play_game_runs_on_mps() -> None:
    model, tok, secret = _setup()
    model = model.to("mps")
    game = play_game(model, tok, secret, sample=False, device="mps")
    assert game.status in (Status.WIN, Status.LOSE)
