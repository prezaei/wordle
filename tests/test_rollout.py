"""Rollout tests (Plan: Y)."""

from __future__ import annotations

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import is_valid, load_answers
from wordle_slm.engine import Status
from wordle_slm.model import Tokenizer
from wordle_slm.model.scorer import CandidateScorer
from wordle_slm.rl.rollout import play_game


def _setup() -> tuple[CandidateScorer, Tokenizer, tuple[str, ...]]:
    torch.manual_seed(0)
    tok = Tokenizer()
    model = CandidateScorer(ModelConfig(), tok.vocab_size)
    return model, tok, load_answers()


def test_play_game_finishes_with_valid_consistent_guesses() -> None:
    model, tok, pool = _setup()
    game = play_game(model, tok, pool[0], pool, sample=False)
    assert game.status in (Status.WIN, Status.LOSE)
    for turn in game.turns:
        assert turn.valid and is_valid(turn.guess)  # only ever guesses real words


def test_greedy_play_is_deterministic() -> None:
    model, tok, pool = _setup()
    g1 = play_game(model, tok, pool[0], pool, sample=False)
    g2 = play_game(model, tok, pool[0], pool, sample=False)
    assert [t.guess for t in g1.turns] == [t.guess for t in g2.turns]


def test_restricted_action_space_wins_most_games_even_untrained() -> None:
    # With the action restricted to still-consistent words, winning is near-automatic even with a
    # random (untrained) scorer — this is the whole point of the v3 architecture.
    model, tok, pool = _setup()
    secrets = pool[:10]
    wins = sum(play_game(model, tok, s, pool, sample=False).won for s in secrets)
    assert wins / len(secrets) >= 0.6


def test_sampling_path_runs() -> None:
    model, tok, pool = _setup()
    gen = torch.Generator().manual_seed(0)
    game = play_game(model, tok, pool[0], pool, sample=True, generator=gen)
    assert game.status in (Status.WIN, Status.LOSE)


def test_sampling_is_reproducible_with_a_seed() -> None:
    model, tok, pool = _setup()
    g1 = play_game(
        model, tok, pool[0], pool, sample=True, generator=torch.Generator().manual_seed(7)
    )
    g2 = play_game(
        model, tok, pool[0], pool, sample=True, generator=torch.Generator().manual_seed(7)
    )
    assert [t.guess for t in g1.turns] == [t.guess for t in g2.turns]


def test_lose_path_when_first_guess_misses_at_cap() -> None:
    model, tok, pool = _setup()
    first = play_game(model, tok, pool[0], pool, sample=False, max_guesses=1).turns[0].guess
    secret = next(w for w in pool if w != first)  # guarantee the single guess misses
    game = play_game(model, tok, secret, pool, sample=False, max_guesses=1)
    assert game.status is Status.LOSE
    assert game.guesses_used == 1
