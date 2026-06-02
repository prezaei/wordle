"""Game-loop tests (Plan: E)."""

from __future__ import annotations

import pytest

from wordle_slm.data import load_valid_guesses
from wordle_slm.engine import Color, Game, Status


def _invalid_word() -> str:
    valid = set(load_valid_guesses())
    cand = "aaaaa"
    while cand in valid:
        cand = cand[:-1] + chr((ord(cand[-1]) - ord("a") + 1) % 26 + ord("a"))
    return cand


def test_win_when_guess_matches_secret() -> None:
    g = Game("crane")
    turn = g.guess("crane")
    assert g.status is Status.WIN and g.won
    assert turn.valid and turn.feedback == (Color.GREEN,) * 5
    assert g.guesses_used == 1


def test_ongoing_then_lose_after_max_guesses() -> None:
    g = Game("crane", max_guesses=3)
    g.guess("slate")
    assert g.status is Status.ONGOING
    g.guess("table")
    assert g.status is Status.ONGOING
    g.guess("plumb")
    assert g.status is Status.LOSE
    assert g.guesses_used == 3


def test_invalid_guess_consumes_a_turn_and_cannot_win() -> None:
    g = Game("crane")
    turn = g.guess(_invalid_word())
    assert turn.valid is False
    assert turn.feedback is None
    assert g.guesses_used == 1
    assert g.status is Status.ONGOING


def test_invalid_guess_on_last_turn_loses() -> None:
    g = Game("crane", max_guesses=1)
    g.guess(_invalid_word())
    assert g.status is Status.LOSE


def test_guess_after_game_over_raises() -> None:
    g = Game("crane")
    g.guess("crane")
    with pytest.raises(RuntimeError):
        g.guess("slate")


def test_secret_must_be_five_letters() -> None:
    for bad in ("abcd", "abcdef", "abcd1", ""):
        with pytest.raises(ValueError):
            Game(bad)


def test_secret_and_guess_are_case_insensitive() -> None:
    g = Game("CRANE")
    g.guess("CrAnE")
    assert g.won
