"""Consistent-candidate filter tests (Plan: v3 action space)."""

from __future__ import annotations

from wordle_slm.data import load_answers
from wordle_slm.engine import Turn, consistent_candidates, is_consistent, score


def _turn(guess: str, secret: str) -> Turn:
    return Turn(guess=guess, feedback=score(guess, secret), valid=True)


def test_secret_is_always_consistent_with_its_own_feedback() -> None:
    history = [_turn("slate", "crane"), _turn("brick", "crane")]
    assert is_consistent("crane", history)


def test_word_reproducing_the_feedback_is_consistent() -> None:
    history = [_turn("slate", "crane")]
    assert score("slate", "grace") == score("slate", "crane")  # same clue
    assert is_consistent("grace", history)


def test_word_contradicting_a_clue_is_inconsistent() -> None:
    # 'slate' marks 's' absent against 'crane'; 'scare' contains 's'.
    history = [_turn("slate", "crane")]
    assert is_consistent("scare", history) is False


def test_consistent_candidates_filters_pool() -> None:
    history = [_turn("slate", "crane")]
    pool = ("crane", "grace", "scare", "slate")
    cands = consistent_candidates(history, pool)
    assert "crane" in cands
    assert "grace" in cands
    assert "scare" not in cands  # absent 's'
    assert "slate" not in cands  # its own all-green != observed feedback


def test_invalid_turns_carry_no_constraint() -> None:
    bad = Turn(guess="zzzzz", feedback=None, valid=False)
    assert is_consistent("crane", [bad])


def test_empty_history_keeps_whole_pool() -> None:
    pool = ("crane", "slate", "grace")
    assert consistent_candidates([], pool) == pool


def test_multi_turn_history_narrows_progressively() -> None:
    pool = load_answers()
    secret = pool[0]
    one = consistent_candidates([_turn("slate", secret)], pool)
    two = consistent_candidates([_turn("slate", secret), _turn("crane", secret)], pool)
    assert secret in one and secret in two
    assert len(two) <= len(one) < len(pool)  # each clue can only shrink the set


def test_mixed_valid_and_invalid_history_uses_only_valid_clues() -> None:
    invalid = Turn(guess="zzzzz", feedback=None, valid=False)
    valid = _turn("slate", "crane")  # 's' absent
    assert is_consistent("grace", [invalid, valid]) is True  # consistent via the valid clue
    assert is_consistent("scare", [invalid, valid]) is False  # 's' present -> contradicts
