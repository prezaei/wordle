"""Engine color-scoring tests (Plan: B).

Per AGENTS.md: exact expected tuples, with the tricky duplicate-letter cases, and
every Color variant exercised through the public entry point (`score`).
"""

from __future__ import annotations

import pytest

from wordle_slm.engine import Color, score

_CHAR_TO_COLOR = {"G": Color.GREEN, "Y": Color.YELLOW, "X": Color.GRAY}


def colors(s: str) -> tuple[Color, ...]:
    """Build an expected tuple from a compact string, e.g. "YXYXG"."""
    return tuple(_CHAR_TO_COLOR[c] for c in s)


@pytest.mark.parametrize(
    ("guess", "answer", "expected"),
    [
        ("CRANE", "CRANE", "GGGGG"),  # exact match -> all green
        ("ABODE", "FILTH", "XXXXX"),  # no shared letters -> all gray
        ("ALLEY", "EARLY", "YYXYG"),  # double L: one yellow, the second gray
        ("SPEED", "ERASE", "YXYYX"),  # double E across positions
        ("LLAMA", "BALSA", "YXYXG"),  # repeated letters, mixed greens/yellows
    ],
)
def test_score_exact(guess: str, answer: str, expected: str) -> None:
    assert score(guess, answer) == colors(expected)


def test_every_color_variant_is_produced() -> None:
    seen: set[Color] = set()
    for g, a in [("ALLEY", "EARLY"), ("LLAMA", "BALSA")]:
        seen.update(score(g, a))
    assert seen == {Color.GREEN, Color.YELLOW, Color.GRAY}


def test_case_insensitive() -> None:
    assert score("crane", "CRANE") == colors("GGGGG")


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        score("ABCD", "ABCDE")
