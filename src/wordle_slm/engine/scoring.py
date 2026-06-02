"""Wordle color scoring with correct duplicate-letter handling.

The only subtle part of Wordle scoring is repeated letters, so this is a two-pass
algorithm (the standard rule):

1. Mark GREEN where guess[i] == answer[i], and build a pool (multiset) of the
   answer's *remaining* (non-green) letters.
2. For each non-green position, assign YELLOW only while that letter still has a
   remaining count in the pool (decrementing on use), otherwise GRAY.
"""

from __future__ import annotations

import enum
from collections import Counter


class Color(enum.Enum):
    """Per-position feedback for a guess."""

    GREEN = "G"  # right letter, right position
    YELLOW = "Y"  # right letter, wrong position
    GRAY = "X"  # letter not in the answer (or no remaining count)


def score(guess: str, answer: str) -> tuple[Color, ...]:
    """Score ``guess`` against ``answer``; returns one Color per position.

    Case-insensitive. Raises ValueError on a length mismatch.
    """
    if len(guess) != len(answer):
        raise ValueError(f"guess/answer length mismatch: {guess!r} vs {answer!r}")

    guess = guess.lower()
    answer = answer.lower()
    result: list[Color] = [Color.GRAY] * len(guess)

    # Pass 1: greens + pool of remaining (non-green) answer letters.
    pool: Counter[str] = Counter()
    for i, (g, a) in enumerate(zip(guess, answer, strict=True)):
        if g == a:
            result[i] = Color.GREEN
        else:
            pool[a] += 1

    # Pass 2: yellows from the remaining pool, else gray.
    for i, g in enumerate(guess):
        if result[i] is Color.GREEN:
            continue
        if pool[g] > 0:
            result[i] = Color.YELLOW
            pool[g] -= 1

    return tuple(result)
