"""Baselines + head-start teachers: random floor, consistent yardstick, near-optimal. (Plan: J)"""

from wordle_slm.baselines.policies import (
    DEFAULT_OPENER,
    ConsistentGuesser,
    Guesser,
    InfoMaxGuesser,
    RandomGuesser,
    expected_remaining,
    play,
)

__all__ = [
    "DEFAULT_OPENER",
    "ConsistentGuesser",
    "Guesser",
    "InfoMaxGuesser",
    "RandomGuesser",
    "expected_remaining",
    "play",
]
