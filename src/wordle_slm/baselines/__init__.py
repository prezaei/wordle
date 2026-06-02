"""Baselines + head-start teachers: random floor, consistent yardstick, near-optimal (J).

The Phase-0 run + budget gate (L) lives in ``wordle_slm.baselines.phase0`` — imported by path (not
re-exported here) because it pulls in ``torch`` via the telemetry writer, while ``policies`` is
deliberately torch-free.
"""

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
