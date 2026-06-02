"""Wordle game engine: color scoring, the Game loop, and consistent-candidate filtering.

Plan: B (scoring), E (Game), candidate filter (v3 action space).
"""

from wordle_slm.engine.constraints import (
    consistent_candidates,
    filter_consistent,
    is_consistent,
    secret_in_pool,
)
from wordle_slm.engine.game import Game, Status, Turn
from wordle_slm.engine.scoring import Color, score

__all__ = [
    "Color",
    "Game",
    "Status",
    "Turn",
    "consistent_candidates",
    "filter_consistent",
    "is_consistent",
    "score",
    "secret_in_pool",
]
