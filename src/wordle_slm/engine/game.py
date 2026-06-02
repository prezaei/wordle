"""The Wordle game loop: tracks turns, enforces the guess cap, detects win/lose. (Plan: E)

An invalid guess (not in the valid-guess list) still consumes a turn and can never win
(spec §4.2, §6.4); its feedback is recorded as None.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from wordle_slm.data import is_valid
from wordle_slm.engine.scoring import Color, score

logger = logging.getLogger(__name__)


class Status(enum.Enum):
    ONGOING = "ongoing"
    WIN = "win"
    LOSE = "lose"


@dataclass(frozen=True)
class Turn:
    """One played guess and its scored feedback (None if the guess was not a valid word)."""

    guess: str
    feedback: tuple[Color, ...] | None
    valid: bool


class Game:
    """A single Wordle game against a fixed secret."""

    def __init__(self, secret: str, max_guesses: int = 6) -> None:
        secret = secret.lower()
        if len(secret) != 5 or not secret.isalpha():
            raise ValueError(f"secret must be 5 letters, got {secret!r}")
        self.secret = secret
        self.max_guesses = max_guesses
        self.turns: list[Turn] = []
        self._status = Status.ONGOING
        logger.info("game started (max_guesses=%d)", max_guesses)

    @property
    def status(self) -> Status:
        return self._status

    @property
    def won(self) -> bool:
        return self._status is Status.WIN

    @property
    def guesses_used(self) -> int:
        return len(self.turns)

    def guess(self, word: str) -> Turn:
        """Play one guess. Records the turn, updates status, and returns the Turn."""
        if self._status is not Status.ONGOING:
            raise RuntimeError("cannot guess: game is already over")
        word = word.lower()
        valid = is_valid(word)
        feedback = score(word, self.secret) if valid else None
        turn = Turn(guess=word, feedback=feedback, valid=valid)
        self.turns.append(turn)
        if feedback is not None and all(c is Color.GREEN for c in feedback):
            self._status = Status.WIN
        elif len(self.turns) >= self.max_guesses:
            self._status = Status.LOSE
        logger.info(
            "guess %d/%d: %r valid=%s -> %s",
            len(self.turns),
            self.max_guesses,
            word,
            valid,
            self._status.value,
        )
        return turn
