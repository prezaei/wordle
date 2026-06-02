"""Performance-triggered word-set curriculum + hard-word replay queue (spec §6.5). (Plan: I)

Start on a small slice of the train answers and widen when win rate on the current tier clears a
threshold. A bounded FIFO replays recently-lost words so the policy keeps practicing hard cases.
"""

from __future__ import annotations

import logging
from collections import deque
from random import Random

from wordle_slm.config import CurriculumConfig

logger = logging.getLogger(__name__)


class Curriculum:
    """Manages the active word tier and the hard-word replay queue."""

    def __init__(self, train_words: tuple[str, ...], config: CurriculumConfig) -> None:
        if not train_words:
            raise ValueError("train_words must be non-empty")
        if not config.tiers:
            raise ValueError("curriculum.tiers must be non-empty")
        if any(t is not None and t <= 0 for t in config.tiers):
            raise ValueError(f"tier sizes must be positive or None, got {config.tiers!r}")
        self.train_words = train_words
        self.config = config
        self._tier_index = 0
        self._replay: deque[str] = deque(maxlen=config.replay_capacity)
        logger.info(
            "curriculum: %d tiers, starting tier size=%d", len(config.tiers), self._tier_size()
        )

    def _tier_size(self) -> int:
        size = self.config.tiers[self._tier_index]
        return len(self.train_words) if size is None else min(size, len(self.train_words))

    @property
    def tier_index(self) -> int:
        return self._tier_index

    def current_words(self) -> tuple[str, ...]:
        return self.train_words[: self._tier_size()]

    def sample(self, rng: Random) -> str:
        """Sample a secret: the replay queue with prob replay_prob, else the current tier."""
        if self._replay and rng.random() < self.config.replay_prob:
            return rng.choice(list(self._replay))
        return rng.choice(self.current_words())

    def record_loss(self, secret: str) -> None:
        self._replay.append(secret)

    def maybe_promote(self, win_rate: float) -> bool:
        """Advance to the next tier if win rate clears the threshold and a next tier exists."""
        if self._tier_index >= len(self.config.tiers) - 1:
            return False
        if win_rate >= self.config.promote_threshold:
            self._tier_index += 1
            logger.info(
                "curriculum promoted to tier %d (size=%d)", self._tier_index, self._tier_size()
            )
            return True
        return False
