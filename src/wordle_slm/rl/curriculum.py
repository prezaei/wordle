"""Difficulty-ordered, diversity-first curriculum + hard-word replay (redesigned; §6.5; Plan I).

Redesign rationale (from the measured generalization wall):
- The secret pool is the FULL valid list, not just the 2,315 answers — 8x the diversity, the lever
  that actually reduces the train/held-out gap. ``build_curriculum_pool`` orders it easy->hard:
  common answers first (the eval distribution), then the rarer non-answer valid words.
- Tiers WIDEN over that ordered pool; ``maybe_promote`` advances on the win-rate gate OR after
  ``promote_patience`` eval points (the old fixed gate never fired at our win rates, leaving the
  policy stuck on a tiny slice — a curriculum that never progresses is worse than none).
- A bounded FIFO replays recently-lost words so the policy keeps practicing hard cases.
"""

from __future__ import annotations

import logging
from collections import deque
from random import Random

from wordle_slm.config import CurriculumConfig
from wordle_slm.data import load_answers, load_valid_guesses, split

logger = logging.getLogger(__name__)


def difficulty(word: str) -> tuple[int, str]:
    """Curriculum difficulty key (lower = easier). Repeated letters are the model's documented weak
    spot, so all-distinct words rank easiest; alphabetical tie-break keeps the order stable."""
    return (len(word) - len(set(word)), word)


def build_curriculum_pool(seed: int = 0, train_frac: float = 0.80) -> tuple[str, ...]:
    """Difficulty-ordered, diversity-first secret pool: common answers first, then rarer words.

    Held-out answers are excluded (they must never be trained on). Returns ~8x the answer-only pool
    so the policy can't memorize a tiny secret set — the redesign's central change.
    """
    train_answers, held_out = split(seed=seed, train_frac=train_frac)
    answer_set = set(load_answers())
    held_set = set(held_out)
    easy = sorted(train_answers, key=difficulty)  # common answers (the eval distribution)
    hard = sorted(
        (w for w in load_valid_guesses() if w not in answer_set and w not in held_set),
        key=difficulty,
    )  # rarer non-answer valid words (diversity / generalization)
    pool = tuple(easy + hard)
    logger.info(
        "curriculum pool: %d secrets (%d train answers + %d rarer valid), held-out excluded",
        len(pool),
        len(easy),
        len(hard),
    )
    return pool


class Curriculum:
    """Manages the active difficulty tier and the hard-word replay queue."""

    def __init__(self, train_words: tuple[str, ...], config: CurriculumConfig) -> None:
        if not train_words:
            raise ValueError("train_words must be non-empty")
        if not config.tiers:
            raise ValueError("curriculum.tiers must be non-empty")
        if any(t is not None and t <= 0 for t in config.tiers):
            raise ValueError(f"tier sizes must be positive or None, got {config.tiers!r}")
        resolved = [float("inf") if t is None else t for t in config.tiers]
        if any(b <= a for a, b in zip(resolved, resolved[1:], strict=False)):
            raise ValueError(f"tiers must strictly increase (widen), got {config.tiers!r}")
        self.train_words = train_words
        self.config = config
        self._tier_index = 0
        self._evals_on_tier = 0  # eval points since the last promotion (drives patience widening)
        self._replay: deque[str] = deque(maxlen=config.replay_capacity)
        logger.info(
            "curriculum: %d tiers over %d secrets, starting tier size=%d",
            len(config.tiers),
            len(train_words),
            self._tier_size(),
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
        """Widen to the next tier on the win-rate gate OR after ``promote_patience`` evals (robust).

        The patience fallback guarantees progress even when the win-rate gate is never cleared (the
        old failure mode): a curriculum stuck on its first tier is strictly worse than none.
        """
        if self._tier_index >= len(self.config.tiers) - 1:
            return False
        self._evals_on_tier += 1
        forced = self._evals_on_tier >= self.config.promote_patience
        if win_rate >= self.config.promote_threshold or forced:
            self._tier_index += 1
            self._evals_on_tier = 0
            logger.info(
                "curriculum promoted to tier %d (size=%d)%s",
                self._tier_index,
                self._tier_size(),
                " [patience]" if forced and win_rate < self.config.promote_threshold else "",
            )
            return True
        return False
