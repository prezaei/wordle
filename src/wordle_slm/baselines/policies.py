"""Phase-0 baselines and head-start teachers (spec §4.3, §5.4). (Plan: J)

Three guessers, weakest to strongest, all driven by the shared `play()` loop:

- `RandomGuesser` — the *floor*: a uniformly random word from a fixed draw-pool, ignoring every
  clue. Instantiate over the answer pool (headline ~0.26% win) or the full valid list (~0.05%).
- `ConsistentGuesser` — the *honest yardstick*: open with a fixed strong word, then a random word
  still consistent with all clues, drawn from the valid list. Expected ~96–99% (spec §4.3).
- `InfoMaxGuesser` — the *near-optimal teacher*: open strong, then the remaining answer that
  minimizes the expected number of still-consistent answers (greedy one-ply lookahead; spec §5.4).

Each guesser owns its `default_pool` — the base set consistency is tracked over (valid list for the
yardstick, answer list for the teacher) — so `play()` is never handed the wrong pool by accident.
`ConsistentGuesser` and `InfoMaxGuesser` reuse the engine's incremental `filter_consistent` via the
driver; the floor ignores consistency entirely (`needs_consistent = False`), so the driver skips the
filtering work for it. This is the heuristic counterpart of the model rollout `rl.rollout.play_game`
(which scores candidates with the network); the two share the loop *shape* but not the policy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from random import Random
from typing import Protocol

from wordle_slm.data import is_valid, load_answers, load_valid_guesses
from wordle_slm.engine import (
    Color,
    Game,
    Status,
    Turn,
    filter_consistent,
    score,
    secret_in_pool,
)

logger = logging.getLogger(__name__)

DEFAULT_OPENER = "slate"  # a strong, fixed starter (a valid guess; spec §4.3 / §5.4)


class Guesser(Protocol):
    """A Wordle policy: choose the next guess from the history and the still-consistent set."""

    needs_consistent: bool

    @property
    def default_pool(self) -> tuple[str, ...]:
        """The base set `play()` tracks consistency over (unused when not `needs_consistent`)."""
        ...

    def choose(self, turns: Sequence[Turn], consistent: tuple[str, ...]) -> str: ...


def _validate_opener(opener: str) -> str:
    opener = opener.lower()
    if not is_valid(opener):
        raise ValueError(f"opener {opener!r} is not a valid guess")
    return opener


def _require_candidates(consistent: tuple[str, ...]) -> None:
    if not consistent:
        raise RuntimeError("no consistent candidates to choose from (need at least one)")


class RandomGuesser:
    """The floor: uniformly random words from `draw_pool`, ignoring all feedback."""

    needs_consistent = False

    def __init__(self, draw_pool: tuple[str, ...], seed: int = 0) -> None:
        if not draw_pool:
            raise ValueError("draw_pool must be non-empty")
        self._draw_pool = draw_pool
        self._rng = Random(seed)
        logger.info("RandomGuesser: draw_pool=%d words (seed=%d)", len(draw_pool), seed)

    @property
    def default_pool(self) -> tuple[str, ...]:
        return self._draw_pool  # the floor never consults it (needs_consistent is False)

    def choose(self, turns: Sequence[Turn], consistent: tuple[str, ...]) -> str:
        return self._rng.choice(self._draw_pool)  # feedback ignored — this is the floor


class ConsistentGuesser:
    """The yardstick: a fixed opener, then a uniformly random still-consistent word (valid list)."""

    needs_consistent = True

    def __init__(
        self, opener: str = DEFAULT_OPENER, seed: int = 0, pool: tuple[str, ...] | None = None
    ) -> None:
        self._opener = _validate_opener(opener)
        self._rng = Random(seed)
        self._pool = pool if pool is not None else load_valid_guesses()
        logger.info(
            "ConsistentGuesser: opener=%r (seed=%d, pool=%d)", self._opener, seed, len(self._pool)
        )

    @property
    def default_pool(self) -> tuple[str, ...]:
        return self._pool

    def choose(self, turns: Sequence[Turn], consistent: tuple[str, ...]) -> str:
        if not turns:
            return self._opener
        _require_candidates(consistent)
        return self._rng.choice(consistent)


def expected_remaining(guess: str, answers: tuple[str, ...]) -> float:
    """Expected # of still-consistent answers after `guess`, over a uniform secret in `answers`.

    Partition `answers` by feedback pattern; E[|remaining|] = Σ_pattern (|bucket|/N) · |bucket|.
    Lower is a more informative guess. `answers` must be non-empty.
    """
    if not answers:
        raise ValueError("answers must be non-empty")
    buckets: dict[tuple[Color, ...], int] = {}
    for answer in answers:
        pattern = score(guess, answer)
        buckets[pattern] = buckets.get(pattern, 0) + 1
    n = len(answers)
    return sum(size * size for size in buckets.values()) / n


class InfoMaxGuesser:
    """The near-optimal teacher: a fixed opener, then the answer minimizing `expected_remaining`."""

    needs_consistent = True

    def __init__(self, opener: str = DEFAULT_OPENER, pool: tuple[str, ...] | None = None) -> None:
        self._opener = _validate_opener(opener)
        self._pool = pool if pool is not None else load_answers()
        logger.info("InfoMaxGuesser: opener=%r (pool=%d answers)", self._opener, len(self._pool))

    @property
    def default_pool(self) -> tuple[str, ...]:
        return self._pool  # the answer list: expected_remaining is meaningful only over answers

    def choose(self, turns: Sequence[Turn], consistent: tuple[str, ...]) -> str:
        if not turns:
            return self._opener
        _require_candidates(consistent)
        if len(consistent) == 1:
            return consistent[0]
        # Greedy one-ply: the candidate leaving the fewest expected still-consistent answers.
        # `min` is deterministic — `consistent` keeps the sorted-pool order, so ties break first.
        return min(consistent, key=lambda g: expected_remaining(g, consistent))


def play(
    guesser: Guesser,
    secret: str,
    *,
    pool: tuple[str, ...] | None = None,
    max_guesses: int = 6,
) -> Game:
    """Drive one full game with `guesser`.

    The base set for consistency tracking defaults to `guesser.default_pool` (the right pool for
    each guesser — valid list for the yardstick, answers for the teacher); pass `pool` only to
    override it (e.g. a small pool in tests). For `needs_consistent` guessers it is narrowed
    incrementally (one clue per turn) and passed to `choose`; for the floor it is unused.
    """
    base_pool = guesser.default_pool if pool is None else pool
    if guesser.needs_consistent and not secret_in_pool(secret, base_pool):
        # Without this the consistent set can empty out mid-game (same guard as play_game / reward).
        raise ValueError(f"secret {secret!r} must be in pool for a consistency-tracking guesser")
    game = Game(secret, max_guesses=max_guesses)
    consistent = base_pool
    while game.status is Status.ONGOING:
        word = guesser.choose(game.turns, consistent)
        turn = game.guess(word)
        if guesser.needs_consistent:
            consistent = filter_consistent(consistent, turn)
    logger.info(
        "baseline %s: secret=%r -> %s in %d",
        type(guesser).__name__,
        secret,
        game.status.value,
        game.guesses_used,
    )
    return game
