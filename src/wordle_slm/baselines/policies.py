"""Phase-0 baselines and head-start teachers (spec §4.3, §5.4). (Plan: J)

Three guessers, weakest to strongest, all driven by the shared `play()` loop:

- `RandomGuesser` — the *floor*: a uniformly random word from a fixed draw-pool, ignoring every
  clue. Instantiate over the answer pool (headline ~0.26% win) or the full valid list (~0.05%).
- `ConsistentGuesser` — the *honest yardstick*: open with a fixed strong word, then a random word
  still consistent with all clues. Expected ~96–99% (spec §4.3).
- `InfoMaxGuesser` — the *near-optimal teacher*: open strong, then the remaining answer that
  minimizes the expected number of still-consistent answers (greedy one-ply lookahead; spec §5.4).

`ConsistentGuesser` and `InfoMaxGuesser` reuse the engine's incremental `filter_consistent` via the
driver; the floor ignores consistency entirely (`needs_consistent = False`), so the driver skips the
filtering work for it. This is the heuristic counterpart of the model rollout `rl.rollout.play_game`
(which scores candidates with the network); the two share the loop *shape* but not the policy.
"""

from __future__ import annotations

import logging
from random import Random
from typing import Protocol

from wordle_slm.data import is_valid
from wordle_slm.engine import Color, Game, Status, Turn, filter_consistent, score

logger = logging.getLogger(__name__)

DEFAULT_OPENER = "slate"  # a strong, fixed starter (a valid guess; spec §4.3 / §5.4)


class Guesser(Protocol):
    """A Wordle policy: choose the next guess from the history and the still-consistent set."""

    needs_consistent: bool

    def choose(self, turns: tuple[Turn, ...], consistent: tuple[str, ...]) -> str: ...


def _validate_opener(opener: str) -> str:
    opener = opener.lower()
    if not is_valid(opener):
        raise ValueError(f"opener {opener!r} is not a valid guess")
    return opener


def _require_candidates(consistent: tuple[str, ...]) -> None:
    if not consistent:
        raise RuntimeError("no consistent candidates: the secret should keep the set non-empty")


class RandomGuesser:
    """The floor: uniformly random words from `draw_pool`, ignoring all feedback."""

    needs_consistent = False

    def __init__(self, draw_pool: tuple[str, ...], seed: int = 0) -> None:
        if not draw_pool:
            raise ValueError("draw_pool must be non-empty")
        self._draw_pool = draw_pool
        self._rng = Random(seed)
        logger.info("RandomGuesser: draw_pool=%d words (seed=%d)", len(draw_pool), seed)

    def choose(self, turns: tuple[Turn, ...], consistent: tuple[str, ...]) -> str:
        return self._rng.choice(self._draw_pool)  # feedback ignored — this is the floor


class ConsistentGuesser:
    """The yardstick: a fixed opener, then a uniformly random still-consistent word."""

    needs_consistent = True

    def __init__(self, opener: str = DEFAULT_OPENER, seed: int = 0) -> None:
        self._opener = _validate_opener(opener)
        self._rng = Random(seed)
        logger.info("ConsistentGuesser: opener=%r (seed=%d)", self._opener, seed)

    def choose(self, turns: tuple[Turn, ...], consistent: tuple[str, ...]) -> str:
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

    def __init__(self, opener: str = DEFAULT_OPENER) -> None:
        self._opener = _validate_opener(opener)
        logger.info("InfoMaxGuesser: opener=%r", self._opener)

    def choose(self, turns: tuple[Turn, ...], consistent: tuple[str, ...]) -> str:
        if not turns:
            return self._opener
        _require_candidates(consistent)
        if len(consistent) == 1:
            return consistent[0]
        # Greedy one-ply: the candidate leaving the fewest expected still-consistent answers.
        # `min` is deterministic — `consistent` keeps the sorted-pool order, so ties break first.
        return min(consistent, key=lambda g: expected_remaining(g, consistent))


def play(guesser: Guesser, secret: str, *, pool: tuple[str, ...], max_guesses: int = 6) -> Game:
    """Drive one full game with `guesser`. `pool` is the base set for consistency tracking.

    For `needs_consistent` guessers, `pool` is narrowed incrementally (one clue per turn) and passed
    to `choose`; for the floor it is unused. Use `pool=valid_guesses` for the consistent guesser and
    `pool=answers` for the info-max teacher (spec §4.3, §5.4).
    """
    if guesser.needs_consistent and secret.lower() not in {w.lower() for w in pool}:
        # Without this the consistent set can empty out mid-game (same guard as play_game / reward).
        raise ValueError(f"secret {secret!r} must be in pool for a consistency-tracking guesser")
    game = Game(secret, max_guesses=max_guesses)
    consistent = pool
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
