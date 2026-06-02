"""Shaped per-guess reward for the free-generation policy (spec §6.4). (Plan: H)

Rewards real, clue-respecting words so the generator *learns to play*. Per game, summed over
guesses, against a knowledge state carried across the game (not per-turn feedback):

- **Letter progress** `a·new_greens + b·new_yellows` — a position pays its green bonus once; a
  yellow→green upgrade pays `b` then `a` (never double); a duplicate letter is credited only when it
  raises the known min-count; re-confirming a known constraint pays 0 (no farming).
- **Invalid-word penalty** `−p_invalid` — a non-word guess consumes the turn, no progress.
- **Clue-violation penalty** `−q` — dropping a known green or reusing a known-gray letter (`q > b`).
- **Step cost** `−c` per guess.
- **Terminal** `+(win_base + win_speed·(max_guesses − t))` on a win, `−loss_penalty` on a loss.

Dominance (asserted in tests): `p_invalid > b`, `q > b`, and max farmable progress `< win_base`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from wordle_slm.config import RewardConfig
from wordle_slm.engine.game import Game, Status
from wordle_slm.engine.scoring import Color

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardBreakdown:
    letter_progress: float
    invalid_penalty: float
    clue_penalty: float
    step_cost: float
    terminal: float

    @property
    def total(self) -> float:
        return (
            self.letter_progress
            - self.invalid_penalty
            - self.clue_penalty
            - self.step_cost
            + self.terminal
        )


def _violates_clue(guess: str, green_known: dict[int, str], gray_known: set[str]) -> bool:
    """A confirmed-clue violation: drops a known green, or reuses a known-absent letter."""
    drops_green = any(guess[pos] != letter for pos, letter in green_known.items())
    reuses_gray = any(ch in gray_known for ch in guess)
    return drops_green or reuses_gray


def compute_reward(game: Game, config: RewardConfig) -> RewardBreakdown:
    """Shaped reward for a finished/ongoing game (spec §6.4).

    Validity, greens, and yellows are read from each turn the engine already scored — no candidate
    pool is needed (the model generates freely; the engine judged each guess).
    """
    green_known: dict[int, str] = {}  # position -> the letter known green there
    min_count: dict[str, int] = {}  # letter -> known minimum count in the answer
    gray_known: set[str] = set()  # letters confirmed absent

    letter_progress = 0.0
    invalid_penalty = 0.0
    clue_penalty = 0.0
    for turn in game.turns:
        if not turn.valid or turn.feedback is None:
            invalid_penalty += config.p_invalid  # consumes the turn, no letter progress
            continue

        if _violates_clue(turn.guess, green_known, gray_known):
            clue_penalty += config.q

        # New greens: positions GREEN this turn not already known green (pay `a` once).
        for pos, color in enumerate(turn.feedback):
            if color is Color.GREEN and pos not in green_known:
                letter_progress += config.a
                green_known[pos] = turn.guess[pos]

        # New yellows / duplicates: credit only when a letter's observed non-gray count exceeds its
        # known min-count (pay `b` per newly-required occurrence); update known min-counts.
        for letter in set(turn.guess):
            obs = sum(
                1
                for pos, ch in enumerate(turn.guess)
                if ch == letter and turn.feedback[pos] is not Color.GRAY
            )
            if obs > min_count.get(letter, 0):
                letter_progress += config.b * (obs - min_count.get(letter, 0))
                min_count[letter] = obs
            if obs == 0:  # this letter showed only gray -> confirmed absent
                gray_known.add(letter)

    step_cost = config.c * len(game.turns)
    terminal = 0.0
    if game.status is Status.WIN:
        terminal = config.win_base + config.win_speed * (game.max_guesses - game.guesses_used)
    elif game.status is Status.LOSE:
        terminal = -config.loss_penalty

    breakdown = RewardBreakdown(
        letter_progress=letter_progress,
        invalid_penalty=invalid_penalty,
        clue_penalty=clue_penalty,
        step_cost=step_cost,
        terminal=terminal,
    )
    logger.info("reward total=%.4f (%s)", breakdown.total, breakdown)
    return breakdown
