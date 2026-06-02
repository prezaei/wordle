"""Speed-dominant reward for the restricted-action-space policy (spec §1.5). (Plan: H)

Per game:
- **Information gain** per guess: ``info_gain_weight * log(|C_before| / |C_after|)``, where ``C``
  is the still-consistent candidate set. Rewards guesses that shrink the field fastest.
- **Win bonus, speed-scaled:** ``win_base + win_speed * (max_guesses - t)`` on the winning guess.
- **Step cost:** ``-step_cost`` per guess.
- **Loss penalty:** ``-loss_penalty`` if the game is lost.

(Invalid-word / legal-word terms are gone: under the v3 action space every guess is valid and
consistent, so winning is near-automatic and the reward optimizes guess count.)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from wordle_slm.config import RewardConfig
from wordle_slm.engine.constraints import consistent_candidates
from wordle_slm.engine.game import Game, Status

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardBreakdown:
    info_gain: float
    step_cost: float
    terminal: float

    @property
    def total(self) -> float:
        return self.info_gain - self.step_cost + self.terminal


def compute_reward(game: Game, config: RewardConfig, pool: tuple[str, ...]) -> RewardBreakdown:
    """Trajectory reward for a finished/ongoing game. ``pool`` is the candidate pool (answers)."""
    info_gain = 0.0
    n_before = len(pool)  # C_0 = the full pool (no clues yet)
    history: list = []
    for turn in game.turns:
        history.append(turn)
        n_after = max(1, len(consistent_candidates(history, pool)))  # secret stays consistent
        info_gain += config.info_gain_weight * math.log(n_before / n_after)
        n_before = n_after

    step_cost = config.step_cost * len(game.turns)
    terminal = 0.0
    if game.status is Status.WIN:
        terminal = config.win_base + config.win_speed * (game.max_guesses - game.guesses_used)
    elif game.status is Status.LOSE:
        terminal = -config.loss_penalty

    breakdown = RewardBreakdown(info_gain=info_gain, step_cost=step_cost, terminal=terminal)
    logger.info("reward total=%.4f (%s)", breakdown.total, breakdown)
    return breakdown
