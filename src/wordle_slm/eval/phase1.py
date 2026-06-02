"""Phase-1 eval: is the model ready for RL? (spec §5.6; Plan: P)

Two readiness bars on the SFT model, the RL go/no-go gate:
- **valid-word rate** — fraction of generated guesses that are real words (≥95%). Direct evidence
  the model learned word structure, not a list.
- **green-retention** — over turns where a green is already known, the next guess keeps every known
  green in place (≥80%). Evidence it respects confirmed clues.

The metrics are pure functions over played `Game`s, so they're exactly testable; `evaluate_phase1`
plays greedy games with the model and applies them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from wordle_slm.config import SFTConfig
from wordle_slm.data import is_valid
from wordle_slm.engine import Color, Game
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Phase1Report:
    valid_word_rate: float
    green_retention: float
    n_games: int
    n_guesses: int

    def passes(self, config: SFTConfig) -> bool:
        return (
            self.valid_word_rate >= config.valid_word_bar
            and self.green_retention >= config.clue_respect_bar
        )


def valid_word_rate(games: Sequence[Game]) -> float:
    """Fraction of all guessed words that are valid (over every guess in every game)."""
    total = sum(len(g.turns) for g in games)
    if total == 0:
        return 0.0
    valid = sum(is_valid(turn.guess) for g in games for turn in g.turns)
    return valid / total


def _known_greens(turns: Sequence) -> dict[int, str]:
    """Positions known GREEN (position → letter) from the valid turns so far."""
    greens: dict[int, str] = {}
    for turn in turns:
        if turn.feedback is None:
            continue
        for pos, color in enumerate(turn.feedback):
            if color is Color.GREEN:
                greens[pos] = turn.guess[pos]
    return greens


def green_retention(games: Sequence[Game]) -> float:
    """Over turns played with ≥1 already-known green, the fraction that keep every known green.

    Returns 1.0 when no such turn ever arises (vacuously clue-respecting — nothing to drop).
    """
    opportunities = 0
    kept = 0
    for game in games:
        for i, turn in enumerate(game.turns):
            greens = _known_greens(game.turns[:i])  # greens known BEFORE this guess
            if not greens:
                continue
            opportunities += 1
            if all(turn.guess[pos] == letter for pos, letter in greens.items()):
                kept += 1
    return kept / opportunities if opportunities else 1.0


def evaluate_phase1(
    model: WordleGenerator,
    tokenizer: Tokenizer,
    secrets: Sequence[str],
    *,
    device: str = "cpu",
) -> Phase1Report:
    """Play greedy games with the model over `secrets` and measure the Phase-1 readiness bars."""
    from wordle_slm.rl.rollout import play_game

    games = [play_game(model, tokenizer, s, sample=False, device=device) for s in secrets]
    report = Phase1Report(
        valid_word_rate=valid_word_rate(games),
        green_retention=green_retention(games),
        n_games=len(games),
        n_guesses=sum(len(g.turns) for g in games),
    )
    logger.info(
        "phase-1 eval over %d games: valid-word=%.3f green-retention=%.3f",
        report.n_games,
        report.valid_word_rate,
        report.green_retention,
    )
    return report
