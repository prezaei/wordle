"""Head-start (SFT) teacher data (spec §5.4; Plan: M).

Plays the train answers with a blend of teachers — ~70% feedback-consistent (`ConsistentGuesser`)
and ~30% near-optimal (`InfoMaxGuesser`), spec §5.4 — each opening with a varied strong starter.
Each played game is a transcript; the SFT trainer (Plan: N) encodes it via `encode_completed_game`
and imitates the teacher's guess letters (the §5.2 grammar, masked to the guess-letter positions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from random import Random

from wordle_slm.baselines.policies import ConsistentGuesser, InfoMaxGuesser, play
from wordle_slm.data import load_answers, load_valid_guesses
from wordle_slm.engine import Game

logger = logging.getLogger(__name__)

# Strong, varied openers (all verified valid guesses) — spec §5.4 "varied openers".
DEFAULT_OPENERS: tuple[str, ...] = ("slate", "crane", "trace", "stare", "raise", "crate")


@dataclass(frozen=True)
class Transcript:
    """One teacher-played game and how it was produced."""

    secret: str
    teacher: str  # "consistent" | "infomax"
    opener: str
    game: Game


def generate_transcripts(
    secrets: tuple[str, ...],
    *,
    weak_frac: float = 0.70,
    openers: tuple[str, ...] = DEFAULT_OPENERS,
    seed: int = 0,
) -> list[Transcript]:
    """Play each secret with a blended, varied-opener teacher → SFT transcripts (spec §5.4).

    `weak_frac` of games use the consistent guesser (over the valid list); the rest use the
    near-optimal info-max teacher (over the answer pool). Deterministic for a given `seed`.
    """
    if not 0.0 <= weak_frac <= 1.0:
        raise ValueError(f"weak_frac must be in [0, 1], got {weak_frac}")
    rng = Random(seed)
    valid, answers = load_valid_guesses(), load_answers()
    transcripts: list[Transcript] = []
    for index, secret in enumerate(secrets):
        opener = rng.choice(openers)
        if rng.random() < weak_frac:
            teacher_name = "consistent"
            game = play(ConsistentGuesser(opener=opener, seed=seed + index, pool=valid), secret)
        else:
            teacher_name = "infomax"
            game = play(InfoMaxGuesser(opener=opener, pool=answers), secret)
        transcripts.append(Transcript(secret, teacher_name, opener, game))
    weak = sum(1 for t in transcripts if t.teacher == "consistent")
    logger.info(
        "generated %d transcripts: %d consistent / %d infomax (weak_frac=%.2f)",
        len(transcripts),
        weak,
        len(transcripts) - weak,
        weak_frac,
    )
    return transcripts
