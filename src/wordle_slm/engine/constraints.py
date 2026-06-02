"""Still-consistent candidate filtering — the v3 action space (spec §1.5). (Plan: candidate filter)

A word ``w`` is a possible secret given the clues iff every past guess, scored against ``w`` as
the hypothetical secret, reproduces the feedback that was actually observed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from wordle_slm.engine.game import Turn
from wordle_slm.engine.scoring import score

logger = logging.getLogger(__name__)


def _consistent_with_turn(word: str, turn: Turn) -> bool:
    if not turn.valid or turn.feedback is None:
        return True  # an invalid guess carries no constraint
    return score(turn.guess, word) == turn.feedback


def is_consistent(word: str, history: Sequence[Turn]) -> bool:
    """True iff ``word`` could be the secret given every valid turn's observed feedback."""
    word = word.lower()
    return all(_consistent_with_turn(word, turn) for turn in history)


def filter_consistent(candidates: Iterable[str], turn: Turn) -> tuple[str, ...]:
    """Keep only candidates still consistent after one more turn (incremental, O(|candidates|))."""
    return tuple(w for w in candidates if _consistent_with_turn(w.lower(), turn))


def consistent_candidates(history: Sequence[Turn], pool: Iterable[str]) -> tuple[str, ...]:
    """The subset of ``pool`` still consistent with all clues in ``history`` (the action space)."""
    candidates = tuple(w for w in pool if is_consistent(w, history))
    logger.info("consistent candidates: %d (after %d turns)", len(candidates), len(history))
    return candidates
