"""Evaluation: Phase-1 readiness bars + generalization gap / checkpoint selection. (Plan: P, S)"""

from wordle_slm.eval.phase1 import (
    Phase1Report,
    evaluate_phase1,
    green_retention,
    valid_word_rate,
)
from wordle_slm.eval.selection import GapReport, generalization_gap

__all__ = [
    "GapReport",
    "Phase1Report",
    "evaluate_phase1",
    "generalization_gap",
    "green_retention",
    "valid_word_rate",
]
