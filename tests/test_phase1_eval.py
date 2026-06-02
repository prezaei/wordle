"""Phase-1 readiness-eval tests (Plan: P; spec §5.6)."""

from __future__ import annotations

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.engine import Color, Game, Turn
from wordle_slm.eval.phase1 import (
    Phase1Report,
    evaluate_phase1,
    green_retention,
    valid_word_rate,
)
from wordle_slm.model import Tokenizer, WordleGenerator

_FB = {"G": Color.GREEN, "Y": Color.YELLOW, "X": Color.GRAY}


def _turn(guess: str, fb: str | None) -> Turn:
    if fb is None:
        return Turn(guess, None, False)
    return Turn(guess, tuple(_FB[c] for c in fb), True)


def _game(*turns: Turn) -> Game:
    g = Game("crane")
    g.turns.extend(turns)  # inject a controlled transcript for the pure metrics
    return g


# --- valid_word_rate --------------------------------------------------------------------------


def test_valid_word_rate_counts_valid_guesses() -> None:
    # 2 valid ("crane","slate") + 1 invalid ("zzzzz") -> 2/3
    g = _game(_turn("crane", "GGGGG"), _turn("slate", "XXXXX"), _turn("zzzzz", None))
    assert valid_word_rate([g]) == 2 / 3


def test_valid_word_rate_empty_is_zero() -> None:
    assert valid_word_rate([_game()]) == 0.0


# --- green_retention --------------------------------------------------------------------------


def test_green_retention_all_kept() -> None:
    # turn1 fixes pos0='c' green; turn2 keeps 'c' at pos0 -> 1 opportunity, kept.
    g = _game(_turn("crane", "GXXXX"), _turn("civic", "GXXXX"))
    assert green_retention([g]) == 1.0


def test_green_retention_dropped() -> None:
    # turn1 fixes pos0='c' green; turn2 puts 's' at pos0 -> dropped.
    g = _game(_turn("crane", "GXXXX"), _turn("slate", "XXXXX"))
    assert green_retention([g]) == 0.0


def test_green_retention_vacuous_when_no_greens_known() -> None:
    # no green is ever established -> no opportunities -> vacuously 1.0
    assert green_retention([_game(_turn("slate", "XXXXX"), _turn("dingo", "XXXXX"))]) == 1.0


def test_green_retention_counts_only_post_green_turns() -> None:
    # opportunities = turns AFTER a green is known: turn2 keeps, turn3 drops -> 1/2.
    g = _game(
        _turn("crane", "GXXXX"),  # establishes pos0='c'
        _turn("civic", "GXXXX"),  # keeps c (opportunity 1, kept)
        _turn("slate", "XXXXX"),  # drops c (opportunity 2, not kept)
    )
    assert green_retention([g]) == 0.5


# --- report bars + integration ----------------------------------------------------------------


def test_report_passes_only_when_both_bars_clear() -> None:
    cfg = SFTConfig()  # valid_word_bar=0.95, clue_respect_bar=0.80
    assert Phase1Report(0.96, 0.85, 10, 30).passes(cfg)
    assert not Phase1Report(0.90, 0.99, 10, 30).passes(cfg)  # valid-word too low
    assert not Phase1Report(0.99, 0.50, 10, 30).passes(cfg)  # clue-respect too low


def test_evaluate_phase1_runs_on_a_model() -> None:
    from wordle_slm.data import load_answers

    torch.manual_seed(0)
    tok = Tokenizer()
    model = WordleGenerator(
        ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0), tok.vocab_size
    )
    report = evaluate_phase1(model, tok, load_answers()[:5])
    assert report.n_games == 5
    assert 0.0 <= report.valid_word_rate <= 1.0 and 0.0 <= report.green_retention <= 1.0
