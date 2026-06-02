"""Head-start teacher-data tests (Plan: M; spec §5.4)."""

from __future__ import annotations

import pytest

from wordle_slm.data import is_valid, split
from wordle_slm.engine import is_consistent
from wordle_slm.teacher import DEFAULT_OPENERS, generate_transcripts


def _train(n: int) -> tuple[str, ...]:
    train, _ = split(seed=0)
    return train[:n]


def test_default_openers_are_valid_words() -> None:
    assert all(is_valid(o) for o in DEFAULT_OPENERS) and len(DEFAULT_OPENERS) >= 3


def test_blend_is_roughly_seventy_thirty() -> None:
    transcripts = generate_transcripts(_train(40), weak_frac=0.7, seed=0)
    weak = sum(1 for t in transcripts if t.teacher == "consistent") / len(transcripts)
    assert 0.5 <= weak <= 0.9  # ~70/30 with sampling slack over 40 games


def test_blend_extremes_are_exact() -> None:
    assert all(t.teacher == "consistent" for t in generate_transcripts(_train(10), weak_frac=1.0))
    assert all(t.teacher == "infomax" for t in generate_transcripts(_train(10), weak_frac=0.0))


def test_openers_vary() -> None:
    openers = {t.opener for t in generate_transcripts(_train(40), seed=0)}
    assert len(openers) > 1 and openers <= set(DEFAULT_OPENERS)


def test_each_transcript_opens_with_its_opener_and_finishes() -> None:
    for t in generate_transcripts(_train(8), seed=0):
        assert t.game.turns[0].guess == t.opener
        assert t.game.status.value in ("win", "lose")


def test_consistent_teacher_never_violates_a_clue() -> None:
    for t in generate_transcripts(_train(6), weak_frac=1.0, seed=0):
        history: list = []
        for turn in t.game.turns:
            assert is_consistent(turn.guess, history)
            history.append(turn)


def test_weak_frac_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="weak_frac"):
        generate_transcripts(_train(4), weak_frac=1.5)


def test_is_deterministic_with_seed() -> None:
    a = generate_transcripts(_train(12), seed=0)
    b = generate_transcripts(_train(12), seed=0)
    assert [(t.teacher, t.opener) for t in a] == [(t.teacher, t.opener) for t in b]
