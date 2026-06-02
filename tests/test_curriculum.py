"""Curriculum + replay tests (Plan: I)."""

from __future__ import annotations

from random import Random

import pytest

from wordle_slm.config import CurriculumConfig
from wordle_slm.rl import Curriculum


def _train(n: int) -> tuple[str, ...]:
    return tuple(f"word{i}" for i in range(n))


def test_starts_at_first_tier_size() -> None:
    cur = Curriculum(_train(2000), CurriculumConfig())  # tiers (200, 500, 1000, None)
    assert cur.tier_index == 0
    assert len(cur.current_words()) == 200


def test_promotion_advances_only_when_threshold_met() -> None:
    cfg = CurriculumConfig()
    cur = Curriculum(_train(2000), cfg)
    assert cur.maybe_promote(cfg.promote_threshold - 0.01) is False
    assert cur.tier_index == 0
    assert cur.maybe_promote(cfg.promote_threshold) is True
    assert cur.tier_index == 1
    assert len(cur.current_words()) == 500


def test_cannot_promote_past_last_tier() -> None:
    cfg = CurriculumConfig()
    cur = Curriculum(_train(2000), cfg)
    for _ in range(10):
        cur.maybe_promote(1.0)
    assert cur.tier_index == len(cfg.tiers) - 1
    assert len(cur.current_words()) == 2000  # None tier = full train


def test_tier_size_clamped_to_train_length() -> None:
    cur = Curriculum(_train(50), CurriculumConfig())  # smaller than tier 0 (200)
    assert len(cur.current_words()) == 50


def test_sample_from_current_tier_when_no_replay() -> None:
    cur = Curriculum(_train(2000), CurriculumConfig(replay_prob=0.0))
    rng = Random(0)
    tier = set(cur.current_words())
    for _ in range(20):
        assert cur.sample(rng) in tier


def test_sample_from_replay_when_prob_one() -> None:
    cur = Curriculum(_train(2000), CurriculumConfig(replay_prob=1.0))
    cur.record_loss("word1999")  # a hard word outside tier 0
    assert cur.sample(Random(0)) == "word1999"


def test_replay_queue_is_bounded() -> None:
    cur = Curriculum(_train(2000), CurriculumConfig(replay_capacity=3, replay_prob=1.0))
    for i in range(5):
        cur.record_loss(f"word{i}")
    seen = {cur.sample(Random(s)) for s in range(50)}
    assert seen <= {"word2", "word3", "word4"}  # the two oldest were dropped


def test_sample_mixes_replay_and_tier_when_prob_between_zero_and_one() -> None:
    cur = Curriculum(_train(2000), CurriculumConfig(replay_prob=0.5))
    cur.record_loss("word1999")  # outside tier 0 (first 200), so it can only come from replay
    tier = set(cur.current_words())
    rng = Random(0)
    got_replay = got_tier = False
    for _ in range(200):
        sampled = cur.sample(rng)
        if sampled == "word1999":
            got_replay = True
        elif sampled in tier:
            got_tier = True
    assert got_replay and got_tier  # both branches exercised at 0 < prob < 1


def test_invalid_tiers_raise() -> None:
    # empty, non-positive, non-increasing (shrinks/duplicates), and None-not-last all rejected.
    for bad in ((), (0, None), (200, 50, None), (10, 10, None), (None, 200)):
        with pytest.raises(ValueError):
            Curriculum(_train(100), CurriculumConfig(tiers=bad))
