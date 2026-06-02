"""Curriculum + replay tests (Plan: I)."""

from __future__ import annotations

from random import Random

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
