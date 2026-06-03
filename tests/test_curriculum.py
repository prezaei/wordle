"""Curriculum tests — redesigned: diverse pool, difficulty order, robust promotion (Plan I)."""

from __future__ import annotations

from random import Random

import pytest

from wordle_slm.config import CurriculumConfig
from wordle_slm.data import is_valid, load_answers, split
from wordle_slm.rl import Curriculum, build_curriculum_pool, difficulty


def _train(n: int) -> tuple[str, ...]:
    return tuple(f"w{i:05d}" for i in range(n))


# --- tiers + widening -------------------------------------------------------------------------


def test_starts_at_first_tier_size() -> None:
    cur = Curriculum(_train(20000), CurriculumConfig())  # tiers (2000, 6000, 10000, None)
    assert cur.tier_index == 0
    assert len(cur.current_words()) == 2000


def test_promotion_advances_only_when_threshold_met() -> None:
    cfg = CurriculumConfig()
    cur = Curriculum(_train(20000), cfg)
    assert cur.maybe_promote(cfg.promote_threshold - 0.01) is False
    assert cur.tier_index == 0
    assert cur.maybe_promote(cfg.promote_threshold) is True
    assert cur.tier_index == 1
    assert len(cur.current_words()) == 6000


def test_cannot_promote_past_last_tier() -> None:
    cfg = CurriculumConfig()
    cur = Curriculum(_train(20000), cfg)
    for _ in range(10):
        cur.maybe_promote(1.0)
    assert cur.tier_index == len(cfg.tiers) - 1
    assert len(cur.current_words()) == 20000  # None tier = full pool


def test_tier_size_clamped_to_pool_length() -> None:
    cur = Curriculum(_train(50), CurriculumConfig())  # smaller than tier 0 (2000)
    assert len(cur.current_words()) == 50


def test_patience_force_promotes_without_clearing_threshold() -> None:
    # The robust-progress fix: widen after promote_patience evals even below threshold.
    cfg = CurriculumConfig(promote_patience=3)
    cur = Curriculum(_train(20000), cfg)
    assert cur.maybe_promote(0.0) is False  # eval 1
    assert cur.maybe_promote(0.0) is False  # eval 2
    assert cur.maybe_promote(0.0) is True  # eval 3 -> forced widen
    assert cur.tier_index == 1


def test_promotion_resets_the_patience_counter() -> None:
    cfg = CurriculumConfig(promote_patience=2)
    cur = Curriculum(_train(20000), cfg)
    assert cur.maybe_promote(1.0) is True  # promote via threshold (counter resets)
    assert cur.maybe_promote(0.0) is False  # eval 1 on the new tier, not yet at patience
    assert cur.tier_index == 1


# --- replay ------------------------------------------------------------------------------------


def test_sample_from_current_tier_when_no_replay() -> None:
    cur = Curriculum(_train(20000), CurriculumConfig(replay_prob=0.0))
    rng = Random(0)
    tier = set(cur.current_words())
    for _ in range(20):
        assert cur.sample(rng) in tier


def test_sample_from_replay_when_prob_one() -> None:
    cur = Curriculum(_train(20000), CurriculumConfig(replay_prob=1.0))
    cur.record_loss("w19999")  # a hard word outside tier 0
    assert cur.sample(Random(0)) == "w19999"


def test_replay_queue_is_bounded() -> None:
    cur = Curriculum(_train(20000), CurriculumConfig(replay_capacity=3, replay_prob=1.0))
    for i in range(5):
        cur.record_loss(f"h{i}")
    seen = {cur.sample(Random(s)) for s in range(50)}
    assert seen <= {"h2", "h3", "h4"}  # the two oldest were dropped


def test_sample_mixes_replay_and_tier_when_prob_between_zero_and_one() -> None:
    cur = Curriculum(_train(20000), CurriculumConfig(replay_prob=0.5))
    cur.record_loss("w19999")  # outside tier 0 (first 2000), so it can only come from replay
    tier = set(cur.current_words())
    rng = Random(0)
    got_replay = got_tier = False
    for _ in range(200):
        sampled = cur.sample(rng)
        if sampled == "w19999":
            got_replay = True
        elif sampled in tier:
            got_tier = True
    assert got_replay and got_tier  # both branches exercised at 0 < prob < 1


def test_invalid_tiers_raise() -> None:
    # empty, non-positive, non-increasing (shrinks/duplicates), and None-not-last all rejected.
    for bad in ((), (0, None), (200, 50, None), (10, 10, None), (None, 200)):
        with pytest.raises(ValueError):
            Curriculum(_train(100), CurriculumConfig(tiers=bad))


# --- difficulty ordering + diverse pool --------------------------------------------------------


def test_difficulty_ranks_repeated_letters_harder() -> None:
    assert difficulty("crane")[0] == 0  # all distinct -> easiest
    assert difficulty("salsa")[0] == 2  # s,a each repeat once
    assert difficulty("mamma")[0] == 3
    assert sorted(["salsa", "crane", "mamma"], key=difficulty)[0] == "crane"


def test_build_pool_is_diverse_ordered_and_excludes_heldout() -> None:
    pool = build_curriculum_pool(seed=0)
    train_answers, held = split(seed=0)
    answers = set(load_answers())
    assert len(pool) > 7 * len(train_answers)  # full valid list, ~8x the answer-only pool
    assert len(set(pool)) == len(pool)  # no duplicates
    assert set(held).isdisjoint(pool)  # held-out never trainable
    assert all(is_valid(w) for w in pool)  # every secret is a real word
    # common answers come first (the eval distribution), rarer valid words after.
    assert pool[: len(train_answers)] == tuple(sorted(train_answers, key=difficulty))
    assert pool[len(train_answers)] not in answers  # the tail is non-answer valid words


def test_build_pool_first_tier_is_the_answer_distribution() -> None:
    pool = build_curriculum_pool(seed=0)
    train_answers, _ = split(seed=0)
    answers = set(load_answers())
    cur = Curriculum(pool, CurriculumConfig(tiers=(len(train_answers), None)))
    assert all(w in answers for w in cur.current_words())  # tier 0 = answers only
