"""v3 reward tests (Plan: H): information gain + speed-scaled win bonus."""

from __future__ import annotations

import math

import pytest

from wordle_slm.config import RewardConfig
from wordle_slm.data import load_answers, load_valid_guesses
from wordle_slm.engine import Game, Status
from wordle_slm.rl import compute_reward


def _invalid_word() -> str:
    valid = set(load_valid_guesses())
    cand = "aaaaa"
    while cand in valid:
        cand = cand[:-1] + chr((ord(cand[-1]) - ord("a") + 1) % 26 + ord("a"))
    return cand


def test_win_in_one_reward_is_exact() -> None:
    cfg = RewardConfig()
    pool = load_answers()
    secret = pool[0]  # guaranteed in the candidate pool
    g = Game(secret)
    g.guess(secret)
    assert g.won
    b = compute_reward(g, cfg, pool)
    # one all-green guess collapses the candidate set to {secret}.
    expected_ig = cfg.info_gain_weight * math.log(len(pool) / 1)
    expected_total = expected_ig - cfg.step_cost + cfg.win_base + cfg.win_speed * (6 - 1)
    assert b.info_gain == pytest.approx(expected_ig)
    assert b.total == pytest.approx(expected_total)


def test_info_gain_telescopes_to_log_pool_for_any_win() -> None:
    # Sum of log(n_before/n_after) over a winning game = log(|pool| / 1), independent of path.
    cfg = RewardConfig()
    pool = load_answers()
    secret = pool[0]
    g = Game(secret)
    g.guess("slate")  # narrows
    g.guess(secret)  # win
    assert g.won
    b = compute_reward(g, cfg, pool)
    assert b.info_gain == pytest.approx(cfg.info_gain_weight * math.log(len(pool)))
    assert b.total > cfg.win_base  # win bonus + info gain dominate


def test_loss_with_invalid_guesses_has_zero_info_gain() -> None:
    cfg = RewardConfig()
    pool = load_answers()
    bad = _invalid_word()
    g = Game(pool[0])
    for _ in range(6):
        g.guess(bad)
    assert g.status is Status.LOSE
    b = compute_reward(g, cfg, pool)
    assert b.info_gain == pytest.approx(0.0)  # invalid guesses don't narrow the field
    assert b.total == pytest.approx(-6 * cfg.step_cost - cfg.loss_penalty)


def test_faster_win_scores_higher() -> None:
    cfg = RewardConfig()
    pool = load_answers()
    secret = pool[0]
    fast = Game(secret)
    fast.guess(secret)  # win in 1
    slow = Game(secret)
    slow.guess("slate")
    slow.guess(secret)  # win in 2
    # info_gain telescopes equal; the speed bonus + fewer step costs make the faster win higher.
    assert compute_reward(fast, cfg, pool).total > compute_reward(slow, cfg, pool).total
