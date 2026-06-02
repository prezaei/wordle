"""Shaped per-guess reward tests (Plan: H; spec §6.4).

Exact values via hand-built feedback (so greens/yellows are controlled), plus the dominance
inequalities and a real-game integration check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wordle_slm.config import RewardConfig
from wordle_slm.engine import Color, Game, Status, Turn
from wordle_slm.rl.reward import compute_reward

_FB = {"G": Color.GREEN, "Y": Color.YELLOW, "X": Color.GRAY}


def _turn(guess: str, fb: str | None) -> Turn:
    if fb is None:
        return Turn(guess, None, False)  # invalid guess
    return Turn(guess, tuple(_FB[c] for c in fb), True)


def _game(turns: list[Turn], *, status: Status = Status.ONGOING, max_guesses: int = 6):
    # compute_reward reads only turns / status / max_guesses / guesses_used.
    return SimpleNamespace(
        turns=turns, status=status, max_guesses=max_guesses, guesses_used=len(turns)
    )


def test_a_new_green_is_paid_once_and_reconfirming_pays_zero() -> None:
    cfg = RewardConfig()
    # turn1 greens pos0 'a' (pays a + the min-count b); turn2 re-confirms it (pays 0).
    g = _game([_turn("abcde", "GXXXX"), _turn("afghi", "GXXXX")])
    b = compute_reward(g, cfg)
    assert b.letter_progress == pytest.approx(cfg.a + cfg.b)  # only turn1 paid


def test_yellow_then_green_pays_b_then_a_never_double() -> None:
    cfg = RewardConfig()
    # 'a' yellow in turn1 (pays b), green in turn2 (pays a, not another b). Fresh grays each turn.
    g = _game([_turn("abcde", "YXXXX"), _turn("afghi", "GXXXX")])
    b = compute_reward(g, cfg)
    assert b.letter_progress == pytest.approx(cfg.a + cfg.b)


def test_duplicate_letter_credited_single_b() -> None:
    cfg = RewardConfig()
    # 'a' appears twice but only one non-gray (pos0 yellow) -> single b.
    g = _game([_turn("aabcd", "YXXXX")])
    b = compute_reward(g, cfg)
    assert b.letter_progress == pytest.approx(cfg.b)


def test_invalid_guess_penalised_and_no_progress() -> None:
    cfg = RewardConfig()
    g = _game([_turn("zzzzz", None)])
    b = compute_reward(g, cfg)
    assert b.invalid_penalty == pytest.approx(cfg.p_invalid)
    assert b.letter_progress == 0.0
    assert b.total == pytest.approx(-cfg.p_invalid - cfg.c)


def test_dropping_a_known_green_is_a_clue_violation() -> None:
    cfg = RewardConfig()
    # turn1 fixes pos0='a' green; turn2 puts 'f' at pos0 (drops the green) with fresh grays.
    g = _game([_turn("abcde", "GXXXX"), _turn("fghij", "XXXXX")])
    b = compute_reward(g, cfg)
    assert b.clue_penalty == pytest.approx(cfg.q)


def test_reusing_a_known_gray_letter_is_a_clue_violation() -> None:
    cfg = RewardConfig()
    # turn1 marks a,b,c,d,e gray; turn2 reuses 'a'.
    g = _game([_turn("abcde", "XXXXX"), _turn("afghi", "XXXXX")])
    b = compute_reward(g, cfg)
    assert b.clue_penalty == pytest.approx(cfg.q)


def test_faster_win_scores_higher_by_win_speed() -> None:
    cfg = RewardConfig()
    fast = _game([_turn("crane", "GGGGG")], status=Status.WIN)
    slow = _game(
        [_turn("aaaaa", "XXXXX"), _turn("bbbbb", "XXXXX"), _turn("crane", "GGGGG")],
        status=Status.WIN,
    )
    fast_terminal = compute_reward(fast, cfg).terminal
    slow_terminal = compute_reward(slow, cfg).terminal
    assert fast_terminal - slow_terminal == pytest.approx(cfg.win_speed * 2)  # 2 fewer guesses


def test_loss_applies_the_loss_penalty() -> None:
    cfg = RewardConfig()
    g = _game([_turn("aaaaa", "XXXXX")], status=Status.LOSE, max_guesses=1)
    assert compute_reward(g, cfg).terminal == pytest.approx(-cfg.loss_penalty)


def test_reward_dominance_inequalities_hold() -> None:
    cfg = RewardConfig()
    assert cfg.p_invalid > cfg.b  # invalid is worse than any honest progress
    assert cfg.q > cfg.b  # a clue violation is worse than any honest progress
    # max farmable progress in a slow game (5 greens, each also raising min-count) < a win.
    assert 5 * cfg.a + 5 * cfg.b < cfg.win_base


def test_real_game_win_is_positive_and_dominates_farming() -> None:
    cfg = RewardConfig()
    game = Game("crane")
    game.guess("slate")
    game.guess("crane")  # win in 2
    assert game.status is Status.WIN
    b = compute_reward(game, cfg)
    assert b.total > 5 * cfg.a + 5 * cfg.b  # a win beats the most you could farm
