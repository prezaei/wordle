"""Baseline + teacher tests (Plan: J).

Contracts: the floor ignores feedback and draws only from its pool; the consistent guesser opens
fixed and never violates a clue; the info-max teacher opens fixed then greedily minimizes the
expected still-consistent set (exact argmin), solving easy secrets fast.
"""

from __future__ import annotations

import pytest

from wordle_slm.baselines import (
    ConsistentGuesser,
    InfoMaxGuesser,
    RandomGuesser,
    expected_remaining,
    play,
)
from wordle_slm.data import load_answers, load_valid_guesses
from wordle_slm.engine import Color, Game, Status, Turn, filter_consistent, is_consistent, score

# A hand-computed partition: "slate" never used here so values are independent of the real lists.
# guess "aaaaa" vs each answer scores G,X,X,X,X (one green, no other a) -> all 3 share one pattern
# -> E = (3/3)*3 = 3.0. guess "abcde" gives 3 distinct patterns -> E = 3*((1/3)*1) = 1.0.
_HAND_ANSWERS = ("abcde", "abcdz", "abczz")


def _turn(guess: str, secret: str) -> Turn:
    return Turn(guess=guess, feedback=score(guess, secret), valid=True)


# --- RandomGuesser (the floor) ---------------------------------------------------------------


def test_random_guesser_needs_no_consistency() -> None:
    assert RandomGuesser(draw_pool=("slate",)).needs_consistent is False


def test_random_guesser_draws_the_whole_pool_and_nothing_else() -> None:
    pool = ("slate", "crane", "money")
    rg = RandomGuesser(draw_pool=pool, seed=1)
    drawn = {rg.choose((), ()) for _ in range(50)}
    assert drawn == set(pool)  # every word appears (not stuck) and nothing outside the pool


def test_random_guesser_empty_pool_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RandomGuesser(draw_pool=())


def test_random_guesser_is_seeded_deterministic() -> None:
    pool = load_answers()
    a = [RandomGuesser(draw_pool=pool, seed=7).choose((), ()) for _ in range(3)]
    b = [RandomGuesser(draw_pool=pool, seed=7).choose((), ()) for _ in range(3)]
    assert a == b


def test_random_guesser_singleton_pool_wins_first_turn() -> None:
    secret = load_answers()[0]
    game = play(RandomGuesser(draw_pool=(secret,)), secret, pool=load_answers())
    assert game.status is Status.WIN and game.guesses_used == 1


def test_floor_does_not_require_secret_in_pool() -> None:
    # The floor ignores consistency, so play() must not demand secret ∈ pool for it.
    secret = load_answers()[0]
    game = play(RandomGuesser(draw_pool=(secret,)), secret, pool=("aaaaa", "bbbbb"))
    assert game.won


def test_floor_plays_a_full_game_over_the_answer_pool() -> None:
    # Exercises play() with the floor and no explicit pool (uses default_pool = draw_pool).
    answers = load_answers()
    game = play(RandomGuesser(draw_pool=answers, seed=0), answers[0])
    assert game.status in (Status.WIN, Status.LOSE)
    assert game.guesses_used == len(game.turns) <= 6


def test_play_respects_max_guesses_and_records_a_loss() -> None:
    answers = load_answers()
    secret = next(w for w in answers if w != "slate")  # 'slate' never solves it
    game = play(RandomGuesser(draw_pool=("slate",)), secret, max_guesses=2)
    assert game.status is Status.LOSE
    assert not game.won and game.guesses_used == 2


def test_default_pools_are_wired_per_guesser() -> None:
    assert InfoMaxGuesser().default_pool == load_answers()
    assert ConsistentGuesser().default_pool == load_valid_guesses()
    pool = ("slate", "crane")
    assert RandomGuesser(draw_pool=pool).default_pool == pool


# --- ConsistentGuesser (the yardstick) -------------------------------------------------------


def test_consistent_guesser_opens_with_fixed_word() -> None:
    game = play(ConsistentGuesser(opener="slate", seed=0), load_answers()[0], pool=load_answers())
    assert game.turns[0].guess == "slate"


def test_consistent_guesser_invalid_opener_raises() -> None:
    with pytest.raises(ValueError, match="not a valid guess"):
        ConsistentGuesser(opener="zzzzz")


def test_consistent_guesser_never_violates_a_clue() -> None:
    answers, valid = load_answers(), load_valid_guesses()
    cg = ConsistentGuesser(seed=0)
    for secret in (answers[0], answers[500], answers[1500]):
        game = play(cg, secret, pool=valid)
        history: list[Turn] = []
        for turn in game.turns:
            assert is_consistent(turn.guess, history)  # each guess respects all prior clues
            history.append(turn)


def test_consistent_guesser_wins_most_over_answers() -> None:
    answers = load_answers()
    cg = ConsistentGuesser(seed=0)
    wins = sum(play(cg, s, pool=answers).won for s in answers[:10])
    assert wins / 10 >= 0.6  # honest yardstick is ~96–99%; 0.6 is a safe floor for the assert


def test_consistent_guesser_choose_empty_candidates_raises() -> None:
    cg = ConsistentGuesser(seed=0)
    with pytest.raises(RuntimeError, match="no consistent candidates"):
        cg.choose((_turn("slate", "money"),), ())


def test_play_requires_secret_in_pool_for_consistent_guesser() -> None:
    with pytest.raises(ValueError, match="must be in pool"):
        play(ConsistentGuesser(seed=0), "aaaaa", pool=load_answers())


# --- expected_remaining + InfoMaxGuesser (the near-optimal teacher) ---------------------------


def test_expected_remaining_exact_no_information() -> None:
    assert expected_remaining("aaaaa", _HAND_ANSWERS) == 3.0


def test_expected_remaining_exact_perfect_split() -> None:
    assert expected_remaining("abcde", _HAND_ANSWERS) == 1.0


def test_expected_remaining_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        expected_remaining("slate", ())


def test_infomax_opens_with_fixed_word() -> None:
    game = play(InfoMaxGuesser(opener="slate"), load_answers()[0], pool=load_answers())
    assert game.turns[0].guess == "slate"


def test_infomax_invalid_opener_raises() -> None:
    with pytest.raises(ValueError, match="not a valid guess"):
        InfoMaxGuesser(opener="zzzzz")


def test_infomax_single_candidate_returns_it() -> None:
    assert InfoMaxGuesser().choose((_turn("slate", "money"),), ("money",)) == "money"


def test_infomax_choose_empty_candidates_raises() -> None:
    with pytest.raises(RuntimeError, match="no consistent candidates"):
        InfoMaxGuesser().choose((_turn("slate", "money"),), ())


def test_infomax_pick_is_the_argmin_of_expected_remaining() -> None:
    # On a real post-opener board, the info-max pick must be exactly the candidate minimizing the
    # expected still-consistent set — the defining contract of the near-optimal teacher.
    answers = load_answers()
    game = Game("money")
    candidates = filter_consistent(answers, game.guess("slate"))
    assert len(candidates) >= 3
    pick = InfoMaxGuesser().choose(game.turns, candidates)
    best = min(candidates, key=lambda c: expected_remaining(c, candidates))
    assert pick == best
    assert expected_remaining(pick, candidates) == min(
        expected_remaining(c, candidates) for c in candidates
    )


def test_infomax_shrinks_more_than_a_random_consistent_pick() -> None:
    # The random-consistent guesser's expected outcome is the mean over candidates; the info-max
    # pick is strictly better on this board.
    answers = load_answers()
    game = Game("money")
    candidates = filter_consistent(answers, game.guess("slate"))
    pick_er = expected_remaining(
        min(candidates, key=lambda c: expected_remaining(c, candidates)), candidates
    )
    mean_er = sum(expected_remaining(c, candidates) for c in candidates) / len(candidates)
    assert pick_er < mean_er


def test_infomax_solves_easy_secrets_in_four() -> None:
    answers = load_answers()
    im = InfoMaxGuesser()  # uses its default_pool (the answer list) — no explicit pool needed
    for secret in ("crane", "money", "light", "abbey"):
        assert secret in answers
        game = play(im, secret)
        assert game.won and game.guesses_used <= 4


def test_winning_game_ends_on_an_all_green_turn() -> None:
    game = play(InfoMaxGuesser(), "crane")
    assert game.won
    assert game.guesses_used == len(game.turns)  # no phantom turns
    last = game.turns[-1]
    assert last.feedback is not None and all(c is Color.GREEN for c in last.feedback)
