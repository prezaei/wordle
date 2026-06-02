"""Word-data tests (Plan: A): counts, integrity, leakage, split, probe."""

from __future__ import annotations

from wordle_slm.data import (
    is_valid,
    load_answers,
    load_valid_guesses,
    split,
    train_probe,
)


def test_counts() -> None:
    assert len(load_answers()) == 2315
    assert len(load_valid_guesses()) == 14855


def test_no_duplicates_within_either_list() -> None:
    answers = load_answers()
    valid = load_valid_guesses()
    assert len(set(answers)) == len(answers)
    assert len(set(valid)) == len(valid)


def test_all_entries_are_lowercase_five_letters() -> None:
    for word in load_answers():
        assert len(word) == 5 and word.isalpha() and word.islower()
    for word in load_valid_guesses():
        assert len(word) == 5 and word.isalpha() and word.islower()


def test_answers_subset_of_valid_guesses() -> None:
    assert set(load_answers()) <= set(load_valid_guesses())


def test_split_counts_disjoint_and_cover() -> None:
    train, held = split(seed=0, train_frac=0.80)
    assert (len(train), len(held)) == (1852, 463)
    assert set(train).isdisjoint(set(held))
    assert set(train) | set(held) == set(load_answers())


def test_split_is_stable_for_a_seed() -> None:
    assert split(seed=0) == split(seed=0)


def test_split_seed_changes_the_heldout_set() -> None:
    _, held0 = split(seed=0)
    _, held1 = split(seed=1)
    assert set(held0) != set(held1)


def test_heldout_words_are_valid_guesses_but_never_training_secrets() -> None:
    train, held = split(seed=0)
    # The model MAY guess held-out words (they are valid)...
    assert all(is_valid(w) for w in held)
    # ...but they are never in the training (secret) pool.
    assert set(held).isdisjoint(set(train))


def test_train_probe_matched_to_heldout_subset_and_stable() -> None:
    train, held = split(seed=0)
    probe = train_probe(seed=0)
    assert len(probe) == len(held)  # held-out-matched (~463)
    assert set(probe) <= set(train)  # subset of train
    assert set(probe).isdisjoint(set(held))  # disjoint from held-out
    assert train_probe(seed=0) == train_probe(seed=0)  # stable


def test_train_probe_respects_explicit_size() -> None:
    probe = train_probe(seed=0, size=100)
    assert len(probe) == 100
    assert set(probe) <= set(split(seed=0)[0])


def test_is_valid_true_for_a_real_word_case_insensitive() -> None:
    word = load_valid_guesses()[0]
    assert is_valid(word) is True
    assert is_valid(word.upper()) is True


def test_is_valid_false_for_a_non_member() -> None:
    valid = set(load_valid_guesses())
    # Construct a 5-letter string guaranteed not to be in the dictionary.
    candidate = "aaaaa"
    while candidate in valid:
        last = chr((ord(candidate[-1]) - ord("a") + 1) % 26 + ord("a"))
        candidate = candidate[:-1] + last
    assert is_valid(candidate) is False
