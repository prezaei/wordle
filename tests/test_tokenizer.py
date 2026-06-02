"""Tokenizer tests (Plan: C): size, stable ids, round-trips, errors."""

from __future__ import annotations

import pytest

from wordle_slm.model import Tokenizer


def test_vocab_size_is_34() -> None:
    assert Tokenizer().vocab_size == 34


def test_pad_is_zero_and_special_ids_are_stable() -> None:
    t = Tokenizer()
    assert t.pad_id == 0
    assert (t.bos_id, t.eos_id, t.sep_id, t.guess_id) == (1, 2, 3, 4)
    assert (t.green_id, t.yellow_id, t.gray_id) == (5, 6, 7)
    assert t.token_to_id("a") == 8
    assert t.token_to_id("z") == 33


def test_encode_decode_round_trip_mixed_tokens() -> None:
    t = Tokenizer()
    tokens = ["<BOS>", "c", "r", "a", "n", "e", "<gray>", "<SEP>", "<GUESS>", "<EOS>"]
    assert t.decode(t.encode(tokens)) == tokens


def test_encode_letters_is_case_insensitive_round_trip() -> None:
    t = Tokenizer()
    ids = t.encode_letters("CRANE")
    assert t.decode(ids) == list("crane")


def test_unknown_token_raises() -> None:
    t = Tokenizer()
    with pytest.raises(KeyError):
        t.token_to_id("<NOPE>")


def test_unknown_id_raises() -> None:
    t = Tokenizer()
    with pytest.raises(KeyError):
        t.id_to_token(999)
