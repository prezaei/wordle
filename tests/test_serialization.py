"""Board/candidate encoding tests (Plan: F)."""

from __future__ import annotations

from wordle_slm.engine import Turn, score
from wordle_slm.model import Tokenizer
from wordle_slm.model.serialization import encode_board, encode_word


def _turn(guess: str, secret: str) -> Turn:
    return Turn(guess=guess, feedback=score(guess, secret), valid=True)


def test_encode_board_structure() -> None:
    tok = Tokenizer()
    toks = tok.decode(encode_board([_turn("crane", "night")], tok))
    assert toks[0] == "<BOS>" and toks[-1] == "<EOS>"
    assert toks[1:6] == list("crane")
    assert toks[6:11] == ["<gray>", "<gray>", "<gray>", "<yellow>", "<gray>"]
    assert toks[11] == "<SEP>"


def test_encode_board_skips_invalid_turns() -> None:
    tok = Tokenizer()
    turns = [Turn("zzzzz", None, False), _turn("crane", "night")]
    toks = tok.decode(encode_board(turns, tok))
    assert toks[1:6] == list("crane")  # only the valid turn is encoded
    assert toks.count("<SEP>") == 1


def test_encode_board_empty_is_bos_eos() -> None:
    tok = Tokenizer()
    assert tok.decode(encode_board([], tok)) == ["<BOS>", "<EOS>"]


def test_encode_word_is_case_insensitive() -> None:
    tok = Tokenizer()
    assert tok.decode(encode_word("CRANE", tok)) == list("crane")
