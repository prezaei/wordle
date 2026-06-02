"""Board ↔ token serialization tests (Plan: F; spec §5.2)."""

from __future__ import annotations

from wordle_slm.engine import Turn, score
from wordle_slm.model import Tokenizer
from wordle_slm.model.serialization import (
    build_prompt,
    decode_word,
    encode_completed_game,
    guess_letter_target_positions,
)


def _turn(guess: str, secret: str) -> Turn:
    return Turn(guess=guess, feedback=score(guess, secret), valid=True)


def test_build_prompt_turn_one_is_bos_then_guess() -> None:
    tok = Tokenizer()
    assert tok.decode(build_prompt([], tok)) == ["<BOS>", "<GUESS>"]


def test_build_prompt_completed_turn_then_cue() -> None:
    tok = Tokenizer()
    toks = tok.decode(build_prompt([_turn("crane", "night")], tok))
    # <BOS> <GUESS> c r a n e <fb×5> <SEP> <GUESS>
    assert toks[0] == "<BOS>"
    assert toks[1] == "<GUESS>"
    assert toks[2:7] == list("crane")
    assert toks[7:12] == ["<gray>", "<gray>", "<gray>", "<yellow>", "<gray>"]  # crane vs night
    assert toks[12] == "<SEP>"
    assert toks[13] == "<GUESS>"  # the cue to generate the next guess
    assert len(toks) == 14


def test_invalid_turn_encodes_as_letters_and_all_gray() -> None:
    tok = Tokenizer()
    turns = [Turn("zzzzz", None, False), _turn("crane", "night")]
    toks = tok.decode(build_prompt(turns, tok))
    # the invalid turn is NOT skipped — the model must see the spent turn (all-gray feedback)
    assert toks[1] == "<GUESS>"
    assert toks[2:7] == list("zzzzz")
    assert toks[7:12] == ["<gray>"] * 5
    assert toks[12] == "<SEP>"
    assert toks[13] == "<GUESS>"  # the valid turn starts


def test_encode_completed_game_is_bos_turns_eos() -> None:
    tok = Tokenizer()
    toks = tok.decode(encode_completed_game([_turn("crane", "night")], tok))
    assert toks[0] == "<BOS>" and toks[-1] == "<EOS>"
    assert toks[1] == "<GUESS>" and toks[2:7] == list("crane")


def test_guess_letter_target_positions_one_per_guess() -> None:
    tok = Tokenizer()
    ids = encode_completed_game([_turn("crane", "night"), _turn("slate", "night")], tok)
    positions = guess_letter_target_positions(ids, tok)
    assert len(positions) == 10  # 5 per completed guess
    assert [tok.id_to_token(ids[p]) for p in positions] == list("craneslate")


def test_build_prompt_trailing_cue_has_no_target_positions() -> None:
    tok = Tokenizer()
    ids = build_prompt([_turn("crane", "night")], tok)
    positions = guess_letter_target_positions(ids, tok)
    assert len(positions) == 5  # only the completed turn; the trailing cue <GUESS> contributes none
    assert [tok.id_to_token(ids[p]) for p in positions] == list("crane")


def test_decode_word_round_trips() -> None:
    tok = Tokenizer()
    assert decode_word(tok.encode_letters("slate"), tok) == "slate"
