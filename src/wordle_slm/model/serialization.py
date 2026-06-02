"""Board + candidate encoding for the v3 candidate scorer (spec §1.5; Plan: F).

The board (completed turns) is encoded as a token sequence the scorer's encoder consumes; each
candidate word is encoded as its 5 letter ids. Reuses the char Tokenizer vocab (§5.1).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from wordle_slm.engine.game import Turn
from wordle_slm.engine.scoring import Color
from wordle_slm.model.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_COLOR_TOKEN: dict[Color, str] = {
    Color.GREEN: "<green>",
    Color.YELLOW: "<yellow>",
    Color.GRAY: "<gray>",
}


def encode_board(turns: Sequence[Turn], tokenizer: Tokenizer) -> list[int]:
    """Encode the board: ``<BOS> (guess(5) feedback(5) <SEP>)* <EOS>`` over the VALID turns.

    Invalid guesses carry no board information and are skipped.
    """
    tokens: list[str] = ["<BOS>"]
    for turn in turns:
        if not turn.valid or turn.feedback is None:
            continue
        tokens.extend(turn.guess)
        tokens.extend(_COLOR_TOKEN[c] for c in turn.feedback)
        tokens.append("<SEP>")
    tokens.append("<EOS>")
    return tokenizer.encode(tokens)


def encode_word(word: str, tokenizer: Tokenizer) -> list[int]:
    """Encode a candidate word as its 5 letter ids."""
    return tokenizer.encode_letters(word)
