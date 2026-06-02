"""Board ↔ token serialization for the generation policy (spec §5.2; Plan: F).

The model is a char generator: it conditions on the board history and emits the next guess's 5
letters. ONE consistent representation is used for both generation and log-prob recomputation (so
the GRPO importance ratio is exact): `<GUESS>` precedes EVERY guess, each completed turn is
`<GUESS> + 5 letters + 5 feedback + <SEP>`, and the guess being generated is cued by a trailing
`<GUESS>`. Keeping `<GUESS>` in history is a deliberate, documented departure from §5.2's
"current-turn only" wording — it makes generation and recomputation share an identical conditioning
context, and §5.5's "the 5 letters after each `<GUESS>`" mask assumes exactly this.

An invalid guess still consumed a turn, so it appears in history as its 5 letters + 5×`<gray>` (it
learned nothing) — the model must see the turn was spent (spec §4.2 / migration doc M1).
"""

from __future__ import annotations

from collections.abc import Sequence

from wordle_slm.engine import Color, Turn
from wordle_slm.model.tokenizer import Tokenizer

GUESS_LEN = 5
_COLOR_TOKEN: dict[Color, str] = {
    Color.GREEN: "<green>",
    Color.YELLOW: "<yellow>",
    Color.GRAY: "<gray>",
}


def _feedback_ids(turn: Turn, tokenizer: Tokenizer) -> list[int]:
    if turn.feedback is None:  # invalid guess: a turn was spent, nothing learned
        return [tokenizer.gray_id] * GUESS_LEN
    return [tokenizer.token_to_id(_COLOR_TOKEN[c]) for c in turn.feedback]


def _completed_turn_ids(turn: Turn, tokenizer: Tokenizer) -> list[int]:
    """`<GUESS>` + 5 letter ids + 5 feedback ids + `<SEP>` (12 tokens)."""
    return [
        tokenizer.guess_id,
        *tokenizer.encode_letters(turn.guess),
        *_feedback_ids(turn, tokenizer),
        tokenizer.sep_id,
    ]


def build_prompt(turns: Sequence[Turn], tokenizer: Tokenizer) -> list[int]:
    """The generation prompt: `<BOS> (completed_turn)* <GUESS>` — the model emits 5 letters next."""
    ids = [tokenizer.bos_id]
    for turn in turns:
        ids += _completed_turn_ids(turn, tokenizer)
    ids.append(tokenizer.guess_id)  # cue to generate the next guess
    return ids


def encode_completed_game(turns: Sequence[Turn], tokenizer: Tokenizer) -> list[int]:
    """A finished game: `<BOS> (completed_turn)* <EOS>` — for SFT / GRPO log-prob recomputation."""
    ids = [tokenizer.bos_id]
    for turn in turns:
        ids += _completed_turn_ids(turn, tokenizer)
    ids.append(tokenizer.eos_id)
    return ids


def guess_letter_target_positions(ids: Sequence[int], tokenizer: Tokenizer) -> list[int]:
    """Indices of the generated guess letters — the 5 tokens after each `<GUESS>` that has them.

    These are the loss / log-prob target positions (spec §5.5). A trailing cue `<GUESS>` with no
    letters after it (a generation prompt) contributes nothing.
    """
    positions: list[int] = []
    for i, token in enumerate(ids):
        if token == tokenizer.guess_id and i + GUESS_LEN < len(ids):
            positions.extend(range(i + 1, i + 1 + GUESS_LEN))
    return positions


def decode_word(letter_ids: Sequence[int], tokenizer: Tokenizer) -> str:
    """Decode generated letter ids back to the guessed word (string of letters)."""
    return "".join(tokenizer.id_to_token(int(i)) for i in letter_ids)
