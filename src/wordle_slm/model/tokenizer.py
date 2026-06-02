"""Character-level tokenizer for board/guess sequences (spec §5.1).

Vocab = 8 special tokens then 26 letters (34 total). ``<PAD>`` is id 0 so padded
positions are zero. The ordering below is the stable contract other components rely on.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SPECIAL_TOKENS: tuple[str, ...] = (
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<SEP>",
    "<GUESS>",
    "<green>",
    "<yellow>",
    "<gray>",
)
LETTERS: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")


class Tokenizer:
    """Maps tokens (letters + special tokens) to/from stable integer ids."""

    def __init__(self) -> None:
        self.vocab: tuple[str, ...] = SPECIAL_TOKENS + LETTERS
        self._tok_to_id: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        logger.info("tokenizer initialized: vocab_size=%d", len(self.vocab))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self._tok_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self._tok_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self._tok_to_id["<EOS>"]

    @property
    def sep_id(self) -> int:
        return self._tok_to_id["<SEP>"]

    @property
    def guess_id(self) -> int:
        return self._tok_to_id["<GUESS>"]

    @property
    def green_id(self) -> int:
        return self._tok_to_id["<green>"]

    @property
    def yellow_id(self) -> int:
        return self._tok_to_id["<yellow>"]

    @property
    def gray_id(self) -> int:
        return self._tok_to_id["<gray>"]

    def token_to_id(self, token: str) -> int:
        try:
            return self._tok_to_id[token]
        except KeyError:
            raise KeyError(f"unknown token: {token!r}") from None

    def id_to_token(self, idx: int) -> str:
        # `type(idx) is int` rejects bool (aliases 0/1) and float (a float in range would
        # still index the tuple) — only a genuine int id is valid.
        if type(idx) is int and 0 <= idx < len(self.vocab):
            return self.vocab[idx]
        raise KeyError(f"unknown token id: {idx!r}")

    def encode(self, tokens: list[str]) -> list[int]:
        """Encode a list of vocab tokens (letters and/or special tokens) to ids."""
        return [self.token_to_id(t) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Decode ids back to their tokens."""
        return [self.id_to_token(i) for i in ids]

    def encode_letters(self, word: str) -> list[int]:
        """Encode a letters-only string (case-insensitive) to letter ids."""
        return [self.token_to_id(c) for c in word.lower()]
