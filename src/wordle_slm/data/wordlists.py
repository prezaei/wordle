"""Word lists, seeded train/held-out split, and validity (spec §4.1).

Committed data files (offline):
- ``words/answers.txt``       — 2,315 secret-answer words (the pool secrets are drawn from)
- ``words/valid_guesses.txt`` — 14,855 NYT-current accepted guesses (a superset of the answers)

The split is seeded and the held-out set is **immutable**: shrinking-to-budget (spec §4.5) may
shrink the *train* set, never held-out. Held-out words remain valid *guesses* — they are only
excluded as training *secrets*.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import resources
from random import Random

logger = logging.getLogger(__name__)

_ANSWERS_FILE = "answers.txt"
_VALID_FILE = "valid_guesses.txt"


def _load_file(name: str) -> tuple[str, ...]:
    text = resources.files("wordle_slm.data").joinpath("words", name).read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


@lru_cache(maxsize=1)
def load_answers() -> tuple[str, ...]:
    """The answer pool (secrets). Cached; returns a stable, sorted tuple."""
    words = _load_file(_ANSWERS_FILE)
    logger.info("loaded %d answer words", len(words))
    return words


@lru_cache(maxsize=1)
def load_valid_guesses() -> tuple[str, ...]:
    """All accepted guesses (validity superset). Cached; stable, sorted tuple."""
    words = _load_file(_VALID_FILE)
    logger.info("loaded %d valid-guess words", len(words))
    return words


@lru_cache(maxsize=1)
def _valid_set() -> frozenset[str]:
    return frozenset(load_valid_guesses())


def is_valid(word: str) -> bool:
    """True iff ``word`` is an accepted guess (case-insensitive)."""
    return word.lower() in _valid_set()


def split(seed: int = 0, train_frac: float = 0.80) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically split the answer pool into ``(train, held_out)``.

    Seeded shuffle then slice; identical across runs for a given seed. ``train`` and
    ``held_out`` are disjoint and together cover the whole answer pool.
    """
    answers = list(load_answers())
    Random(seed).shuffle(answers)
    n_train = round(len(answers) * train_frac)
    train = tuple(answers[:n_train])
    held_out = tuple(answers[n_train:])
    logger.info(
        "split: %d train / %d held-out (seed=%d, frac=%.2f)",
        len(train),
        len(held_out),
        seed,
        train_frac,
    )
    return train, held_out


def train_probe(
    seed: int = 0, train_frac: float = 0.80, size: int | None = None
) -> tuple[str, ...]:
    """A fixed subset of TRAIN used to measure the generalization gap (spec §6.7).

    Default size matches the held-out set, for an apples-to-apples gap. Deterministic
    for a given seed; always ⊆ train and disjoint from held-out. Returned sorted.
    """
    train, held_out = split(seed=seed, train_frac=train_frac)
    if size is None:
        size = len(held_out)
    size = min(size, len(train))
    # A seed-derived RNG independent of the split's own shuffle, so the probe is stable.
    shuffled = list(train)
    Random(seed + 1).shuffle(shuffled)
    probe = tuple(sorted(shuffled[:size]))
    logger.info(
        "train probe: %d words (seed=%d, matched-to-heldout=%s)",
        len(probe),
        seed,
        size == len(held_out),
    )
    return probe
