"""Shared rollout: play one game by GENERATING each guess (spec §6.1; Plan: Y).

Each turn: build the board prompt (§5.2, ending in `<GUESS>`), have the model generate 5 letters,
form the word, and submit it to the engine — which validates (a non-word consumes the turn, exactly
as for a human). **No candidate list, no consistency filter.** This is the eval / data-generation
rollout; the GRPO trainer reuses it for sampling and recomputes the generation log-probs WITH
gradients separately (rl.tracer).
"""

from __future__ import annotations

import logging
import string

import torch

from wordle_slm.data import is_valid
from wordle_slm.engine import Game, Status
from wordle_slm.model.serialization import build_prompt, decode_word
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator

logger = logging.getLogger(__name__)


def letter_id_tensor(tokenizer: Tokenizer, device: str = "cpu") -> torch.Tensor:
    """The 26 letter token ids (the generator's action space), as a tensor on ``device``."""
    return torch.tensor(tokenizer.encode_letters(string.ascii_lowercase), device=device)


def play_game(
    model: WordleGenerator,
    tokenizer: Tokenizer,
    secret: str,
    *,
    sample: bool = False,
    generator: torch.Generator | None = None,
    device: str = "cpu",
    max_guesses: int = 6,
) -> Game:
    """Play one game; the model generates each guess letter-by-letter."""
    if not is_valid(secret):
        raise ValueError(f"secret {secret!r} must be a valid word")
    letter_ids = letter_id_tensor(tokenizer, device)
    game = Game(secret, max_guesses=max_guesses)
    while game.status is Status.ONGOING:
        prompt = torch.tensor(build_prompt(game.turns, tokenizer), device=device)
        chosen = model.generate(prompt, letter_ids, sample=sample, generator=generator)
        game.guess(decode_word(chosen.tolist(), tokenizer))  # engine validates the generated word
    logger.info("rollout secret=%r -> %s in %d", secret, game.status.value, game.guesses_used)
    return game
