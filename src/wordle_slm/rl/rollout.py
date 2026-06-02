"""Shared rollout: play a game by scoring the still-consistent candidates (spec §1.5; Plan: Y).

Each turn: take the still-consistent candidate set (the v3 action space), score it with the model,
and pick one — argmax (greedy, eval) or a sample from the softmax (train). The candidate set is
narrowed incrementally. This is the eval / data-generation rollout (scoring under no-grad); the GRPO
trainer (later wave) reuses the scoring + selection logic, recomputing logits WITH gradients.
"""

from __future__ import annotations

import logging

import torch

from wordle_slm.engine import Game, Status, filter_consistent, secret_in_pool
from wordle_slm.model.scorer import CandidateScorer
from wordle_slm.model.serialization import encode_board, encode_word
from wordle_slm.model.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def play_game(
    model: CandidateScorer,
    tokenizer: Tokenizer,
    secret: str,
    pool: tuple[str, ...],
    *,
    sample: bool = False,
    generator: torch.Generator | None = None,
    device: str = "cpu",
    max_guesses: int = 6,
) -> Game:
    """Play one game; the model selects among still-consistent candidates each turn."""
    candidates: tuple[str, ...] = tuple(pool)  # turn 1: every word is still consistent
    if not secret_in_pool(secret, candidates):
        raise ValueError(f"secret {secret!r} must be in the candidate pool")
    game = Game(secret, max_guesses=max_guesses)
    pad_id = tokenizer.pad_id
    was_training = model.training
    model.eval()
    try:
        while game.status is Status.ONGOING:
            if not candidates:  # unreachable: the secret is in the pool and stays consistent
                raise RuntimeError(f"no consistent candidates for secret {secret!r}")
            board = torch.tensor(encode_board(game.turns, tokenizer), device=device).unsqueeze(0)
            cand_ids = torch.tensor([encode_word(w, tokenizer) for w in candidates], device=device)
            with torch.no_grad():
                logits = model.score(board, cand_ids, pad_id)
            if sample:
                probs = torch.softmax(
                    logits, dim=0
                ).cpu()  # sample on CPU; `generator` must be a CPU generator
                index = int(torch.multinomial(probs, 1, generator=generator).item())
            else:
                index = int(torch.argmax(logits).item())
            turn = game.guess(candidates[index])
            candidates = filter_consistent(candidates, turn)  # narrow for the next turn
    finally:
        model.train(was_training)  # restore the caller's train/eval state
    logger.info("rollout secret=%r -> %s in %d", secret, game.status.value, game.guesses_used)
    return game
