"""Head-start (SFT) training: imitate the teacher transcripts (spec §5.5; Plan: N).

Next-token cross-entropy **masked to the guess-letter positions** (the 5 letters after each
`<GUESS>`) — the board + feedback are context, not scored (spec §5.5 / §6.2). Teaches a from-scratch
generator to spell valid words and respect clues before RL. AdamW; saves a reloadable checkpoint so
the GRPO trainer (Plan: Q) can init the policy and freeze `π_ref`.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from random import Random

import torch

from wordle_slm.config import GRPOConfig, SFTConfig
from wordle_slm.engine import Game
from wordle_slm.model.serialization import encode_completed_game, guess_letter_target_positions
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator
from wordle_slm.rl.rollout import letter_id_tensor
from wordle_slm.telemetry.run_log import RunLog

logger = logging.getLogger(__name__)


def make_batch(
    games: list[Game], tokenizer: Tokenizer, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded batch: (input_ids [B,L], target_letter_idx [B,L], loss_mask [B,L]).

    `target_letter_idx[i, p]` (where the mask is 1) is the realized letter's index in the 26-letter
    action space for the token predicted at position `p` (= q-1 for each guess-letter position q).
    """
    seqs = [encode_completed_game(g.turns, tokenizer) for g in games]
    return pad_and_mask(seqs, tokenizer, device)


def pad_and_mask(
    seqs: list[list[int]], tokenizer: Tokenizer, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad token-id sequences and mask the loss to the guess-letter positions (the §5.5 mask).

    Shared by SFT (game transcripts) and the spell warm-up (single-word sequences). Returns
    (input_ids, target_letter_idx, loss_mask), each ``[B, L]``.
    """
    letter_lo = tokenizer.token_to_id("a")
    max_len = max(len(s) for s in seqs)
    batch = len(seqs)
    input_ids = torch.full((batch, max_len), tokenizer.pad_id, dtype=torch.long)
    target_idx = torch.zeros((batch, max_len), dtype=torch.long)
    loss_mask = torch.zeros((batch, max_len))
    for i, seq in enumerate(seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq)
        for q in guess_letter_target_positions(seq, tokenizer):
            target_idx[i, q - 1] = seq[q] - letter_lo
            loss_mask[i, q - 1] = 1.0
    return input_ids.to(device), target_idx.to(device), loss_mask.to(device)


def sft_loss(
    model: WordleGenerator,
    input_ids: torch.Tensor,
    target_idx: torch.Tensor,
    loss_mask: torch.Tensor,
    letter_ids: torch.Tensor,
) -> torch.Tensor:
    """Guess-letter-masked next-token NLL over the 26-letter action space (spec §5.5)."""
    logits = model.forward(input_ids)  # [B, L, vocab]
    logp = torch.log_softmax(logits[:, :, letter_ids], dim=-1)  # [B, L, 26]
    nll = -logp.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)  # [B, L]
    return (nll * loss_mask).sum() / loss_mask.sum()


def _batches(n: int, batch_size: int, rng: Random) -> list[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def train_sft(
    model: WordleGenerator,
    games: list[Game],
    tokenizer: Tokenizer,
    config: SFTConfig,
    *,
    epochs: int = 1,
    batch_size: int = 64,
    device: str = "cpu",
    max_seconds: float | None = None,
    run_log: RunLog | None = None,
    seed: int = 0,
) -> dict:
    """Train the generator to imitate the teacher games (masked guess-letter loss). AdamW.

    Stops at `epochs` or `max_seconds` (the ~15-min cap — spec §5.5). Returns {step, loss}.
    """
    letter_ids = letter_id_tensor(tokenizer, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    rng = Random(seed)
    model.train()
    start = time.perf_counter()
    step = 0
    last_loss = float("nan")
    for epoch in range(epochs):
        for indices in _batches(len(games), batch_size, rng):
            ids, target_idx, mask = make_batch([games[i] for i in indices], tokenizer, device)
            loss = sft_loss(model, ids, target_idx, mask, letter_ids)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite SFT loss at step {step}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            last_loss = float(loss.detach())
            if run_log is not None:
                run_log.log_scalar("sft/loss", last_loss, step)
            if max_seconds is not None and time.perf_counter() - start >= max_seconds:
                logger.info(
                    "SFT hit the %.0fs cap at step %d (loss=%.4f)", max_seconds, step, last_loss
                )
                return {"step": step, "loss": last_loss, "optimizer": optimizer}
        logger.info("SFT epoch %d done: step=%d loss=%.4f", epoch, step, last_loss)
    return {"step": step, "loss": last_loss, "optimizer": optimizer}


def save_checkpoint(
    path: str | Path,
    model: WordleGenerator,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: SFTConfig | GRPOConfig,
) -> None:
    """Save a reloadable checkpoint {model, optim, step, rng, config} (spec §5.5 / Plan N).

    `config` is any stage config dataclass (stored via ``dataclasses.asdict`` as run metadata).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(config),
            "torch_rng": torch.get_rng_state(),
        },
        path,
    )
    logger.info("saved SFT checkpoint to %s (step=%d)", path, step)


def load_checkpoint(
    path: str | Path,
    model: WordleGenerator,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Restore model (and optionally optimizer + RNG) from a checkpoint; returns the metadata."""
    # weights_only=True: the checkpoint holds only tensors + primitive containers (no custom
    # classes), so loading stays safe against pickle code-execution.
    ckpt = torch.load(Path(path), weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim"])
    torch.set_rng_state(ckpt["torch_rng"])
    logger.info("loaded SFT checkpoint from %s (step=%d)", path, ckpt["step"])
    return ckpt
