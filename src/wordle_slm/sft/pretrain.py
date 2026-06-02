"""Spell warm-up: a plain LM pass over the word list so a from-scratch model learns to spell.

Before SFT/RL, a random generator emits garbage ("hhhhh"). This warm-up teaches the marginal
distribution of real 5-letter words: each word becomes the tiny sequence ``<BOS> <GUESS> w0..w4``
(the model's turn-1 generation context, spec §5.2), and we train the **same masked guess-letter
loss** the SFT uses (`sft.train.sft_loss`) on the 5 letters. After it, generation produces valid
words; SFT then teaches clue-respect and strategy on top. (Migration doc §6.)

Pre-training data = the full valid-guess list — the model should be able to spell *any* legal guess.
Learning to spell a word is not the same as learning it is a secret, so this carries no held-out
leakage (the held-out split governs which words are *secrets*, not which are spellable — spec §4.1).
"""

from __future__ import annotations

import logging
import time
from random import Random

import torch

from wordle_slm.config import SFTConfig
from wordle_slm.data import load_valid_guesses
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator
from wordle_slm.rl.rollout import letter_id_tensor
from wordle_slm.sft.train import pad_and_mask, sft_loss
from wordle_slm.telemetry.run_log import RunLog

logger = logging.getLogger(__name__)


def pretrain_words() -> tuple[str, ...]:
    """The spell-warm-up corpus: every valid 5-letter guess (the words the model may ever emit)."""
    return load_valid_guesses()


def _word_sequence(word: str, tokenizer: Tokenizer) -> list[int]:
    """One pre-training example: ``<BOS> <GUESS> w0 w1 w2 w3 w4`` (the turn-1 generation prompt)."""
    if len(word) != 5 or not word.isalpha():
        raise ValueError(f"pretrain word must be 5 letters, got {word!r}")
    return [tokenizer.bos_id, tokenizer.guess_id, *tokenizer.encode_letters(word)]


def make_pretrain_batch(
    words: list[str], tokenizer: Tokenizer, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded LM batch over words: (input_ids, target_letter_idx, loss_mask), each ``[B, L]``.

    Reuses the SFT `pad_and_mask`, so the loss falls exactly on the 5 letters after `<GUESS>`.
    """
    return pad_and_mask([_word_sequence(w, tokenizer) for w in words], tokenizer, device)


def pretrain_lm(
    model: WordleGenerator,
    words: tuple[str, ...],
    tokenizer: Tokenizer,
    config: SFTConfig,
    *,
    epochs: int = 1,
    batch_size: int = 256,
    device: str = "cpu",
    max_seconds: float | None = None,
    run_log: RunLog | None = None,
    seed: int = 0,
) -> dict:
    """Warm up the generator to spell valid words (masked-letter LM over `words`). AdamW.

    Stops at `epochs` or `max_seconds`. Returns {step, loss, optimizer}. The resulting model (and
    its optimizer) can be checkpointed via `sft.train.save_checkpoint` to seed SFT.
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
    word_list = list(words)
    for epoch in range(epochs):
        rng.shuffle(word_list)
        for i in range(0, len(word_list), batch_size):
            ids, target_idx, mask = make_pretrain_batch(
                word_list[i : i + batch_size], tokenizer, device
            )
            loss = sft_loss(model, ids, target_idx, mask, letter_ids)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite pretrain loss at step {step}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            last_loss = float(loss.detach())
            if run_log is not None:
                run_log.log_scalar("pretrain/loss", last_loss, step)
            if max_seconds is not None and time.perf_counter() - start >= max_seconds:
                logger.info("pretrain hit the %.0fs cap at step %d", max_seconds, step)
                return {"step": step, "loss": last_loss, "optimizer": optimizer}
        logger.info("pretrain epoch %d done: step=%d loss=%.4f", epoch, step, last_loss)
    return {"step": step, "loss": last_loss, "optimizer": optimizer}
