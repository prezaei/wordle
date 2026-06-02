"""Spell-warm-up (pre-training) data + warm-up tests — correctness of the generated data."""

from __future__ import annotations

import pytest
import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_valid_guesses
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.rl.rollout import letter_id_tensor
from wordle_slm.sft.pretrain import (
    _word_sequence,
    make_pretrain_batch,
    pretrain_lm,
    pretrain_words,
)
from wordle_slm.sft.train import sft_loss


def _small() -> tuple[WordleGenerator, Tokenizer]:
    torch.manual_seed(0)
    tok = Tokenizer()
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)
    return WordleGenerator(cfg, tok.vocab_size), tok


# --- the generated data is correct ------------------------------------------------------------


def test_corpus_is_the_full_valid_list_all_valid_and_unique() -> None:
    words = pretrain_words()
    assert words == load_valid_guesses()  # the model must be able to spell any legal guess
    assert all(len(w) == 5 and w.isalpha() and is_valid(w) for w in words)
    assert len(set(words)) == len(words)  # no duplicates


def test_word_sequence_matches_the_generation_context() -> None:
    tok = Tokenizer()
    # exactly the turn-1 generation prompt + the 5 letters: <BOS> <GUESS> w0..w4
    assert tok.decode(_word_sequence("crane", tok)) == ["<BOS>", "<GUESS>", *list("crane")]


def test_word_sequence_rejects_non_five_letter_words() -> None:
    tok = Tokenizer()
    for bad in ("cran", "craness", "cr4ne"):
        with pytest.raises(ValueError, match="5 letters"):
            _word_sequence(bad, tok)


def test_batch_targets_reconstruct_every_word_and_mask_is_exact() -> None:
    tok = Tokenizer()
    words = list(load_valid_guesses()[:300])
    ids, target_idx, mask = make_pretrain_batch(words, tok)
    assert ids.shape == (len(words), 7)  # <BOS> <GUESS> + 5 letters (no padding needed, all len 5)
    assert int(mask.sum()) == 5 * len(words)  # exactly the 5 letters per word are scored
    letter_lo = tok.token_to_id("a")
    for i, word in enumerate(words):
        positions = (mask[i] > 0).nonzero().flatten().tolist()
        recovered = "".join(tok.id_to_token(int(target_idx[i, p]) + letter_lo) for p in positions)
        assert recovered == word  # the masked targets are exactly the word's letters


def test_batch_scores_exactly_the_five_letter_predictions() -> None:
    # Sequence: <BOS>(0) <GUESS>(1) w0(2) w1(3) w2(4) w3(5) w4(6). The loss mask is on the PREDICT
    # positions q-1 (the logit at q-1 predicts the letter at q): {1,2,3,4,5}. <BOS>(0) predicts
    # <GUESS> and is never scored; position 6 (last letter) predicts nothing trained.
    tok = Tokenizer()
    _, _, mask = make_pretrain_batch(["crane"], tok)
    assert mask[0, 0] == 0  # <BOS> is not scored
    assert mask[0, 1:6].sum() == 5 and mask[0, 6] == 0  # exactly the 5 letter predictions
    assert int(mask.sum()) == 5


# --- the data trains the model (it learns to spell) -------------------------------------------


def test_warmup_reduces_the_spelling_loss() -> None:
    model, tok = _small()
    words = tuple(load_valid_guesses()[:256])
    ids, target_idx, mask = make_pretrain_batch(list(words), tok)
    letter_ids = letter_id_tensor(tok)
    model.eval()
    before = float(sft_loss(model, ids, target_idx, mask, letter_ids))
    pretrain_lm(model, words, tok, SFTConfig(), epochs=20, batch_size=64)
    model.eval()
    after = float(sft_loss(model, ids, target_idx, mask, letter_ids))
    assert after < before  # the warm-up demonstrably learns from the data


def test_warmup_respects_the_time_cap() -> None:
    model, tok = _small()
    out = pretrain_lm(
        model,
        tuple(load_valid_guesses()[:512]),
        tok,
        SFTConfig(),
        epochs=100,
        batch_size=64,
        max_seconds=0.0,
    )
    assert out["step"] == 1  # the 0s cap stops after the first batch
