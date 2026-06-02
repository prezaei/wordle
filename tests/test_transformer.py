"""Decoder-only generator tests (Plan: G; spec §5.3)."""

from __future__ import annotations

import string

import pytest
import torch

from wordle_slm.config import ModelConfig
from wordle_slm.model import Tokenizer, WordleGenerator


def _model(seed: int = 0) -> WordleGenerator:
    torch.manual_seed(seed)
    return WordleGenerator(ModelConfig(), Tokenizer().vocab_size)


def _letter_ids(tok: Tokenizer) -> torch.Tensor:
    return torch.tensor(tok.encode_letters(string.ascii_lowercase))


def _prompt(tok: Tokenizer) -> torch.Tensor:
    return torch.tensor(tok.encode(["<BOS>", "<GUESS>"]))  # turn-1 prompt


def test_param_count_within_target_range() -> None:
    n = sum(p.numel() for p in _model().parameters())
    assert 1_000_000 <= n <= 5_000_000


def test_forward_shape() -> None:
    tok = Tokenizer()
    ids = torch.tensor([tok.encode(["<BOS>", "<GUESS>", "c", "r", "a", "n", "e"])])
    logits = _model().forward(ids)
    assert logits.shape == (1, 7, tok.vocab_size)


def test_forward_rejects_overlong_sequence() -> None:
    model = _model()
    too_long = torch.zeros((1, model.context_len + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds context_len"):
        model.forward(too_long)


def test_generate_emits_exactly_five_letters() -> None:
    tok = Tokenizer()
    out = _model().generate(_prompt(tok), _letter_ids(tok), sample=False)
    assert out.shape == (5,)
    letters = set(tok.encode_letters(string.ascii_lowercase))
    assert all(int(i) in letters for i in out)  # never a special token


def test_greedy_generation_is_deterministic() -> None:
    tok = Tokenizer()
    model = _model()
    a = model.generate(_prompt(tok), _letter_ids(tok), sample=False)
    b = model.generate(_prompt(tok), _letter_ids(tok), sample=False)
    assert torch.equal(a, b)


def test_sampling_is_reproducible_with_a_seed() -> None:
    tok = Tokenizer()
    model = _model()
    a = model.generate(
        _prompt(tok), _letter_ids(tok), sample=True, generator=torch.Generator().manual_seed(7)
    )
    b = model.generate(
        _prompt(tok), _letter_ids(tok), sample=True, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(a, b)


def test_generate_restores_training_mode() -> None:
    tok = Tokenizer()
    model = _model()
    model.train()
    model.generate(_prompt(tok), _letter_ids(tok), sample=False)
    assert model.training is True


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
def test_forward_and_generate_run_on_mps() -> None:
    tok = Tokenizer()
    model = _model().to("mps")
    prompt = _prompt(tok).to("mps")
    letter_ids = _letter_ids(tok).to("mps")
    out = model.generate(prompt, letter_ids, sample=False)  # CPU sampling path off (greedy)
    assert out.shape == (5,)
    assert torch.isfinite(model.forward(prompt.unsqueeze(0))).all()
