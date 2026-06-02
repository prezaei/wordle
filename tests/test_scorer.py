"""Candidate scorer tests (Plan: G)."""

from __future__ import annotations

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.model import Tokenizer
from wordle_slm.model.scorer import CandidateScorer


def _model(seed: int = 0) -> CandidateScorer:
    torch.manual_seed(seed)
    model = CandidateScorer(ModelConfig(), Tokenizer().vocab_size)
    model.eval()  # dropout off -> deterministic scoring
    return model


def _board(tok: Tokenizer) -> torch.Tensor:
    return torch.tensor([tok.encode(["<BOS>", "<EOS>"])])


def test_param_count_within_target_range() -> None:
    n = sum(p.numel() for p in _model().parameters())
    assert 1_000_000 <= n <= 5_000_000


def test_score_returns_one_logit_per_candidate() -> None:
    tok = Tokenizer()
    cands = torch.tensor([tok.encode_letters(w) for w in ("slate", "grace", "brick")])
    logits = _model().score(_board(tok), cands, tok.pad_id)
    assert logits.shape == (3,)


def test_scoring_is_deterministic_for_a_seed() -> None:
    tok = Tokenizer()
    cands = torch.tensor([tok.encode_letters(w) for w in ("slate", "crane")])
    out1 = _model(0).score(_board(tok), cands, tok.pad_id)
    out2 = _model(0).score(_board(tok), cands, tok.pad_id)
    assert torch.allclose(out1, out2)


def test_board_and_candidate_vector_shapes() -> None:
    tok = Tokenizer()
    model = _model()
    d = ModelConfig().d_model
    assert model.board_vector(_board(tok), tok.pad_id).shape == (1, d)
    cands = torch.tensor([tok.encode_letters("slate")])
    assert model.candidate_vectors(cands).shape == (1, d)
