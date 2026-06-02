"""From-scratch transformer candidate scorer for v3 (spec §1.5; Plan: G).

A small Transformer encoder turns the board token sequence into a board vector; each candidate
word (5 letters) is encoded into a candidate vector; candidates are scored by dot-product
compatibility. A softmax over a turn's consistent candidates is the policy. Trained from random
init (no pretrained weights). Default config ≈ 3.2M params (within the 1–5M target).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from wordle_slm.config import ModelConfig

logger = logging.getLogger(__name__)


class CandidateScorer(nn.Module):
    """Scores candidate words against the current board state."""

    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        d = config.d_model
        self.d_model = d
        self.context_len = config.context_len
        self.token_embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Embedding(config.context_len, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.board_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.cand_proj = nn.Linear(d, d)
        logger.info(
            "CandidateScorer: %d params (d_model=%d, layers=%d, heads=%d)",
            sum(p.numel() for p in self.parameters()),
            d,
            config.n_layers,
            config.n_heads,
        )

    def board_vector(self, board_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Encode board sequences ``[B, L]`` to mean-pooled board vectors ``[B, d]``."""
        length = board_ids.shape[1]
        positions = torch.arange(length, device=board_ids.device).unsqueeze(0)
        x = self.token_embed(board_ids) + self.pos_embed(positions)
        pad_mask = board_ids == pad_id  # True at padded positions
        hidden = self.board_encoder(x, src_key_padding_mask=pad_mask)  # [B, L, d]
        keep = (~pad_mask).unsqueeze(-1).to(hidden.dtype)
        return (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

    def candidate_vectors(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Encode candidate words ``[N, 5]`` to candidate vectors ``[N, d]``."""
        width = candidate_ids.shape[1]
        positions = torch.arange(width, device=candidate_ids.device).unsqueeze(0)
        emb = self.token_embed(candidate_ids) + self.pos_embed(positions)
        return self.cand_proj(emb.mean(dim=1))

    def score(
        self, board_ids: torch.Tensor, candidate_ids: torch.Tensor, pad_id: int
    ) -> torch.Tensor:
        """Logits over the candidate set for one board (one logit per candidate)."""
        if board_ids.dim() == 1:
            board_ids = board_ids.unsqueeze(0)
        board = self.board_vector(board_ids, pad_id).squeeze(0)  # [d]
        cands = self.candidate_vectors(candidate_ids)  # [N, d]
        return (cands @ board) / (self.d_model**0.5)  # [N]
