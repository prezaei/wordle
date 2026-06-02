"""From-scratch decoder-only char transformer — the generation policy (spec §5.3; Plan: G).

A small autoregressive language model over the 34-token char vocab: token + learned positional
embeddings, pre-norm causal transformer blocks, **weight-tied** output head. It conditions on the
board prompt (§5.2) and `generate`s the next guess's 5 letters from its own weights — no candidate
list. The engine then validates the word (free generation + validate). Default ≈ 3.2M params
(1–5M target). MPS; trained from random init (no pretrained weights).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from wordle_slm.config import ModelConfig

logger = logging.getLogger(__name__)


class WordleGenerator(nn.Module):
    """Decoder-only transformer LM that generates Wordle guesses letter-by-letter."""

    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        d = config.d_model
        self.d_model = d
        self.context_len = config.context_len
        self.token_embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Embedding(config.context_len, d)
        block = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        # enable_nested_tensor=False: the nested-tensor fast path is unimplemented on MPS (and we
        # use a causal mask, not a padding mask, so it never applies anyway).
        self.blocks = nn.TransformerEncoder(
            block, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d)
        # Weight-tied output head: logits = hidden @ token_embed.weightᵀ (no separate matrix).
        logger.info(
            "WordleGenerator: %d params (d_model=%d, layers=%d, heads=%d)",
            sum(p.numel() for p in self.parameters()),
            d,
            config.n_layers,
            config.n_heads,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Causal LM forward. ``ids`` ``[B, L]`` -> logits ``[B, L, vocab]``."""
        length = ids.shape[1]
        if length > self.context_len:
            raise ValueError(f"sequence length {length} exceeds context_len {self.context_len}")
        positions = torch.arange(length, device=ids.device).unsqueeze(0)
        x = self.token_embed(ids) + self.pos_embed(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(length, device=ids.device)
        hidden = self.final_norm(self.blocks(x, mask=causal, is_causal=True))
        return hidden @ self.token_embed.weight.t()  # weight-tied head -> [B, L, vocab]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        letter_ids: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate exactly 5 letter token ids after the prompt (length-masked to the 26 letters).

        ``prompt_ids`` is a 1-D id tensor ending in `<GUESS>`. Logits are restricted to
        ``letter_ids`` (the policy's action space is the 26 letters — §5.3 / migration C2), so a
        special token can never be emitted. ``generator``, if given, must be a CPU generator
        (`multinomial` runs on CPU — the MPS-sampling lesson).
        """
        was_training = self.training
        self.eval()
        try:
            seq = prompt_ids.clone()
            chosen: list[int] = []
            for _ in range(5):
                logits = self.forward(seq.unsqueeze(0))[0, -1]  # [vocab]
                letter_logits = logits[letter_ids]  # [26] — the action space
                if sample:
                    probs = torch.softmax(letter_logits, dim=0).cpu()
                    index = int(torch.multinomial(probs, 1, generator=generator).item())
                else:
                    index = int(torch.argmax(letter_logits).item())
                token = int(letter_ids[index].item())
                chosen.append(token)
                seq = torch.cat([seq, torch.tensor([token], device=seq.device)])
            return torch.tensor(chosen, device=prompt_ids.device)
        finally:
            self.train(was_training)
