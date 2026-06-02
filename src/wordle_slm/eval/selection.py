"""Generalization gap + checkpoint selection (spec §6.7; Plan: S).

- **Generalization gap** = (fixed train-probe win rate) − (held-out win rate), on the **same fixed
  probe set** each time so it's comparable across a run. Signed: negative early is fine;
  a persistent large positive gap = memorization.
- **Best-checkpoint-by-held-out** lives in the trainer (`rl.grpo.train_grpo`, `best_checkpoint=`):
  it keeps the checkpoint with the highest held-out win rate, so a late collapse can't lose it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator
from wordle_slm.rl.grpo import eval_win_rate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GapReport:
    probe_win_rate: float  # on the fixed TRAIN probe set
    heldout_win_rate: float
    gap: float  # probe − heldout; large + = memorization, ~0 = generalizing

    @property
    def memorizing(self) -> bool:
        return self.gap > 0.0


def generalization_gap(
    model: WordleGenerator,
    tokenizer: Tokenizer,
    *,
    probe_secrets: Sequence[str],
    heldout_secrets: Sequence[str],
    device: str = "cpu",
) -> GapReport:
    """Signed generalization gap = probe win rate − held-out win rate (greedy, spec §6.7)."""
    probe = eval_win_rate(model, tokenizer, tuple(probe_secrets), device=device)
    heldout = eval_win_rate(model, tokenizer, tuple(heldout_secrets), device=device)
    report = GapReport(probe, heldout, probe - heldout)
    logger.info("generalization gap: probe=%.3f heldout=%.3f gap=%+.3f", probe, heldout, report.gap)
    return report
