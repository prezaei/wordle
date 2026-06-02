"""Run telemetry: TensorBoard scalars + a structured JSON run log + transcripts (spec §8).

Every result must be diagnosable from logs alone: ``run.json`` captures the resolved config,
seed, and git SHA; ``transcripts.jsonl`` captures per-game records; TensorBoard holds scalars.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import Any

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    """Current commit SHA, or "unknown" if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


class RunLog:
    """Owns a run directory: TensorBoard scalars + ``run.json`` + ``transcripts.jsonl``."""

    def __init__(self, run_dir: str | Path, config: dict[str, Any], seed: int) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        try:
            self._transcripts_path = self.run_dir / "transcripts.jsonl"
            self.meta: dict[str, Any] = {
                "seed": seed,
                "git_sha": _git_sha(),
                "created_at": time.time(),
                "config": config,
            }
            (self.run_dir / "run.json").write_text(
                json.dumps(self.meta, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            # Don't leak the open SummaryWriter (and its event-file handle) on partial init.
            self._writer.close()
            raise
        logger.info(
            "run log initialized at %s (seed=%d, git=%s)",
            self.run_dir,
            seed,
            self.meta["git_sha"][:8],
        )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)
        # DEBUG, not INFO: scalars are logged per-step during RL and would flood the log;
        # TensorBoard is the system of record for them (spec §8).
        logger.debug("scalar %s=%.6g @ step %d", tag, value, step)

    def log_transcript(self, record: dict[str, Any]) -> None:
        """Append one game transcript (or any structured record) as a JSON line."""
        with self._transcripts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        logger.info("logged transcript (%d keys)", len(record))

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
        logger.info("run log closed: %s", self.run_dir)

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
