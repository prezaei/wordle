"""Per-epoch progress logging for the live dashboard (driver helper, not committed).

The contract that makes the dashboard show *the result of every inference in every epoch*:
a training driver calls ``append_epoch(...)`` once per epoch with the metrics it just measured
plus the actual eval ``Game``s it played; ``scripts/live_viz.py`` reads the freshest
``runs/*_progress.jsonl`` and renders the latest epoch's boards + the per-epoch curve — no model
reload, no re-play, so it can't contend with training on the GPU.

Drop-in for any driver:

    from viz_progress import append_epoch          # scripts/ is on sys.path when run as a script
    PROG = "runs/<run-name>_progress.jsonl"
    ...
    games = [play(model, s) for s in VIZ]          # a fixed held-out subset, played greedily
    append_epoch(PROG, epoch, {"win": w, "valid": v, "avg": a}, games)
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from wordle_slm.engine import Color
from wordle_slm.engine.scoring import score

_NAME: dict[Color, str] = {Color.GREEN: "green", Color.YELLOW: "yellow", Color.GRAY: "gray"}


def game_record(game: Any, **extra: Any) -> dict[str, Any]:
    """A finished ``Game`` -> board JSON (per-turn colors; ghost colors for invalid words).

    An invalid guess has no real feedback (``turn.feedback is None``); we still surface what it
    *would* have scored as ``ghost`` so the dashboard can fade it in (it was a wasted turn).
    ``extra`` carries per-game annotations the dashboard renders as a badge — for RL rollouts pass
    ``reward=`` (the grade) and ``adv=`` (group-relative advantage); non-finite values become None.
    """
    turns: list[dict[str, Any]] = []
    for t in game.turns:
        if t.feedback is None:
            turns.append(
                {"guess": t.guess, "fb": None, "ghost": [_NAME[c] for c in score(t.guess, game.secret)]}
            )
        else:
            turns.append({"guess": t.guess, "fb": [_NAME[c] for c in t.feedback]})
    record = {"secret": game.secret, "status": game.status.value, "used": game.guesses_used, "turns": turns}
    record.update({key: _finite(val) for key, val in extra.items()})
    return record


def _finite(value: float) -> float | None:
    """JSON-safe scalar: NaN/inf -> None (a JSON ``NaN`` would break the browser's ``JSON.parse``)."""
    v = float(value)
    return v if math.isfinite(v) else None


def append_epoch(
    path: str,
    epoch: int,
    metrics: Mapping[str, float],
    games: Sequence[Any],
    *,
    sample: int = 12,
    kind: str = "sft",
    grades: Sequence[Mapping[str, float]] | None = None,
) -> None:
    """Append one record (``epoch``/update index + metrics + a ``sample`` of boards) as a JSON line.

    ``kind`` labels the phase (``"sft"`` greedy eval, ``"rl"`` rollouts). ``grades`` is an optional
    per-game annotation list aligned with ``games`` (e.g. ``[{"reward": r, "adv": a}, ...]``) — for
    RL this is the grade shown on each board.
    """
    record: dict[str, Any] = {"epoch": int(epoch), "kind": kind}
    record.update({key: _finite(val) for key, val in metrics.items()})
    record["games"] = [
        game_record(g, **(grades[i] if grades is not None and i < len(grades) else {}))
        for i, g in enumerate(games[:sample])
    ]
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_progress(path: str) -> list[dict[str, Any]]:
    """Read every epoch record from a progress file, tolerating a half-flushed trailing line."""
    records: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # the driver may be mid-write on the last line
    except OSError:
        return []
    return records
