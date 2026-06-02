"""Telemetry tests (Plan: D): run.json, scalars, transcripts, context manager."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from wordle_slm.telemetry import RunLog


def test_run_log_writes_meta_scalar_and_transcript(tmp_path: Path) -> None:
    config = {"seed": 0, "grpo": {"group_size": 16}}
    run = RunLog(tmp_path / "run1", config=config, seed=7)
    run.log_scalar("win_rate", 0.42, step=1)
    run.log_transcript({"secret": "crane", "guesses": ["slate", "crane"], "win": True})
    run.close()

    meta = json.loads((tmp_path / "run1" / "run.json").read_text())
    assert meta["seed"] == 7
    assert meta["git_sha"]  # non-empty (a SHA or "unknown")
    assert meta["config"] == config

    lines = (tmp_path / "run1" / "transcripts.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["secret"] == "crane"
    assert record["win"] is True

    assert (tmp_path / "run1" / "tb").is_dir()


def test_run_log_context_manager_closes(tmp_path: Path) -> None:
    with RunLog(tmp_path / "r", config={}, seed=1) as run:
        run.log_scalar("x", 1.0, 0)
    assert (tmp_path / "r" / "run.json").exists()


def test_transcripts_append_across_calls(tmp_path: Path) -> None:
    run = RunLog(tmp_path / "run2", config={}, seed=0)
    run.log_transcript({"i": 1})
    run.log_transcript({"i": 2})
    run.close()
    lines = (tmp_path / "run2" / "transcripts.jsonl").read_text().splitlines()
    assert [json.loads(line)["i"] for line in lines] == [1, 2]


def test_git_sha_is_40_hex_or_unknown(tmp_path: Path) -> None:
    run = RunLog(tmp_path / "g", config={}, seed=0)
    run.close()
    sha = json.loads((tmp_path / "g" / "run.json").read_text())["git_sha"]
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_run_json_is_valid_reloadable_json(tmp_path: Path) -> None:
    config = {"a": 1, "nested": {"b": [1, 2, 3]}}
    run = RunLog(tmp_path / "j", config=config, seed=3)
    run.close()
    meta = json.loads((tmp_path / "j" / "run.json").read_text())
    assert meta["config"] == config
    assert meta["seed"] == 3


def test_run_log_closes_writer_if_init_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = {"v": False}

    class _FakeWriter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            closed["v"] = True

    monkeypatch.setattr("wordle_slm.telemetry.run_log.SummaryWriter", _FakeWriter)

    def _boom(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(OSError):
        RunLog(tmp_path / "x", config={}, seed=0)
    assert closed["v"] is True  # writer was closed on the failed init
