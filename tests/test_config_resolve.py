"""Config-resolution tests (Plan: K): presets, overrides, errors, round-trip."""

from __future__ import annotations

import json

import pytest

from wordle_slm.config import RunConfig
from wordle_slm.config.resolve import (
    from_dict,
    load_preset,
    resolve,
    to_dict,
)


def test_default_preset_equals_runconfig() -> None:
    assert load_preset("default") == RunConfig()


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        load_preset("nope")


def test_override_changes_only_the_targeted_field() -> None:
    cfg = resolve("default", ["grpo.group_size=16"])
    expected = RunConfig()
    expected.grpo.group_size = 16
    assert cfg == expected


def test_override_type_coercion() -> None:
    cfg = resolve(
        "default",
        ["sft.lr=0.001", "grpo.group_size=8", "grpo.filter_zero_variance=false"],
    )
    assert isinstance(cfg.sft.lr, float) and cfg.sft.lr == 0.001
    assert isinstance(cfg.grpo.group_size, int) and cfg.grpo.group_size == 8
    assert cfg.grpo.filter_zero_variance is False


def test_unknown_override_key_raises() -> None:
    with pytest.raises(KeyError):
        resolve("default", ["grpo.nope=1"])
    with pytest.raises(KeyError):
        resolve("default", ["nope.x=1"])


def test_malformed_override_raises() -> None:
    with pytest.raises(ValueError):
        resolve("default", ["grpo.group_size"])  # missing '='


def test_dict_round_trip_is_identity() -> None:
    cfg = resolve("default", ["grpo.group_size=12"])
    assert from_dict(to_dict(cfg)) == cfg


def test_to_dict_is_json_serializable() -> None:
    blob = json.dumps(to_dict(RunConfig()), sort_keys=True)
    assert "group_size" in blob
