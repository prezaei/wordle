"""Config-resolution tests (Plan: K): presets, overrides, errors, round-trip."""

from __future__ import annotations

import json

import pytest

from wordle_slm.config import RunConfig
from wordle_slm.config.resolve import (
    _SUBCONFIGS,
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


def test_non_finite_float_override_is_rejected() -> None:
    # CLAUDE.md invariant: no NaN/inf hidden. A poisoned float must error loudly.
    for bad in ("nan", "inf", "-inf"):
        with pytest.raises(ValueError):
            resolve("default", [f"sft.lr={bad}"])


def test_dict_round_trip_in_memory_is_identity() -> None:
    cfg = resolve("default", ["grpo.group_size=12"])
    assert from_dict(to_dict(cfg)) == cfg


def test_dict_round_trip_through_json_is_identity() -> None:
    # The real persistence path (run.json): tuples must survive as tuples, not lists.
    cfg = resolve("default", ["grpo.group_size=12"])
    reloaded = from_dict(json.loads(json.dumps(to_dict(cfg))))
    assert reloaded == cfg
    assert isinstance(reloaded.curriculum.tiers, tuple)


def test_to_dict_is_json_serializable() -> None:
    blob = json.dumps(to_dict(RunConfig()), sort_keys=True)
    assert "group_size" in blob


def test_overriding_a_tuple_field_errors_clearly() -> None:
    with pytest.raises(TypeError):
        resolve("default", ["curriculum.tiers=200,500"])


def test_override_value_may_contain_equals() -> None:
    cfg = resolve("default", ["data.data_dir=a=b"])
    assert cfg.data.data_dir == "a=b"


def test_subconfigs_map_covers_all_dataclass_fields() -> None:
    import dataclasses

    sub_fields = {
        f.name
        for f in dataclasses.fields(RunConfig)
        if dataclasses.is_dataclass(getattr(RunConfig(), f.name))
    }
    assert set(_SUBCONFIGS) == sub_fields
