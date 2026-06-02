"""Preset loading + CLI-override resolution into a RunConfig (spec §10).

A run loads a named *preset*, applies dotted ``key=value`` *overrides*, and yields one
fully-resolved ``RunConfig`` that is logged with the run. Unknown presets/keys error loudly.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Any

from wordle_slm.config import (
    CurriculumConfig,
    DataConfig,
    EvalConfig,
    GRPOConfig,
    ModelConfig,
    RewardConfig,
    RunConfig,
    SFTConfig,
    TokenizerConfig,
)

logger = logging.getLogger(__name__)

# Named presets. Each builds a fresh RunConfig (extend with model-size presets, etc.).
PRESETS: dict[str, Callable[[], RunConfig]] = {
    "default": RunConfig,
}

# Sub-config field name -> dataclass, for round-trip reconstruction.
_SUBCONFIGS: dict[str, type] = {
    "model": ModelConfig,
    "tokenizer": TokenizerConfig,
    "reward": RewardConfig,
    "sft": SFTConfig,
    "grpo": GRPOConfig,
    "curriculum": CurriculumConfig,
    "eval": EvalConfig,
    "data": DataConfig,
}


def load_preset(name: str) -> RunConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    logger.info("loaded preset %r", name)
    return PRESETS[name]()


def _coerce(value: str, current: Any) -> Any:
    # bool must precede int (bool is a subclass of int).
    if isinstance(current, bool):
        low = value.lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
        raise ValueError(f"cannot parse bool from {value!r}")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return value
    raise TypeError(f"unsupported override target type {type(current).__name__} for {value!r}")


def apply_override(cfg: RunConfig, dotted_key: str, raw_value: str) -> None:
    """Set ``cfg.<dotted_key> = coerce(raw_value)``; raise KeyError on an unknown path."""
    parts = dotted_key.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"unknown config path: {dotted_key!r}")
        obj = getattr(obj, part)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown config key: {dotted_key!r}")
    setattr(obj, leaf, _coerce(raw_value, getattr(obj, leaf)))
    logger.info("override %s=%s", dotted_key, raw_value)


def resolve(preset: str = "default", overrides: list[str] | None = None) -> RunConfig:
    """Load ``preset`` then apply ``key=value`` overrides, returning the resolved config."""
    cfg = load_preset(preset)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, _, val = ov.partition("=")
        apply_override(cfg, key.strip(), val.strip())
    logger.info("resolved config (preset=%s, overrides=%d)", preset, len(overrides or []))
    return cfg


def to_dict(cfg: RunConfig) -> dict[str, Any]:
    """Plain-dict view of the config (JSON-serializable; for logging)."""
    return dataclasses.asdict(cfg)


def from_dict(d: dict[str, Any]) -> RunConfig:
    """Reconstruct a RunConfig from ``to_dict`` output (round-trip inverse)."""
    kwargs: dict[str, Any] = {}
    for key, value in d.items():
        kwargs[key] = _SUBCONFIGS[key](**value) if key in _SUBCONFIGS else value
    return RunConfig(**kwargs)
