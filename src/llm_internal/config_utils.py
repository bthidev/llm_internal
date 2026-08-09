"""Shared YAML -> dataclass config loading, used by data/train/eval configs."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import yaml

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

T = TypeVar("T", bound="DataclassInstance")


def load_yaml_dataclass(path: str | Path, cls: type[T]) -> T:
    """Load a YAML file's top-level mapping and construct `cls` (a dataclass)
    from it. Raises ValueError on unknown keys, TypeError (via the dataclass
    constructor) on missing required fields.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    field_names = {field.name for field in dataclasses.fields(cls)}
    unknown = set(raw) - field_names
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

    return cls(**raw)
