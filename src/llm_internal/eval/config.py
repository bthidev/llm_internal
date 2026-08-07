"""Config for the held-out eval gate."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class EvalConfig:
    model_dir: str
    eval_file: str
    max_new_tokens: int
    min_plain_chat_chars: int
    tool_call_accuracy_threshold: float
    plain_chat_pass_rate_threshold: float


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_yaml_dataclass(path, EvalConfig)
