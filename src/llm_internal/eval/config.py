"""Config for the held-out eval gate and the independent benchmark."""
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
    backend: str

    def __post_init__(self) -> None:
        if self.backend not in ("cuda", "mlx"):
            raise ValueError(f"backend must be 'cuda' or 'mlx', got {self.backend!r}")


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_yaml_dataclass(path, EvalConfig)


@dataclasses.dataclass
class BenchmarkEvalConfig:
    model_dir: str
    benchmark_files: list[str]
    max_new_tokens: int
    min_plain_chat_chars: int
    backend: str
    seed: int
    model_revision: str | None = None
    gate_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    report_path: str = "benchmark_report.json"
    comparison_report_path: str = "comparison_report.json"

    def __post_init__(self) -> None:
        if self.backend not in ("cuda", "mlx"):
            raise ValueError(f"backend must be 'cuda' or 'mlx', got {self.backend!r}")
        if not self.benchmark_files:
            raise ValueError("benchmark_files must list at least one benchmark JSONL file")
        if not self.report_path:
            raise ValueError("report_path must not be empty")
        if not self.comparison_report_path:
            raise ValueError("comparison_report_path must not be empty")


def load_benchmark_eval_config(path: str | Path) -> BenchmarkEvalConfig:
    return load_yaml_dataclass(path, BenchmarkEvalConfig)
