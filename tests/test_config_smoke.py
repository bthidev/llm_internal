import dataclasses
from pathlib import Path

from llm_internal.data.config import load_data_config
from llm_internal.eval.config import load_benchmark_eval_config, load_eval_config
from llm_internal.eval.gates import DEFAULT_GATES, METRIC_DIRECTIONS
from llm_internal.eval.metrics import BenchmarkMetrics
from llm_internal.export.config import load_export_config
from llm_internal.train.config import load_train_config

ROOT = Path(__file__).resolve().parents[1]


def test_real_repository_configs_load_without_gpu():
    load_data_config(ROOT / "configs/data.yaml")
    load_train_config(ROOT / "configs/train.yaml")
    load_eval_config(ROOT / "configs/eval.yaml")
    load_export_config(ROOT / "configs/export.yaml")
    load_benchmark_eval_config(ROOT / "configs/benchmark_eval.yaml")
    load_benchmark_eval_config(ROOT / "configs/benchmark_eval_base.yaml")


def test_every_quality_metric_has_exactly_one_direction():
    metric_fields = {field.name for field in dataclasses.fields(BenchmarkMetrics)} - {"n_cases"}
    assert set(METRIC_DIRECTIONS) == metric_fields


def test_default_gates_use_registered_metric_directions():
    assert len({gate.metric for gate in DEFAULT_GATES}) == len(DEFAULT_GATES)
    for gate in DEFAULT_GATES:
        assert gate.direction == METRIC_DIRECTIONS[gate.metric]
