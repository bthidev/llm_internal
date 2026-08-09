# tests/eval/test_config.py
import pytest

from llm_internal.eval.config import BenchmarkEvalConfig, load_benchmark_eval_config


def test_load_benchmark_eval_config_reads_real_config_file():
    cfg = load_benchmark_eval_config("configs/benchmark_eval.yaml")

    assert isinstance(cfg, BenchmarkEvalConfig)
    assert cfg.benchmark_files == ["data/benchmark/cases.jsonl"]
    assert cfg.backend == "cuda"
    assert cfg.gate_overrides == {}


def test_benchmark_eval_config_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        BenchmarkEvalConfig(
            model_dir="m", benchmark_files=["f.jsonl"], max_new_tokens=64,
            min_plain_chat_chars=5, backend="tpu", seed=0,
        )


def test_benchmark_eval_config_rejects_empty_benchmark_files():
    with pytest.raises(ValueError, match="benchmark_files"):
        BenchmarkEvalConfig(
            model_dir="m", benchmark_files=[], max_new_tokens=64,
            min_plain_chat_chars=5, backend="cuda", seed=0,
        )


def test_benchmark_eval_config_gate_overrides_default_to_empty_dict():
    cfg = BenchmarkEvalConfig(
        model_dir="m", benchmark_files=["f.jsonl"], max_new_tokens=64,
        min_plain_chat_chars=5, backend="cuda", seed=0,
    )

    assert cfg.gate_overrides == {}
