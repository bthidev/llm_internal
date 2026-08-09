# tests/eval/test_run_benchmark.py
from llm_internal.eval.benchmark import BenchmarkCase, load_benchmark
from llm_internal.eval.config import BenchmarkEvalConfig
from llm_internal.eval.run_benchmark import format_report, report_to_json, run_benchmark

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _cfg(**overrides: float) -> BenchmarkEvalConfig:
    base = BenchmarkEvalConfig(
        model_dir="unused", benchmark_files=["data/benchmark/cases.jsonl"], max_new_tokens=64,
        min_plain_chat_chars=5, backend="cuda", seed=0,
    )
    if overrides:
        import dataclasses
        return dataclasses.replace(base, **overrides)
    return base


def test_run_benchmark_scores_and_gates_against_real_benchmark_file():
    cases = load_benchmark("data/benchmark/cases.jsonl")
    predictions = ["I'm not sure how to help with that." for _ in cases]

    report, gate_results = run_benchmark(cases, predictions, _cfg())

    assert report.overall.n_cases == len(cases)
    assert len(gate_results) > 0


def test_run_benchmark_applies_gate_overrides_from_config():
    case = BenchmarkCase(
        id="c1", category="single_tool_selection", description="d", tools=[_TOOL],
        messages=[{"role": "user", "content": "weather?"}],
        expects_tool_call=True, expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
    )
    cfg = BenchmarkEvalConfig(
        model_dir="unused", benchmark_files=["f.jsonl"], max_new_tokens=64,
        min_plain_chat_chars=5, backend="cuda", seed=0,
        gate_overrides={"tool_selection_accuracy": 0.0},
    )
    predictions = ["no tool call at all"]  # would fail the default 0.85 gate

    _, gate_results = run_benchmark([case], predictions, cfg)

    gate = next(g for g in gate_results if g.metric == "tool_selection_accuracy")
    assert gate.threshold == 0.0
    assert gate.passed is True


def test_format_report_includes_overall_category_and_gate_sections():
    cases = load_benchmark("data/benchmark/cases.jsonl")
    predictions = ["Sure!" for _ in cases]
    report, gate_results = run_benchmark(cases, predictions, _cfg())

    text = format_report(report, gate_results)

    assert "=== overall ===" in text
    assert "=== by category ===" in text
    assert "=== gates ===" in text


def test_report_to_json_is_json_serializable_and_matches_gate_verdict():
    import json

    cases = load_benchmark("data/benchmark/cases.jsonl")
    predictions = ["Sure!" for _ in cases]
    report, gate_results = run_benchmark(cases, predictions, _cfg())

    payload = report_to_json(report, gate_results)
    serialized = json.dumps(payload)  # must not raise

    reloaded = json.loads(serialized)
    assert reloaded["overall"]["n_cases"] == len(cases)
    assert isinstance(reloaded["passed"], bool)
