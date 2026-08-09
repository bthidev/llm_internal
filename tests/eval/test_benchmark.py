# tests/eval/test_benchmark.py
import json
from pathlib import Path

import pytest

from llm_internal.eval.benchmark import CATEGORIES, BenchmarkCase, group_by_category, load_benchmark


def _write_cases(tmp_path: Path, cases: list[dict], name: str = "cases.jsonl") -> Path:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    return path


def _minimal_tool_call_case(id_: str = "c1") -> dict:
    return {
        "id": id_,
        "category": "single_tool_selection",
        "description": "desc",
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"properties": {}, "required": []}}}],
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "expects_tool_call": True,
        "expected_tool_calls": [{"name": "f", "arguments": {}}],
    }


def _minimal_plain_case(id_: str = "c2") -> dict:
    return {
        "id": id_,
        "category": "no_tool_plain_chat",
        "description": "desc",
        "tools": [],
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "expects_tool_call": False,
    }


def test_load_benchmark_reads_real_repo_dataset():
    cases = load_benchmark("data/benchmark/cases.jsonl")

    assert len(cases) >= 20
    covered = {c.category for c in cases}
    assert CATEGORIES <= covered


def test_load_benchmark_parses_minimal_valid_cases(tmp_path: Path):
    path = _write_cases(tmp_path, [_minimal_tool_call_case(), _minimal_plain_case()])

    cases = load_benchmark(path)

    assert len(cases) == 2
    assert isinstance(cases[0], BenchmarkCase)
    assert cases[0].expected_tool_calls == [{"name": "f", "arguments": {}}]
    assert cases[1].expects_tool_call is False


def test_load_benchmark_merges_multiple_files(tmp_path: Path):
    path_a = _write_cases(tmp_path, [_minimal_tool_call_case("a1")], "a.jsonl")
    path_b = _write_cases(tmp_path, [_minimal_plain_case("b1")], "b.jsonl")

    cases = load_benchmark([path_a, path_b])

    assert [c.id for c in cases] == ["a1", "b1"]


def test_load_benchmark_rejects_duplicate_ids(tmp_path: Path):
    path = _write_cases(tmp_path, [_minimal_tool_call_case("dup"), _minimal_plain_case("dup")])

    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark(path)


def test_load_benchmark_rejects_unknown_keys(tmp_path: Path):
    bad = _minimal_plain_case()
    bad["not_a_field"] = 1
    path = _write_cases(tmp_path, [bad])

    with pytest.raises(ValueError, match="unknown"):
        load_benchmark(path)


def test_benchmark_case_rejects_tool_call_expectation_without_ground_truth():
    with pytest.raises(ValueError, match="expected_tool_calls"):
        BenchmarkCase(
            id="x", category="single_tool_selection", description="d", tools=[],
            messages=[{"role": "user", "content": "hi"}],
            expects_tool_call=True,
        )


def test_benchmark_case_rejects_expected_calls_without_expectation():
    with pytest.raises(ValueError, match="expected_tool_calls"):
        BenchmarkCase(
            id="x", category="single_tool_selection", description="d", tools=[],
            messages=[{"role": "user", "content": "hi"}],
            expects_tool_call=False,
            expected_tool_calls=[{"name": "f", "arguments": {}}],
        )


def test_benchmark_case_rejects_empty_messages():
    with pytest.raises(ValueError, match="messages"):
        BenchmarkCase(
            id="x", category="c", description="d", tools=[], messages=[],
            expects_tool_call=False,
        )


def test_group_by_category_partitions_cases(tmp_path: Path):
    path = _write_cases(tmp_path, [_minimal_tool_call_case("a"), _minimal_tool_call_case("b"), _minimal_plain_case("c")])
    cases = load_benchmark(path)

    grouped = group_by_category(cases)

    assert len(grouped["single_tool_selection"]) == 2
    assert len(grouped["no_tool_plain_chat"]) == 1
