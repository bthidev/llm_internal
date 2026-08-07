"""Pure scoring logic for the held-out eval gate. No model/GPU involved --
takes already-generated prediction text."""
from __future__ import annotations

import dataclasses
import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract every <tool_call>{json}</tool_call> block from `text`. Blocks
    that don't parse as JSON are skipped (a malformed model output is a
    scoring failure, not a crash)."""
    calls = []
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return calls


def _last_expected_tool_call(expected_messages: list[dict]) -> dict | None:
    for message in reversed(expected_messages):
        if message["role"] == "assistant" and "<tool_call>" in message["content"]:
            calls = parse_tool_calls(message["content"])
            return calls[0] if calls else None
    return None


def score_tool_call_example(expected_messages: list[dict], predicted_text: str) -> dict:
    """Compare the first tool call in `predicted_text` against the first tool
    call in the last tool-calling assistant turn of `expected_messages`."""
    expected = _last_expected_tool_call(expected_messages)
    predicted_calls = parse_tool_calls(predicted_text)
    predicted = predicted_calls[0] if predicted_calls else None

    if expected is None or predicted is None:
        return {"correct_name": False, "correct_args": False, "structural_match": False}

    correct_name = predicted.get("name") == expected.get("name")
    correct_args = predicted.get("arguments") == expected.get("arguments")
    return {
        "correct_name": correct_name,
        "correct_args": correct_args,
        "structural_match": correct_name and correct_args,
    }


def score_plain_chat_example(predicted_text: str, min_chars: int) -> bool:
    return len(predicted_text.strip()) >= min_chars


@dataclasses.dataclass
class Report:
    tool_call_accuracy: float
    plain_chat_pass_rate: float
    passed: bool


def aggregate_results(
    per_example_results: list[dict],
    tool_call_accuracy_threshold: float,
    plain_chat_pass_rate_threshold: float,
) -> Report:
    tool_call_results = [r for r in per_example_results if r["category"] == "tool_call"]
    plain_chat_results = [r for r in per_example_results if r["category"] == "plain_chat"]

    tool_call_accuracy = (
        sum(1 for r in tool_call_results if r["structural_match"]) / len(tool_call_results)
        if tool_call_results else 1.0
    )
    plain_chat_pass_rate = (
        sum(1 for r in plain_chat_results if r["plain_chat_pass"]) / len(plain_chat_results)
        if plain_chat_results else 1.0
    )

    passed = (
        tool_call_accuracy >= tool_call_accuracy_threshold
        and plain_chat_pass_rate >= plain_chat_pass_rate_threshold
    )
    return Report(
        tool_call_accuracy=tool_call_accuracy,
        plain_chat_pass_rate=plain_chat_pass_rate,
        passed=passed,
    )
