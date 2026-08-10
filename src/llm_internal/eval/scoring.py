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


def parse_tool_calls_detailed(text: str) -> tuple[list[dict], int]:
    """Like `parse_tool_calls`, but also reports how many <tool_call> blocks
    failed to parse as JSON (a structural-validity signal `parse_tool_calls`
    alone discards)."""
    calls = []
    malformed = 0
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            malformed += 1
    return calls, malformed


def summarize_tool_call_failures(
    examples: list[dict], predictions: list[str], max_samples: int = 5,
) -> str:
    """Human-readable diagnostic report for a failed/low tool_call_accuracy
    run: aggregate signal (how many predictions contain no <tool_call> block
    at all, vs. malformed JSON, vs. well-formed-but-wrong) plus a handful of
    expected-vs-predicted samples. Printed to stdout at eval time (not part
    of the pass/fail gate itself) so a failing run's Kaggle log carries
    enough signal to diagnose without re-downloading the checkpoint.
    """
    lines = ["--- tool_call diagnostic samples ---"]
    no_call = 0
    malformed_any = 0
    wrong_name = 0
    wrong_args = 0
    shown = 0
    for example, predicted_text in zip(examples, predictions, strict=True):
        if example["category"] != "tool_call":
            continue
        expected = _last_expected_tool_call(example["messages"])
        predicted_calls, malformed = parse_tool_calls_detailed(predicted_text)
        predicted = predicted_calls[0] if predicted_calls else None

        if predicted is None:
            if malformed:
                malformed_any += 1
            else:
                no_call += 1
        elif expected is not None and predicted.get("name") != expected.get("name"):
            wrong_name += 1
        elif expected is not None and predicted.get("arguments") != expected.get("arguments"):
            wrong_args += 1

        if shown < max_samples:
            lines.append(
                f"  expected={expected!r}\n"
                f"  predicted_calls={predicted_calls!r} malformed={malformed}\n"
                f"  raw_predicted_text={predicted_text[:300]!r}"
            )
            shown += 1

    lines.append(
        f"  totals: no_tool_call_block={no_call} malformed_json={malformed_any} "
        f"wrong_name={wrong_name} wrong_args={wrong_args}"
    )
    return "\n".join(lines)


def _last_tool_call_turn_index(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message["role"] == "assistant" and "<tool_call>" in message["content"]:
            return i
    return None


def _last_expected_tool_call(expected_messages: list[dict]) -> dict | None:
    idx = _last_tool_call_turn_index(expected_messages)
    if idx is None:
        return None
    calls = parse_tool_calls(expected_messages[idx]["content"])
    return calls[0] if calls else None


def prompt_messages_for_example(example: dict) -> list[dict]:
    """Messages to render (with a generation prompt appended) so the model's
    continuation is comparable to what's actually scored.

    For `plain_chat`, that's everything but the final assistant turn, as
    before. For `tool_call`, the raw hermes-function-calling-v1 data often
    continues *past* the scored tool call -- e.g. system/user/.../
    assistant(<tool_call>)/tool(response)/assistant(final NL summary).
    Dropping only the last message would then prompt the model to continue
    *after* the tool call already happened (correctly producing a plain NL
    summary), while scoring compares that summary against the tool call
    itself -- a guaranteed mismatch. Truncate instead to end right before
    the same tool-calling turn `_last_expected_tool_call` scores against,
    so the model is actually asked to produce it.
    """
    messages = example["messages"]
    if example["category"] == "tool_call":
        idx = _last_tool_call_turn_index(messages)
        if idx is not None:
            return messages[:idx]
    return messages[:-1]


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
