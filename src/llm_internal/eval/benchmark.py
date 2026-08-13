"""Extensible, hand-written tool-calling benchmark. Independent from the
Hermes training/eval split (see data/prepare.py): every prompt here is
authored specifically for this benchmark, so a model can't have memorized
it during fine-tuning.

Format: one JSON object per line in a `.jsonl` file (default
`data/benchmark/cases.jsonl`). Each line is a self-contained test case:

    {
      "id": "unique-string-id",
      "category": "single_tool_selection",         # see CATEGORIES below
      "description": "human-readable intent of the case",
      "tools": [ {"type": "function", "function": {...}}, ... ],
      "messages": [ {"role": "system", "content": "..."},
                     {"role": "user", "content": "..."} ],
      "expects_tool_call": true,
      "expected_tool_calls": [ {"name": "...", "arguments": {...}}, ... ],
      "acceptable_tool_names": null
    }

`tools` uses the same `{"type": "function", "function": {"name",
"description", "parameters"}}` shape as the Hermes training data (an
OpenAI/JSON-Schema-style function spec) so cases render through the
production chat template unchanged. `messages` holds every turn up to
(not including) the assistant's reply under test -- typically a `system`
message embedding `tools` as `<tools>[...]</tools>` (matching the Hermes
convention `format_example`/training uses) followed by one or more
`user`/`assistant`/`tool` turns.

`expected_tool_calls` is the ground truth the model's reply is scored
against (see eval/metrics.py); empty when `expects_tool_call` is false.
`acceptable_tool_names`, when set, widens tool-selection scoring for
inherently ambiguous cases: any one of the listed names (with any
arguments) counts as a correct selection, instead of requiring an exact
match against `expected_tool_calls`.

A `"code_correctness"`-category case instead sets `expects_code: true`
(mutually exclusive with `expects_tool_call`), `entry_point` (the function
name the model must define), and `test_code` (assertions exercising that
function, executed against the model's extracted code in a sandboxed
subprocess -- see eval/code_exec.py), e.g.:

    {
      "id": "code_reverse_string",
      "category": "code_correctness",
      "description": "...",
      "tools": [],
      "messages": [ {"role": "system", "content": "..."},
                     {"role": "user", "content": "Write a Python function..."} ],
      "expects_tool_call": false,
      "expects_code": true,
      "entry_point": "reverse_string",
      "test_code": "assert reverse_string('abc') == 'cba'"
    }

To add a case: append one JSON line to the benchmark file (or a new file
merged in `configs/eval.yaml`'s `benchmark_files`). No schema migration,
no code change required.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

# Every category the benchmark is expected to cover (task requirement);
# `load_benchmark` doesn't enforce this set is exhaustive, it's documentation
# plus a sanity check used by tests.
CATEGORIES = frozenset(
    {
        "single_tool_selection",
        "no_tool_plain_chat",
        "ambiguous_tool_request",
        "nonexistent_tool_request",
        "multi_tool_choice",
        "complex_arguments",
        "optional_nullable_arguments",
        "nested_arguments",
        "parallel_tool_calls",
        "must_not_call_tool",
        "code_correctness",
    }
)


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    description: str
    tools: list[dict]
    messages: list[dict]
    expects_tool_call: bool
    expected_tool_calls: list[dict] = dataclasses.field(default_factory=list)
    acceptable_tool_names: list[str] | None = None
    expects_code: bool = False
    entry_point: str = ""
    test_code: str = ""
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("benchmark case is missing a non-empty 'id'")
        if not self.category:
            raise ValueError(f"case {self.id!r} is missing a non-empty 'category'")
        if not self.messages:
            raise ValueError(f"case {self.id!r} has no 'messages'")
        if self.messages[-1]["role"] != "user" and self.messages[-1]["role"] != "tool":
            raise ValueError(
                f"case {self.id!r}: last message must be 'user' or 'tool' "
                f"(the turn the model replies to), got {self.messages[-1]['role']!r}"
            )
        if self.expects_code and self.expects_tool_call:
            raise ValueError(f"case {self.id!r}: expects_code and expects_tool_call are mutually exclusive")
        if self.expects_code and (not self.entry_point or not self.test_code):
            raise ValueError(f"case {self.id!r}: expects_code=true requires entry_point and test_code")
        if not self.expects_code and (self.entry_point or self.test_code):
            raise ValueError(f"case {self.id!r}: entry_point/test_code set but expects_code=false")
        if self.expects_tool_call and not self.expected_tool_calls and not self.acceptable_tool_names:
            raise ValueError(
                f"case {self.id!r}: expects_tool_call=true requires expected_tool_calls and/or acceptable_tool_names"
            )
        if not self.expects_tool_call and self.expected_tool_calls:
            raise ValueError(f"case {self.id!r}: expects_tool_call=false but expected_tool_calls is non-empty")


def _case_from_dict(raw: dict, source: str, line_no: int) -> BenchmarkCase:
    known_fields = {f.name for f in dataclasses.fields(BenchmarkCase)}
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(f"{source}:{line_no}: unknown benchmark case keys {sorted(unknown)}")
    try:
        return BenchmarkCase(**raw)
    except TypeError as e:
        raise ValueError(f"{source}:{line_no}: {e}") from e


def load_benchmark(paths: str | Path | Sequence[str | Path]) -> list[BenchmarkCase]:
    """Load and validate benchmark cases from one or more JSONL files.
    Later files are concatenated after earlier ones. Raises ValueError on
    a duplicate `id` (across all files combined) or a malformed case."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for path in paths:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                raw = json.loads(line)
                case = _case_from_dict(raw, str(path), line_no)
                if case.id in seen_ids:
                    raise ValueError(f"{path}:{line_no}: duplicate benchmark case id {case.id!r}")
                seen_ids.add(case.id)
                cases.append(case)
    return cases


def group_by_category(cases: list[BenchmarkCase]) -> dict[str, list[BenchmarkCase]]:
    grouped: dict[str, list[BenchmarkCase]] = {}
    for case in cases:
        grouped.setdefault(case.category, []).append(case)
    return grouped
