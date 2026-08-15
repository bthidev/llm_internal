"""Sandboxed execution scorer for code-correctness benchmark cases.

Extracts the model's generated code from its reply, then runs it together
with the case's `test_code` in a fresh, isolated Python subprocess (its own
process, its own interpreter, an enforced wall-clock timeout) rather than
`exec()` in this process -- a hanging loop or a crash in generated code
can't stall or take down the benchmark run itself.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT_S = 5.0

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclasses.dataclass(frozen=True)
class CodeExecResult:
    passed: bool
    error: str | None = None  # None on pass; short diagnostic on failure


def extract_code(predicted_text: str) -> str | None:
    """Pull the first fenced code block out of `predicted_text`; falls back
    to the raw text if the model replied with bare code (no fences).
    Returns `None` for an empty/whitespace-only reply.
    """
    match = _CODE_FENCE_RE.search(predicted_text)
    if match:
        return match.group(1)
    stripped = predicted_text.strip()
    return stripped or None


def run_code_case(
    predicted_text: str, entry_point: str, test_code: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> CodeExecResult:
    """Run `predicted_text`'s extracted code plus `test_code` (which must
    exercise the function named `entry_point`, e.g.
    `assert entry_point(1, 2) == 3`) in a fresh subprocess. Passes iff the
    candidate defines `entry_point` and the subprocess exits 0.
    """
    code = extract_code(predicted_text)
    if code is None:
        return CodeExecResult(passed=False, error="no_code_block")
    if entry_point not in code:
        return CodeExecResult(passed=False, error="entry_point_not_defined")

    script = f"{code}\n\n{test_code}\n"
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return CodeExecResult(passed=False, error="timeout")

    if proc.returncode == 0:
        return CodeExecResult(passed=True)
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return CodeExecResult(passed=False, error=tail[-1] if tail else f"exit_code_{proc.returncode}")
