"""Isolated execution scorer for code-correctness benchmark cases.

Generated code is executed in an ephemeral Docker/Podman container by default.
The container has no network, a read-only root filesystem, dropped Linux
capabilities, no-new-privileges, and CPU/memory/PID limits. If no supported
container runtime is available the scorer fails closed instead of executing
untrusted model output on the host.

The default container backend deliberately uses ``--pull=never`` so benchmark
execution never performs an implicit network pull. Pre-pull
``python:3.11-alpine`` (or the image selected by
``LLM_INTERNAL_CODE_EXEC_IMAGE``) before running code-correctness cases. A
missing runtime returns ``sandbox_unavailable``; a missing local image returns
``sandbox_image_unavailable``. Both are infrastructure failures, not evidence
that the generated code itself was incorrect.

Set ``LLM_INTERNAL_CODE_EXEC_BACKEND=unsafe-subprocess`` only for explicitly
trusted/local benchmark data when container isolation is unavailable. Model
output is still arbitrary code, so this backend must not be enabled merely to
work around a missing container runtime on a host that carries secrets or
other valuable state.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_CONTAINER_IMAGE = "python:3.11-alpine"
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclasses.dataclass(frozen=True)
class CodeExecResult:
    passed: bool
    error: str | None = None


def extract_code(predicted_text: str) -> str | None:
    """Extract the first Python fenced block, falling back to bare reply text."""
    match = _CODE_FENCE_RE.search(predicted_text)
    if match:
        return match.group(1)
    stripped = predicted_text.strip()
    return stripped or None


def _container_engine() -> str | None:
    for engine in ("docker", "podman"):
        if shutil.which(engine):
            return engine
    return None


def _container_command(engine: str, script_path: Path) -> list[str]:
    image = os.environ.get("LLM_INTERNAL_CODE_EXEC_IMAGE", DEFAULT_CONTAINER_IMAGE)
    mount = f"{script_path.resolve()}:/workspace/candidate.py:ro"
    return [
        engine,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=128m",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
        "--volume",
        mount,
        "--workdir=/workspace",
        image,
        "python",
        "-I",
        "/workspace/candidate.py",
    ]


def _run_script_in_container(script_path: Path, timeout_s: float) -> CodeExecResult:
    engine = _container_engine()
    if engine is None:
        return CodeExecResult(passed=False, error="sandbox_unavailable")

    try:
        proc = subprocess.run(
            _container_command(engine, script_path),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired:
        return CodeExecResult(passed=False, error="timeout")
    except OSError as exc:
        return CodeExecResult(passed=False, error=f"sandbox_error:{type(exc).__name__}")

    if proc.returncode == 0:
        return CodeExecResult(passed=True)
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    if any("No such image" in line or "image not known" in line.lower() for line in tail):
        return CodeExecResult(passed=False, error="sandbox_image_unavailable")
    return CodeExecResult(passed=False, error=tail[-1] if tail else f"exit_code_{proc.returncode}")


def _run_script_unsafe(script_path: Path, timeout_s: float) -> CodeExecResult:
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={},
            cwd=script_path.parent,
        )
    except subprocess.TimeoutExpired:
        return CodeExecResult(passed=False, error="timeout")

    if proc.returncode == 0:
        return CodeExecResult(passed=True)
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return CodeExecResult(passed=False, error=tail[-1] if tail else f"exit_code_{proc.returncode}")


def run_code_case(
    predicted_text: str,
    entry_point: str,
    test_code: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    backend: str | None = None,
) -> CodeExecResult:
    """Execute generated code and tests in an isolated container by default.

    ``backend`` accepts ``container`` (default) or ``unsafe-subprocess``.
    The container backend requires Docker/Podman plus a locally available
    image (default ``python:3.11-alpine``); it never pulls automatically.
    Missing infrastructure is reported as ``sandbox_unavailable`` or
    ``sandbox_image_unavailable`` and fails closed.

    The unsafe backend is intentionally explicit and should only be used
    when executing code that is genuinely trusted in a constrained,
    disposable environment.
    """
    code = extract_code(predicted_text)
    if code is None:
        return CodeExecResult(passed=False, error="no_code_block")
    if entry_point not in code:
        return CodeExecResult(passed=False, error="entry_point_not_defined")

    selected_backend = backend or os.environ.get("LLM_INTERNAL_CODE_EXEC_BACKEND", "container")
    if selected_backend not in {"container", "unsafe-subprocess"}:
        return CodeExecResult(passed=False, error=f"unknown_backend:{selected_backend}")

    script = f"{code}\n\n{test_code}\n"
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        if selected_backend == "unsafe-subprocess":
            return _run_script_unsafe(script_path, timeout_s)
        return _run_script_in_container(script_path, timeout_s)
