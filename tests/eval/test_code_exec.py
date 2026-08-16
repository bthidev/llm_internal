from pathlib import Path

from llm_internal.eval.code_exec import _container_command, extract_code, run_code_case

_ADD_TEST = "assert add(2, 3) == 5"
_UNSAFE = "unsafe-subprocess"


def test_extract_code_pulls_fenced_python_block():
    text = "Sure, here you go:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nLet me know if that helps."
    code = extract_code(text)
    assert code is not None
    assert "def add" in code
    assert "```" not in code


def test_extract_code_falls_back_to_bare_text_when_no_fence():
    assert extract_code("def add(a, b):\n    return a + b") == "def add(a, b):\n    return a + b"


def test_extract_code_returns_none_for_empty_reply():
    assert extract_code("   \n  ") is None


def test_container_command_hardens_runtime(tmp_path: Path):
    script = tmp_path / "candidate.py"
    script.write_text("print('ok')", encoding="utf-8")
    cmd = _container_command("docker", script)
    joined = " ".join(cmd)
    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--pids-limit=64" in cmd
    assert "--memory=128m" in cmd
    assert "--cpus=1" in cmd
    assert "--user=65534:65534" in cmd
    assert ":/workspace/candidate.py:ro" in joined


def test_run_code_case_passes_on_correct_implementation():
    predicted = "```python\ndef add(a, b):\n    return a + b\n```"
    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST, backend=_UNSAFE)
    assert result.passed
    assert result.error is None


def test_run_code_case_fails_on_incorrect_implementation():
    predicted = "```python\ndef add(a, b):\n    return a - b\n```"
    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST, backend=_UNSAFE)
    assert not result.passed
    assert result.error is not None


def test_run_code_case_fails_when_no_code_block_present():
    result = run_code_case("Sorry, I can't help with that.", entry_point="add", test_code=_ADD_TEST)
    assert not result.passed
    assert result.error == "entry_point_not_defined"


def test_run_code_case_fails_when_reply_is_empty():
    result = run_code_case("   ", entry_point="add", test_code=_ADD_TEST)
    assert not result.passed
    assert result.error == "no_code_block"


def test_run_code_case_fails_when_entry_point_not_defined():
    predicted = "```python\ndef subtract(a, b):\n    return a - b\n```"
    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST)
    assert not result.passed
    assert result.error == "entry_point_not_defined"


def test_run_code_case_times_out_on_infinite_loop():
    predicted = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST, timeout_s=0.5, backend=_UNSAFE)
    assert not result.passed
    assert result.error == "timeout"


def test_run_code_case_is_isolated_from_calling_process():
    predicted = "```python\ndef add(a, b)\n    return a + b\n```"
    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST, backend=_UNSAFE)
    assert not result.passed


def test_run_code_case_rejects_unknown_backend():
    predicted = "```python\ndef add(a, b):\n    return a + b\n```"
    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST, backend="bogus")
    assert not result.passed
    assert result.error == "unknown_backend:bogus"
