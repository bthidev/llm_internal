from llm_internal.eval.code_exec import extract_code, run_code_case

_ADD_TEST = "assert add(2, 3) == 5"


def test_extract_code_pulls_fenced_python_block():
    text = "Sure, here you go:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nLet me know if that helps."

    code = extract_code(text)

    assert code is not None
    assert "def add" in code
    assert "```" not in code


def test_extract_code_falls_back_to_bare_text_when_no_fence():
    code = extract_code("def add(a, b):\n    return a + b")

    assert code == "def add(a, b):\n    return a + b"


def test_extract_code_returns_none_for_empty_reply():
    assert extract_code("   \n  ") is None


def test_run_code_case_passes_on_correct_implementation():
    predicted = "```python\ndef add(a, b):\n    return a + b\n```"

    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST)

    assert result.passed
    assert result.error is None


def test_run_code_case_fails_on_incorrect_implementation():
    predicted = "```python\ndef add(a, b):\n    return a - b\n```"

    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST)

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

    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST, timeout_s=0.5)

    assert not result.passed
    assert result.error == "timeout"


def test_run_code_case_is_isolated_from_calling_process():
    # A syntax error / runtime crash in generated code must not raise in
    # this process -- it should surface as a normal failed CodeExecResult.
    predicted = "```python\ndef add(a, b)\n    return a + b\n```"

    result = run_code_case(predicted, entry_point="add", test_code=_ADD_TEST)

    assert not result.passed
