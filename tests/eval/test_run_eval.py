# tests/eval/test_run_eval.py
import pytest

from llm_internal.eval.config import EvalConfig, load_eval_config
from llm_internal.eval.run_eval import _load_and_generate, evaluate_examples


def test_load_eval_config_reads_real_config_file():
    cfg = load_eval_config("configs/eval.yaml")

    assert isinstance(cfg, EvalConfig)
    assert 0 < cfg.tool_call_accuracy_threshold <= 1
    assert 0 < cfg.plain_chat_pass_rate_threshold <= 1
    assert cfg.backend == "cuda"


def test_eval_config_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        EvalConfig(
            model_dir="m", eval_file="e", max_new_tokens=1, min_plain_chat_chars=1,
            tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8,
            backend="tpu",
        )


def test_evaluate_examples_scores_mixed_batch_and_gates():
    examples = [
        {
            "category": "tool_call",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
            ],
        },
        {
            "category": "plain_chat",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ],
        },
    ]
    predictions = [
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        "Hello! How can I help you today?",
    ]
    cfg = EvalConfig(
        model_dir="unused",
        eval_file="unused",
        max_new_tokens=256,
        min_plain_chat_chars=5,
        tool_call_accuracy_threshold=0.8,
        plain_chat_pass_rate_threshold=0.8,
        backend="cuda",
    )

    report = evaluate_examples(examples, predictions, cfg)

    assert report.tool_call_accuracy == 1.0
    assert report.plain_chat_pass_rate == 1.0
    assert report.passed is True


def test_evaluate_examples_requires_matching_lengths():
    with pytest.raises(ValueError):
        evaluate_examples([{"category": "plain_chat", "messages": []}], [], EvalConfig(
            model_dir="unused", eval_file="unused", max_new_tokens=1,
            min_plain_chat_chars=1, tool_call_accuracy_threshold=0.8,
            plain_chat_pass_rate_threshold=0.8, backend="cuda",
        ))


def test_load_and_generate_dispatches_to_mlx(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "llm_internal.eval.run_eval.generate_predictions_mlx",
        lambda examples, model_dir, cfg: calls.append(("mlx", model_dir)) or ["pred"],
    )
    cfg = EvalConfig(
        model_dir="m", eval_file="unused", max_new_tokens=1, min_plain_chat_chars=1,
        tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8, backend="mlx",
    )

    result = _load_and_generate([{"messages": []}], cfg)

    assert result == ["pred"]
    assert calls == [("mlx", "m")]


def test_load_and_generate_dispatches_to_cuda(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: "tok"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", classmethod(lambda cls, *a, **k: "model"))
    calls = []
    monkeypatch.setattr(
        "llm_internal.eval.run_eval.generate_predictions",
        lambda examples, model, tokenizer, cfg: calls.append((model, tokenizer)) or ["pred"],
    )
    cfg = EvalConfig(
        model_dir="m", eval_file="unused", max_new_tokens=1, min_plain_chat_chars=1,
        tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8, backend="cuda",
    )

    result = _load_and_generate([{"messages": []}], cfg)

    assert result == ["pred"]
    assert calls == [("model", "tok")]
