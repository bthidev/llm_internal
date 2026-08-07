# tests/eval/test_run_eval.py
import pytest

from llm_internal.eval.config import EvalConfig, load_eval_config
from llm_internal.eval.run_eval import evaluate_examples


def test_load_eval_config_reads_real_config_file():
    cfg = load_eval_config("configs/eval.yaml")

    assert isinstance(cfg, EvalConfig)
    assert 0 < cfg.tool_call_accuracy_threshold <= 1
    assert 0 < cfg.plain_chat_pass_rate_threshold <= 1


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
            plain_chat_pass_rate_threshold=0.8,
        ))
