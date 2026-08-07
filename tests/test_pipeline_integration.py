# tests/test_pipeline_integration.py
from pathlib import Path

from llm_internal.data.transform import format_example
from llm_internal.data.prepare import prepare_dataset
from llm_internal.eval.config import EvalConfig
from llm_internal.eval.run_eval import evaluate_examples
from llm_internal.export.to_gguf import render_modelfile, write_modelfile
from llm_internal.train.sft import load_split


def _fixture_raw_examples():
    return [
        {
            "id": "tc-1",
            "conversations": [
                {"from": "system", "value": "You are a function calling AI model.\n<tools>[]</tools>"},
                {"from": "human", "value": "What's the weather in Paris?"},
                {"from": "gpt", "value": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
            ],
        },
        {
            "id": "pc-1",
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": "Hi there"},
                {"from": "gpt", "value": "Hello! How can I help you today?"},
            ],
        },
    ] * 10  # 20 raw examples, enough for a non-degenerate 80/10/10 split


def test_full_offline_pipeline_slice_runs_end_to_end(tmp_path: Path):
    # 1. format
    formatted = [format_example(r) for r in _fixture_raw_examples()]

    # 2. split + write
    data_dir = tmp_path / "processed"
    counts = prepare_dataset(formatted, output_dir=data_dir, train_ratio=0.8, val_ratio=0.1, eval_ratio=0.1, seed=0)
    assert sum(counts.values()) == 20

    # 3. load the eval split back
    eval_examples = load_split(data_dir / "eval.jsonl")
    assert eval_examples

    # 4. pretend a model generated perfect predictions, score them
    cfg = EvalConfig(
        model_dir="unused", eval_file="unused", max_new_tokens=64,
        min_plain_chat_chars=3, tool_call_accuracy_threshold=0.8,
        plain_chat_pass_rate_threshold=0.8,
    )
    predictions = []
    for ex in eval_examples:
        last = ex["messages"][-1]["content"]
        predictions.append(last)  # perfect prediction
    report = evaluate_examples(eval_examples, predictions, cfg)
    assert report.passed is True

    # 5. render + write the Modelfile a passing run would export
    modelfile_path = tmp_path / "export" / "Modelfile"
    write_modelfile(modelfile_path, render_modelfile("homemade-llm-q4_k_m.gguf"))
    assert modelfile_path.exists()
    assert "FROM ./homemade-llm-q4_k_m.gguf" in modelfile_path.read_text()
