# tests/data/test_prepare.py
import json
from pathlib import Path

from llm_internal.data.config import DataConfig, load_data_config
from llm_internal.data.prepare import prepare_dataset, write_jsonl
from llm_internal.data.transform import format_example


def _raw_examples(n_tool_call: int, n_plain: int):
    examples = []
    for i in range(n_tool_call):
        examples.append({
            "id": f"tc-{i}",
            "conversations": [
                {"from": "system", "value": "sys<tools>[]</tools>"},
                {"from": "human", "value": f"query {i}"},
                {"from": "gpt", "value": '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'},
            ],
        })
    for i in range(n_plain):
        examples.append({
            "id": f"pc-{i}",
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": f"hi {i}"},
                {"from": "gpt", "value": "hello"},
            ],
        })
    return examples


def test_write_jsonl_round_trips(tmp_path: Path):
    examples = [{"a": 1}, {"a": 2}]
    path = tmp_path / "out.jsonl"

    write_jsonl(examples, path)

    lines = path.read_text().strip().splitlines()
    assert [json.loads(line) for line in lines] == examples


def test_prepare_dataset_writes_three_split_files(tmp_path: Path):
    raw = _raw_examples(n_tool_call=40, n_plain=40)
    formatted = [format_example(r) for r in raw]

    counts = prepare_dataset(formatted, output_dir=tmp_path, train_ratio=0.8, val_ratio=0.1, eval_ratio=0.1, seed=1)

    assert counts == {"train": 64, "val": 8, "eval": 8}
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()

    train_lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == 64
    first = json.loads(train_lines[0])
    assert "messages" in first and "category" in first


def test_load_data_config_reads_real_config_file():
    cfg = load_data_config("configs/data.yaml")

    assert isinstance(cfg, DataConfig)
    assert cfg.dataset_repo == "NousResearch/hermes-function-calling-v1"
    assert cfg.dataset_revision == "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
    assert cfg.dataset_files == [
        "func-calling.json",
        "func-calling-singleturn.json",
        "glaive-function-calling-5k.json",
        "json-mode-agentic.json",
        "json-mode-singleturn.json",
    ]
    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.eval_ratio - 1.0) < 1e-9
