"""Orchestrates dataset preparation: download raw hermes-function-calling-v1
files, format them for Qwen3, split, and write train/val/eval JSONL files."""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from llm_internal.data.config import DataConfig, load_data_config
from llm_internal.data.transform import dedupe_examples, filter_malformed_tool_calls, format_example, stratified_split


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def prepare_dataset(
    raw_examples: list[dict],
    output_dir: Path,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    eval_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """`raw_examples` are already-formatted examples (output of `format_example`,
    each with a `"category"` key). Drops `tool_call` examples with a
    malformed (non-JSON) `<tool_call>` target, then exact-content
    duplicates (keeping the first occurrence), before splitting, so the
    same conversation can't leak across `train`/`val`/`eval`. Writes
    train/val/eval.jsonl into `output_dir`. Returns the example count per
    split.
    """
    well_formed = filter_malformed_tool_calls(raw_examples)
    deduped = dedupe_examples(well_formed)
    train, val, ev = stratified_split(deduped, train_ratio, val_ratio, eval_ratio, seed)
    output_dir = Path(output_dir)
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(ev, output_dir / "eval.jsonl")
    return {"train": len(train), "val": len(val), "eval": len(ev)}


def download_raw_examples(dataset_repo: str, dataset_revision: str, dataset_files: list[str]) -> list[dict]:
    """Download each `dataset_files` entry from `dataset_repo` at the pinned
    `dataset_revision` and concatenate their JSON list contents.
    """
    merged: list[dict] = []
    for filename in dataset_files:
        local_path = hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=filename,
            revision=dataset_revision,
        )
        with open(local_path, encoding="utf-8") as f:
            merged.extend(json.load(f))
    return merged


def main() -> None:
    cfg: DataConfig = load_data_config("configs/data.yaml")
    raw = download_raw_examples(cfg.dataset_repo, cfg.dataset_revision, cfg.dataset_files)
    formatted = [format_example(r) for r in raw]
    before = len(formatted)
    counts = prepare_dataset(
        formatted,
        output_dir=Path(cfg.output_dir),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
    )
    after = counts["train"] + counts["val"] + counts["eval"]
    print(f"dropped {before - after} examples (malformed tool_call target or exact duplicate)")
    print(f"wrote splits: {counts}")


if __name__ == "__main__":
    main()
