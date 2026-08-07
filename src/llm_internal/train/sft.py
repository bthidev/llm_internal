"""QLoRA SFT training stage. Chat rendering/dataset assembly are pure and
tested locally; `run_training` requires a CUDA GPU and is exercised on a
rented GPU (see scripts/run_on_runpod.sh)."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from llm_internal.train.config import TrainConfig, load_train_config


def build_training_text(messages: list[dict], tokenizer) -> str:
    """Render a full conversation (list of {"role","content"}) through the
    model's chat template for training. Uses add_generation_prompt=False
    since the whole conversation, including the final assistant reply, is
    present -- no enable_thinking flag is needed here (it only affects the
    generation-prompt-only branch of Qwen3's template).
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_split(path: str | Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_hf_dataset(examples: list[dict], tokenizer) -> Dataset:
    texts = [build_training_text(ex["messages"], tokenizer) for ex in examples]
    return Dataset.from_dict({"text": texts})


def run_training(cfg: TrainConfig) -> None:
    """GPU-only: loads Qwen3-1.7B in 4-bit via Unsloth, attaches a LoRA
    adapter, and runs TRL's SFTTrainer. Resumes automatically from the
    latest checkpoint in `cfg.output_dir` if one exists.
    """
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        random_state=cfg.seed,
    )

    train_examples = load_split(Path(cfg.data_dir) / "train.jsonl")
    val_examples = load_split(Path(cfg.data_dir) / "val.jsonl")
    train_ds = build_hf_dataset(train_examples, tokenizer)
    val_ds = build_hf_dataset(val_examples, tokenizer)

    output_dir = Path(cfg.output_dir)
    resume = output_dir.exists() and any(output_dir.glob("checkpoint-*"))

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        save_steps=cfg.checkpoint_every_steps,
        eval_strategy="steps",
        eval_steps=cfg.checkpoint_every_steps,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))


def main() -> None:
    cfg = load_train_config("configs/train.yaml")
    run_training(cfg)


if __name__ == "__main__":
    main()
