"""Shared, backend-dispatching model loading + text generation. GPU
(`transformers`)/Metal (`mlx_lm`) only -- not exercised by the unit test
suite, same caveat as the rest of the eval package's generation code.
Used by both eval/run_eval.py (Hermes held-out split) and
eval/run_benchmark.py (independent benchmark) so the two share one
prompt-rendering and generation code path."""
from __future__ import annotations

from typing import Any


def render_prompt(messages: list[dict], tokenizer: Any) -> str:
    """Render `messages` through the chat template with a generation
    prompt appended. enable_thinking=False matches the training
    distribution (see train/sft.py build_training_text): the training
    dataset has no <think> content."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def load_cuda_model(model_dir: str, revision: str | None = None) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", revision=revision)
    return model, tokenizer


def generate_cuda(prompts: list[str], model: Any, tokenizer: Any, max_new_tokens: int) -> list[str]:
    predictions = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return predictions


def load_mlx_model(model_dir: str, revision: str | None = None) -> tuple[Any, Any]:
    from mlx_lm import load as mlx_load

    return mlx_load(model_dir, revision=revision)


def generate_mlx(prompts: list[str], model: Any, tokenizer: Any, max_new_tokens: int) -> list[str]:
    from mlx_lm import generate as mlx_generate

    return [mlx_generate(model, tokenizer, prompt=p, max_tokens=max_new_tokens) for p in prompts]


def generate_for_messages(
    messages_list: list[list[dict]], backend: str, model_dir: str, max_new_tokens: int,
    revision: str | None = None,
) -> list[str]:
    """Dispatch to the CUDA (transformers) or MLX generation path based on
    `backend`, rendering each `messages` entry (already the full prompt --
    callers drop any trailing target turn before calling this) through the
    chat template and greedily generating a reply. `revision` pins a
    Hugging Face repo revision (relevant for `model_dir`s that are hub
    repo ids, e.g. the base model in a base-vs-fine-tuned comparison --
    see configs/train.yaml's base_model_revision); ignored for local
    checkpoint/export directories."""
    if backend == "mlx":
        model, tokenizer = load_mlx_model(model_dir, revision)
        prompts = [render_prompt(m, tokenizer) for m in messages_list]
        return generate_mlx(prompts, model, tokenizer, max_new_tokens)

    model, tokenizer = load_cuda_model(model_dir, revision)
    prompts = [render_prompt(m, tokenizer) for m in messages_list]
    return generate_cuda(prompts, model, tokenizer, max_new_tokens)
