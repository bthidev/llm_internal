"""Pure transforms: raw hermes-function-calling-v1 examples -> Qwen3-ready
chat examples, and a stratified train/val/eval split."""
from __future__ import annotations

import json
import random

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "tool": "tool",
}


def format_example(raw: dict) -> dict:
    """Convert one raw hermes-function-calling-v1 example (`{"id", "conversations"}`,
    each conversation turn `{"from", "value"}`) into `{"id", "messages", "category"}`
    where `messages` is a list of `{"role", "content"}` dicts using Qwen3 chat-template
    role names, and `category` is `"tool_call"` if any assistant turn contains a
    `<tool_call>` block, else `"plain_chat"`.
    """
    messages = []
    for turn in raw["conversations"]:
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            raise ValueError(f"unknown role {turn['from']!r} in example {raw.get('id')!r}")
        messages.append({"role": role, "content": turn["value"]})

    category = "tool_call" if any(
        m["role"] == "assistant" and "<tool_call>" in m["content"] for m in messages
    ) else "plain_chat"

    return {"id": raw.get("id"), "messages": messages, "category": category}


def dedupe_examples(examples: list[dict]) -> list[dict]:
    """Drop examples whose `messages` content exactly duplicates an earlier
    example's, keeping the first occurrence. Source files (e.g.
    `func-calling.json` and `func-calling-singleturn.json`) share verbatim
    examples under different `id`s; letting duplicates survive lets the same
    conversation land in both `train` and `eval`, leaking eval signal.
    """
    seen: set[str] = set()
    result = []
    for ex in examples:
        key = json.dumps(ex["messages"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(ex)
    return result


def stratified_split(
    examples: list[dict],
    train_ratio: float,
    val_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split `examples` (each with a `"category"` key) into train/val/eval lists,
    preserving each category's proportions in every split. Deterministic for a
    given `seed`.
    """
    if abs((train_ratio + val_ratio + eval_ratio) - 1.0) > 1e-9:
        raise ValueError(
            f"train_ratio + val_ratio + eval_ratio must equal 1.0, got "
            f"{train_ratio} + {val_ratio} + {eval_ratio}"
        )

    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for ex in examples:
        by_category.setdefault(ex["category"], []).append(ex)

    train: list[dict] = []
    val: list[dict] = []
    ev: list[dict] = []
    for items in by_category.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        ev.extend(shuffled[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(ev)
    return train, val, ev
