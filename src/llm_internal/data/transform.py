"""Pure transforms: raw source-dataset examples -> Qwen3-ready chat examples,
and a stratified train/val/eval split."""

from __future__ import annotations

import json
import random
import re

from llm_internal.tool_protocol import tool_call_blocks

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "tool": "tool",
}

CODE_SYSTEM_PROMPT = (
    "You are an expert software engineer. Write complete, correct, runnable code "
    "in the language requested by the user. Follow the requested language, "
    "interface, and constraints. Explain the approach briefly when requested."
)

_HERMES_TOOL_SYSTEM_PREFIX = (
    "You are a function calling AI model. You are provided with function "
    "signatures within <tools> </tools> XML tags. You may call one or more "
    "functions to assist with the user query. Don't make assumptions about "
    "what values to plug into functions."
)

_GLAIVE_FUNCTIONS_MARKER = "Use them if required -"
_GLAIVE_TURN_RE = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE): ")
_GLAIVE_FUNCTIONCALL_QUOTED_ARGS_RE = re.compile(
    r"""^<functioncall>\s*\{"name":\s*"([^"]+)",\s*"arguments":\s*'(.*)'\}\s*$""",
    re.DOTALL,
)


def format_example(raw: dict) -> dict:
    messages = []
    for turn in raw["conversations"]:
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            raise ValueError(f"unknown role {turn['from']!r} in example {raw.get('id')!r}")
        messages.append({"role": role, "content": turn["value"]})

    category = (
        "tool_call"
        if any(m["role"] == "assistant" and "<tool_call>" in m["content"] for m in messages)
        else "plain_chat"
    )
    return {"id": raw.get("id"), "messages": messages, "category": category}


def format_code_example(raw: dict, source: str, index: int) -> dict:
    extra_input = raw.get("input") or ""
    user_content = raw["instruction"] if not extra_input else f"{raw['instruction']}\n\n{extra_input}"
    messages = [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": raw["output"]},
    ]
    return {"id": f"{source}-{index}", "messages": messages, "category": "plain_chat"}


def _extract_glaive_tools(system_body: str) -> list[dict]:
    marker_idx = system_body.find(_GLAIVE_FUNCTIONS_MARKER)
    if marker_idx == -1:
        return []
    text = system_body[marker_idx + len(_GLAIVE_FUNCTIONS_MARKER) :].strip()

    decoder = json.JSONDecoder()
    tools = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        tools.append(obj)
        pos = end
    return tools


def _parse_glaive_functioncall(content: str) -> dict | None:
    body = content.removeprefix("<functioncall>").strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            return {"name": payload["name"], "arguments": payload.get("arguments", {})}
        return None

    quoted_match = _GLAIVE_FUNCTIONCALL_QUOTED_ARGS_RE.match(content)
    if quoted_match is None:
        return None
    name, args_raw = quoted_match.group(1), quoted_match.group(2)
    try:
        arguments = json.loads(args_raw)
    except json.JSONDecodeError:
        return None
    return {"name": name, "arguments": arguments}


def _parse_glaive_chat(chat_text: str) -> list[dict] | None:
    matches = list(_GLAIVE_TURN_RE.finditer(chat_text))
    if not matches:
        return None

    messages: list[dict] = []
    last_function_name: str | None = None
    for i, match in enumerate(matches):
        role_label = match.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chat_text)
        content = chat_text[match.end() : end].strip()

        if role_label == "USER":
            messages.append({"role": "user", "content": content})
            continue

        if role_label == "ASSISTANT":
            content = content.split("<|endoftext|>")[0].strip()
            if "<functioncall>" not in content:
                messages.append({"role": "assistant", "content": content})
                continue
            tool_call = _parse_glaive_functioncall(content)
            if tool_call is None:
                return None
            last_function_name = tool_call["name"]
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>",
                }
            )
            continue

        if last_function_name is None:
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        tool_response = {"name": last_function_name, "content": parsed}
        messages.append(
            {
                "role": "tool",
                "content": f"<tool_response>\n{json.dumps(tool_response)}\n</tool_response>",
            }
        )

    return messages or None


def format_glaive_example(raw: dict, source: str, index: int) -> dict | None:
    system_body = re.sub(r"^SYSTEM:\s*", "", raw.get("system") or "")
    tool_defs = _extract_glaive_tools(system_body)
    if tool_defs:
        system_content = (
            _HERMES_TOOL_SYSTEM_PREFIX
            + "\n<tools>\n"
            + json.dumps([{"type": "function", "function": tool} for tool in tool_defs])
            + "\n</tools>"
        )
    else:
        system_content = system_body.strip() or "You are a helpful assistant."

    turns = _parse_glaive_chat(raw.get("chat") or "")
    if turns is None:
        return None

    messages = [{"role": "system", "content": system_content}, *turns]
    category = (
        "tool_call"
        if any(m["role"] == "assistant" and "<tool_call>" in m["content"] for m in messages)
        else "plain_chat"
    )
    return {"id": f"{source}-{index}", "messages": messages, "category": category}


def dedupe_examples(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for ex in examples:
        key = json.dumps(ex["messages"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(ex)
    return result


def _tool_calls_are_well_formed(messages: list[dict]) -> bool:
    return all(
        _is_valid_json(raw)
        for message in messages
        if message["role"] == "assistant"
        for raw in tool_call_blocks(message["content"])
    )


def _is_valid_json(raw: str) -> bool:
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def filter_malformed_tool_calls(examples: list[dict]) -> list[dict]:
    """Drop tool-call examples containing malformed JSON protocol blocks."""
    return [ex for ex in examples if ex["category"] != "tool_call" or _tool_calls_are_well_formed(ex["messages"])]


def stratified_split(
    examples: list[dict],
    train_ratio: float,
    val_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if abs((train_ratio + val_ratio + eval_ratio) - 1.0) > 1e-9:
        raise ValueError(
            f"train_ratio + val_ratio + eval_ratio must equal 1.0, got {train_ratio} + {val_ratio} + {eval_ratio}"
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
        val.extend(shuffled[n_train : n_train + n_val])
        ev.extend(shuffled[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(ev)
    return train, val, ev
