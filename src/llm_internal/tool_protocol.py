"""Shared parsing primitives for the Qwen/Hermes tool-call protocol."""

from __future__ import annotations

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def tool_call_blocks(text: str) -> list[str]:
    """Return raw payloads from every `<tool_call>...</tool_call>` block."""
    return TOOL_CALL_RE.findall(text)


def parse_tool_calls_detailed(text: str) -> tuple[list[dict], int]:
    """Parse tool-call JSON blocks and count malformed JSON payloads."""
    calls: list[dict] = []
    malformed = 0
    for raw in tool_call_blocks(text):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            malformed += 1
    return calls, malformed


def parse_tool_calls(text: str) -> list[dict]:
    """Parse valid tool-call blocks, silently excluding malformed JSON."""
    calls, _ = parse_tool_calls_detailed(text)
    return calls
