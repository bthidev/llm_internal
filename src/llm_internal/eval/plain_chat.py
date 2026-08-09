"""Deterministic plain-chat response validation. No LLM-as-judge: every
check is a cheap, reproducible heuristic over the predicted text itself.
Replaces the old "min_chars only" check (llm_internal.eval.scoring's
`score_plain_chat_example`, kept for the existing Hermes-split eval) with
checks that catch the failure modes a length threshold misses: the model
calling a tool when it shouldn't, echoing chat-template/protocol tokens,
emitting an empty/degenerate reply, or leaking a role prefix."""
from __future__ import annotations

import dataclasses
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]*\|>")
_PROTOCOL_TAG_RE = re.compile(r"</?(?:tools|tool_response|s|im_start|im_end)\b", re.IGNORECASE)
_ROLE_PREFIX_RE = re.compile(r"^\s*(system|user|assistant|tool)\s*:", re.IGNORECASE)

# A degenerate reply repeats one word/token far more than natural language
# would; this is a decoding failure mode (e.g. a runaway loop), not a
# stylistic choice.
_DEGENERATE_MIN_WORDS = 20
_DEGENERATE_MAX_UNIQUE_RATIO = 0.15


@dataclasses.dataclass(frozen=True)
class PlainChatCheck:
    passed: bool
    reasons: tuple[str, ...] = ()


def _is_degenerate_repetition(text: str) -> bool:
    words = text.split()
    if len(words) < _DEGENERATE_MIN_WORDS:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio <= _DEGENERATE_MAX_UNIQUE_RATIO


def check_plain_chat_response(text: str, min_chars: int = 5) -> PlainChatCheck:
    """Deterministically validate a reply that should NOT contain a tool
    call. Returns every applicable failure reason (not just the first), so
    callers can report exactly what went wrong."""
    stripped = text.strip()
    reasons: list[str] = []

    if not stripped:
        reasons.append("empty_response")
    elif len(stripped) < min_chars:
        reasons.append("too_short")

    if _TOOL_CALL_RE.search(text):
        reasons.append("unexpected_tool_call")
    if _SPECIAL_TOKEN_RE.search(text):
        reasons.append("special_token_leakage")
    if _PROTOCOL_TAG_RE.search(text):
        reasons.append("protocol_tag_leakage")
    if stripped and _ROLE_PREFIX_RE.match(stripped):
        reasons.append("role_prefix_leakage")
    if stripped and _is_degenerate_repetition(stripped):
        reasons.append("degenerate_repetition")

    return PlainChatCheck(passed=not reasons, reasons=tuple(reasons))
