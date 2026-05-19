"""Tests for the ``{"output": ...}`` tool_result content wrap.

Non-Anthropic DIAL deployments route through Bedrock Converse. DIAL's
OpenAI→Converse translator auto-parses any bare-string ``tool`` role
content as JSON; when the parse yields a non-OBJECT (array, scalar,
null), Bedrock 400s on ``toolResult.content[0].json is invalid``. The
adapter wraps the payload in ``{"output": <text>}`` for non-Anthropic
upstreams so the auto-parse always lands on a valid JSON object.

These tests cover:

* The wrap fires for non-Anthropic upstreams (string, list-of-text-parts,
  empty content).
* The wrap is bypassed for Anthropic upstreams.
* The OpenAI ``tool`` message shape stays valid (bare-string content,
  not a content-parts list) — preventing the DIAL ``502 No route`` we
  hit when we previously tried sending structured content.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ADAPTER_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anthropic_dial_adapter_app", ADAPTER_DIR / "app.py")
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
sys.modules["anthropic_dial_adapter_app"] = app
spec.loader.exec_module(app)


def _msg_with_tool_result(model: str, tool_result_content):
    """Build a 3-message body that triggers the tool_result translation path."""
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_x", "name": "get_thing", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_x",
                 "content": tool_result_content},
            ]},
        ],
    }


def _last_tool_message(openai_body):
    """Return the most recent OpenAI ``tool`` role message in the translated body."""
    for m in reversed(openai_body["messages"]):
        if m.get("role") == "tool":
            return m
    raise AssertionError("no tool message in translated body")


# ---------------------------------------------------------------------------
# Wrap fires for non-Anthropic upstreams
# ---------------------------------------------------------------------------


def test_wrap_fires_for_qwen_with_string_content() -> None:
    body = _msg_with_tool_result("qwen.qwen3-coder-480b-a35b-v1:0", "plain text result")
    out, _ = app.anthropic_to_openai(body)
    tool_msg = _last_tool_message(out)
    assert isinstance(tool_msg["content"], str)
    parsed = json.loads(tool_msg["content"])
    assert isinstance(parsed, dict)
    assert parsed["output"] == "plain text result"


def test_wrap_fires_for_moonshotai_with_json_shape_text() -> None:
    """The exact failure mode we shipped this for: gh pr list --json returns a
    JSON array string, which DIAL would auto-parse into
    ``toolResult.content[0].json`` and Bedrock would 400 on (array, not object)."""
    body = _msg_with_tool_result(
        "moonshotai.kimi-k2.5",
        '[{"number": 42, "title": "first"}]',
    )
    out, _ = app.anthropic_to_openai(body)
    tool_msg = _last_tool_message(out)
    # Result is wrapped, and the inner JSON-shape text is preserved verbatim
    parsed = json.loads(tool_msg["content"])
    assert isinstance(parsed, dict)
    assert parsed["output"] == '[{"number": 42, "title": "first"}]'


def test_wrap_fires_for_minimax_with_list_content() -> None:
    body = _msg_with_tool_result(
        "minimax.minimax-m2.5",
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
    )
    out, _ = app.anthropic_to_openai(body)
    tool_msg = _last_tool_message(out)
    parsed = json.loads(tool_msg["content"])
    assert isinstance(parsed, dict)
    # List of text parts is flattened with newlines before wrapping
    assert "a" in parsed["output"] and "b" in parsed["output"]


def test_wrap_handles_empty_content_for_non_anthropic_upstream() -> None:
    body = _msg_with_tool_result("qwen.qwen3-coder-480b-a35b-v1:0", "")
    out, _ = app.anthropic_to_openai(body)
    tool_msg = _last_tool_message(out)
    parsed = json.loads(tool_msg["content"])
    assert isinstance(parsed, dict)
    assert "output" in parsed


# ---------------------------------------------------------------------------
# Wrap bypasses for Anthropic upstreams
# ---------------------------------------------------------------------------


def test_wrap_bypassed_for_anthropic_upstream_with_string_content() -> None:
    body = _msg_with_tool_result("anthropic.claude-sonnet-4-6", "plain text result")
    out, _ = app.anthropic_to_openai(body)
    tool_msg = _last_tool_message(out)
    # Anthropic upstream is unaffected — passes the original string through
    assert tool_msg["content"] == "plain text result"


def test_wrap_bypassed_for_global_anthropic_namespace() -> None:
    """``global.anthropic.*`` cross-region inference profiles count as Anthropic."""
    body = _msg_with_tool_result("global.anthropic.claude-opus-4-7", "plain text result")
    out, _ = app.anthropic_to_openai(body)
    tool_msg = _last_tool_message(out)
    assert tool_msg["content"] == "plain text result"


# ---------------------------------------------------------------------------
# Regression: the legacy failure path
# ---------------------------------------------------------------------------


def test_qwen_tool_result_is_never_structured_list() -> None:
    """We previously tried sending an OpenAI content-parts list for the
    ``tool`` role to fix the JSON-coercion 400. That triggered DIAL's
    chat-completions contract validation and produced ``502 No route``.
    Regression guard: the ``tool`` message content must always be a bare
    string for non-Anthropic upstreams."""
    body = _msg_with_tool_result(
        "qwen.qwen3-coder-480b-a35b-v1:0",
        [{"type": "text", "text": "hello"}],
    )
    out, _ = app.anthropic_to_openai(body)
    tool_msg = _last_tool_message(out)
    assert isinstance(tool_msg["content"], str), \
        "DIAL chat-completions requires bare-string content on the tool role"
