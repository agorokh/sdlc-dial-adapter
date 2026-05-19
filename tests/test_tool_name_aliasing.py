"""Tests for Bedrock 64-char ``toolSpec.name`` aliasing.

DIAL routes every non-Anthropic deployment through AWS Bedrock's Converse
API, which caps ``toolSpec.name`` at 64 chars. Anthropic-on-DIAL does not.
The adapter aliases long MCP tool names like
``mcp__plugin_deploy-on-aws_awsknowledge__aws___get_regional_availability``
(74 chars) to a deterministic <=64-char form on the request, and
reverse-maps ``tool_use[].name`` on the response so the client never sees
the alias.

These tests cover:

* The alias builder (deterministic, <=64 chars, MCP prefix preserved when
  possible, no rewrite for already-short names).
* The end-to-end ``_alias_long_tool_names`` mutation across ``tools[]``,
  ``tool_choice``, and assistant ``messages[].tool_calls[].function.name``.
* Anthropic-upstream passthrough (no aliasing).
* The response-side reverse map in ``openai_to_anthropic_response`` for
  ``tool_use`` blocks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anthropic_dial_adapter_app", ADAPTER_DIR / "app.py")
assert spec is not None
assert spec.loader is not None
app = importlib.util.module_from_spec(spec)
sys.modules["anthropic_dial_adapter_app"] = app
spec.loader.exec_module(app)


LONG_NAME = "mcp__plugin_deploy-on-aws_awsknowledge__aws___get_regional_availability"
LONG_NAME_2 = "mcp__plugin_deploy-on-aws_awsknowledge__aws___search_documentation"


# ---------------------------------------------------------------------------
# _build_tool_name_alias
# ---------------------------------------------------------------------------


def test_alias_truncates_to_max_len() -> None:
    alias = app._build_tool_name_alias(LONG_NAME)
    assert len(alias) == app._BEDROCK_TOOL_NAME_MAX


def test_alias_preserves_mcp_prefix() -> None:
    alias = app._build_tool_name_alias(LONG_NAME)
    assert alias.startswith("mcp__")


def test_alias_is_deterministic() -> None:
    assert app._build_tool_name_alias(LONG_NAME) == app._build_tool_name_alias(LONG_NAME)


def test_alias_distinct_for_different_names() -> None:
    a = app._build_tool_name_alias(LONG_NAME)
    b = app._build_tool_name_alias(LONG_NAME_2)
    assert a != b


def test_alias_passthrough_for_short_names() -> None:
    short = "get_weather"
    assert app._build_tool_name_alias(short) == short


def test_alias_passthrough_at_exactly_max_len() -> None:
    edge = "x" * app._BEDROCK_TOOL_NAME_MAX
    assert app._build_tool_name_alias(edge) == edge


# ---------------------------------------------------------------------------
# _alias_long_tool_names
# ---------------------------------------------------------------------------


def _tool_body(model: str) -> dict:
    return {
        "model": model,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": LONG_NAME,
                    "description": "",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": LONG_NAME_2,
                    "description": "",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {"name": "short_tool", "description": "", "parameters": {}},
            },
        ],
        "tool_choice": {"type": "function", "function": {"name": LONG_NAME}},
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "toolu_abc",
                        "type": "function",
                        "function": {"name": LONG_NAME, "arguments": "{}"},
                    },
                    {
                        "id": "toolu_def",
                        "type": "function",
                        "function": {"name": "short_tool", "arguments": "{}"},
                    },
                ],
            },
        ],
    }


def test_aliases_rewrites_tools_for_bedrock_upstream() -> None:
    body = _tool_body("qwen.qwen3-coder-480b-a35b-v1:0")
    aliases = app._alias_long_tool_names(body, body["model"])
    assert len(aliases) == 2  # only the two long names
    # All rewritten names fit Bedrock cap.
    for entry in body["tools"]:
        assert len(entry["function"]["name"]) <= app._BEDROCK_TOOL_NAME_MAX
    # Aliases map back to the originals.
    for alias, original in aliases.items():
        assert original in (LONG_NAME, LONG_NAME_2)
        assert len(alias) <= app._BEDROCK_TOOL_NAME_MAX


def test_aliases_rewrites_tool_choice_name() -> None:
    body = _tool_body("qwen.qwen3-coder-480b-a35b-v1:0")
    app._alias_long_tool_names(body, body["model"])
    new_name = body["tool_choice"]["function"]["name"]
    assert new_name != LONG_NAME
    assert len(new_name) <= app._BEDROCK_TOOL_NAME_MAX


def test_aliases_rewrites_assistant_tool_calls_history() -> None:
    body = _tool_body("qwen.qwen3-coder-480b-a35b-v1:0")
    app._alias_long_tool_names(body, body["model"])
    calls = body["messages"][0]["tool_calls"]
    assert len(calls[0]["function"]["name"]) <= app._BEDROCK_TOOL_NAME_MAX
    assert calls[0]["function"]["name"] != LONG_NAME
    # Short name is not touched.
    assert calls[1]["function"]["name"] == "short_tool"


def test_aliases_short_names_are_not_rewritten() -> None:
    body = _tool_body("qwen.qwen3-coder-480b-a35b-v1:0")
    app._alias_long_tool_names(body, body["model"])
    short_entry = next(t for t in body["tools"] if t["function"]["name"] == "short_tool")
    assert short_entry is not None  # untouched


def test_aliases_passthrough_for_anthropic_upstream() -> None:
    body = _tool_body("anthropic.claude-opus-4-7")
    aliases = app._alias_long_tool_names(body, body["model"])
    assert aliases == {}
    # Long names remain.
    assert body["tools"][0]["function"]["name"] == LONG_NAME
    assert body["tools"][1]["function"]["name"] == LONG_NAME_2


def test_aliases_passthrough_for_global_anthropic_upstream() -> None:
    body = _tool_body("global.anthropic.claude-sonnet-4-6")
    assert app._alias_long_tool_names(body, body["model"]) == {}
    assert body["tools"][0]["function"]["name"] == LONG_NAME


def test_aliases_passthrough_for_empty_body() -> None:
    assert app._alias_long_tool_names({"model": ""}, "") == {}
    assert app._alias_long_tool_names({}, "qwen.x") == {}


def test_aliases_deterministic_across_calls() -> None:
    """Same input produces the same alias map (no random salt)."""
    b1 = _tool_body("qwen.qwen3-coder-480b-a35b-v1:0")
    b2 = _tool_body("qwen.qwen3-coder-480b-a35b-v1:0")
    a1 = app._alias_long_tool_names(b1, b1["model"])
    a2 = app._alias_long_tool_names(b2, b2["model"])
    assert a1 == a2


# ---------------------------------------------------------------------------
# openai_to_anthropic_response reverse map
# ---------------------------------------------------------------------------


def _upstream_with_tool_call(name: str) -> dict:
    return {
        "id": "resp_abc",
        "model": "qwen.qwen3-coder-480b-a35b-v1:0",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_bdrk_xyz",
                            "type": "function",
                            "function": {"name": name, "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_response_reverses_alias_to_original_tool_name() -> None:
    alias = app._build_tool_name_alias(LONG_NAME)
    upstream = _upstream_with_tool_call(alias)
    anth, by_name = app.openai_to_anthropic_response(
        upstream, "qwen.qwen3-coder-480b-a35b-v1:0", tool_name_aliases={alias: LONG_NAME}
    )
    tool_use = next(c for c in anth["content"] if c["type"] == "tool_use")
    assert tool_use["name"] == LONG_NAME
    assert LONG_NAME in by_name
    assert by_name[LONG_NAME] == 1
    # Alias name MUST NOT leak into per-name counter.
    assert alias not in by_name


def test_response_passthrough_when_no_aliases() -> None:
    upstream = _upstream_with_tool_call("get_weather")
    anth, by_name = app.openai_to_anthropic_response(
        upstream, "anthropic.claude-opus-4-7", tool_name_aliases=None
    )
    tool_use = next(c for c in anth["content"] if c["type"] == "tool_use")
    assert tool_use["name"] == "get_weather"
    assert by_name == {"get_weather": 1}


def test_response_unmatched_upstream_name_is_kept_as_is() -> None:
    """Upstream returns a name that's not in the alias map (e.g. native short name) — keep verbatim."""
    upstream = _upstream_with_tool_call("native_tool")
    anth, _by_name = app.openai_to_anthropic_response(
        upstream,
        "qwen.qwen3-coder-480b-a35b-v1:0",
        tool_name_aliases={"some_alias": "some_long_original"},
    )
    tool_use = next(c for c in anth["content"] if c["type"] == "tool_use")
    assert tool_use["name"] == "native_tool"


# ---------------------------------------------------------------------------
# anthropic_to_openai integration — alias map ends up in cache_metric
# ---------------------------------------------------------------------------


def test_anthropic_to_openai_publishes_alias_map_in_cache_metric() -> None:
    body = {
        "model": "qwen.qwen3-coder-480b-a35b-v1:0",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": LONG_NAME, "description": "x", "input_schema": {"type": "object"}},
            {"name": "short_tool", "description": "y", "input_schema": {"type": "object"}},
        ],
    }
    out, cache_metric = app.anthropic_to_openai(body)
    assert "tool_name_aliases" in cache_metric
    aliases = cache_metric["tool_name_aliases"]
    assert len(aliases) == 1
    # The rewritten name appears in out["tools"]; the original does not.
    names_out = {t["function"]["name"] for t in out["tools"]}
    assert LONG_NAME not in names_out
    assert "short_tool" in names_out


def test_anthropic_to_openai_no_aliases_for_anthropic_upstream() -> None:
    body = {
        "model": "anthropic.claude-opus-4-7",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": LONG_NAME, "description": "x", "input_schema": {"type": "object"}},
        ],
    }
    out, cache_metric = app.anthropic_to_openai(body)
    assert cache_metric.get("tool_name_aliases") == {}
    assert out["tools"][0]["function"]["name"] == LONG_NAME
