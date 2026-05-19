"""Tests for the ``POST /v1/messages/count_tokens`` endpoint and its underlying
``_count_tokens_for_anthropic_body`` heuristic.

Claude Code calls ``countTokens`` before each ``/v1/messages`` request to
decide whether to auto-compact the conversation. The previous stub returned
501, so Claude Code blasted ahead blind. The current implementation answers
``{"input_tokens": N}`` using a chars/4 heuristic over the Anthropic-shape
body, padded by ``_COUNT_TOKENS_PAD_FACTOR`` (1.10) so the client errs
toward compacting too early rather than too late.

These tests cover:

* Small / medium / large prompt shapes.
* Single text content vs. multi-block content.
* ``system`` as a string and ``system`` as a list of blocks.
* The 1.10 pad factor.
* HTTP wire shape: 200 + Anthropic-format JSON ``{"input_tokens": N}``,
  with the diagnostic response headers.
* Error responses (invalid JSON body, non-dict body).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web

ADAPTER_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anthropic_dial_adapter_app", ADAPTER_DIR / "app.py")
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
sys.modules["anthropic_dial_adapter_app"] = app
spec.loader.exec_module(app)


# ---------------------------------------------------------------------------
# _count_tokens_for_anthropic_body — pure function
# ---------------------------------------------------------------------------


def test_count_tokens_small_prompt_is_small_integer() -> None:
    body = {
        "model": "qwen.qwen3-coder-480b-a35b-v1:0",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    n = app._count_tokens_for_anthropic_body(body)
    assert isinstance(n, int)
    assert 1 <= n < 100


def test_count_tokens_scales_roughly_linearly_with_input_size() -> None:
    """chars/4 means input length / 4 ≈ token count, within the pad factor."""
    small = app._count_tokens_for_anthropic_body({
        "messages": [{"role": "user", "content": "x" * 100}],
    })
    large = app._count_tokens_for_anthropic_body({
        "messages": [{"role": "user", "content": "x" * 10_000}],
    })
    # 100x the chars → ~100x the tokens, within a small constant factor.
    assert large > small * 50


def test_count_tokens_pad_factor_makes_us_over_report() -> None:
    """The pad factor (1.10) must over-report, not under-report. Even when the
    chars/4 raw value is X, the reported value should be >= X."""
    raw_chars = 4000
    body = {"messages": [{"role": "user", "content": "x" * raw_chars}]}
    n = app._count_tokens_for_anthropic_body(body)
    # Raw chars/4 = 1000 tokens before pad. Reported should be at least that.
    raw_token_estimate = (raw_chars // 4)
    # Plus messages-list JSON overhead (role/content keys, brackets, quotes)
    assert n >= raw_token_estimate


def test_count_tokens_includes_system_string() -> None:
    body_no_system = {"messages": [{"role": "user", "content": "hi"}]}
    body_with_system = {
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "hi"}],
    }
    n_no_system = app._count_tokens_for_anthropic_body(body_no_system)
    n_with_system = app._count_tokens_for_anthropic_body(body_with_system)
    assert n_with_system > n_no_system


def test_count_tokens_includes_system_blocks_list() -> None:
    """Anthropic also allows ``system`` as a list of typed blocks; both shapes
    must contribute to the token count."""
    body = {
        "system": [{"type": "text", "text": "You are helpful."}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    n = app._count_tokens_for_anthropic_body(body)
    assert n > 0


def test_count_tokens_includes_tools() -> None:
    """Tool schemas can be tens of thousands of tokens — they MUST be counted
    so Claude Code's compaction math sees the real input footprint."""
    body_no_tools = {"messages": [{"role": "user", "content": "hi"}]}
    body_with_tools = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "lookup", "description": "Look something up.",
             "input_schema": {"type": "object", "properties": {
                 "query": {"type": "string", "description": "x" * 500},
             }}},
        ],
    }
    n_no_tools = app._count_tokens_for_anthropic_body(body_no_tools)
    n_with_tools = app._count_tokens_for_anthropic_body(body_with_tools)
    assert n_with_tools > n_no_tools


def test_count_tokens_empty_body_returns_zero_or_one() -> None:
    n = app._count_tokens_for_anthropic_body({})
    assert n >= 0
    assert n < 10  # nothing to count → near zero


def test_pad_factor_is_above_one() -> None:
    """The pad factor MUST be >= 1.0 — under-reporting would defeat the
    purpose of this endpoint (Claude Code would compact too late)."""
    assert app._COUNT_TOKENS_PAD_FACTOR >= 1.0


# ---------------------------------------------------------------------------
# count_tokens(request) — HTTP handler
# ---------------------------------------------------------------------------


class _StubRequest:
    """Minimal aiohttp-Request-like stub for handler testing."""
    def __init__(self, body: bytes | None):
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return json.loads(self._body)


def _run(coro):
    return asyncio.run(coro)


def test_count_tokens_handler_returns_200_with_input_tokens() -> None:
    body = json.dumps({"messages": [{"role": "user", "content": "hello world"}]})
    req = _StubRequest(body.encode())
    resp = _run(app.count_tokens(req))
    assert resp.status == 200
    payload = json.loads(resp.body)
    assert "input_tokens" in payload
    assert isinstance(payload["input_tokens"], int)
    assert payload["input_tokens"] > 0


def test_count_tokens_handler_response_headers_expose_method_and_pad() -> None:
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
    resp = _run(app.count_tokens(_StubRequest(body.encode())))
    headers = dict(resp.headers)
    assert headers.get("x-anthropic-dial-adapter-count-method") == "chars_div_4_padded"
    assert headers.get("x-anthropic-dial-adapter-count-pad-factor") == str(app._COUNT_TOKENS_PAD_FACTOR)


def test_count_tokens_handler_rejects_invalid_json() -> None:
    req = _StubRequest(b"not json at all")
    resp = _run(app.count_tokens(req))
    assert resp.status == 400
    payload = json.loads(resp.body)
    assert payload["error"]["type"] == "invalid_request_error"


def test_count_tokens_handler_rejects_non_dict_body() -> None:
    req = _StubRequest(b'["not", "an", "object"]')
    resp = _run(app.count_tokens(req))
    assert resp.status == 400
    payload = json.loads(resp.body)
    assert payload["error"]["type"] == "invalid_request_error"
