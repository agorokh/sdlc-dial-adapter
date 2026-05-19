"""Tests for the dynamic ``max_tokens`` clamp.

Claude Code unconditionally sends ``max_tokens=32000`` on every streaming
request. Anthropic upstreams (200k+ context) handle it fine; Qwen-on-Bedrock
caps at 131k. Once input crosses ~99k tokens, even a one-shot reply is
impossible — a long agentic-loop session deterministically 400s.

The adapter clamps ``max_tokens`` to the remaining budget
(``max_context - estimated_input - 4096-token safety margin``) for
non-Anthropic upstreams, and surfaces the clamp event on
``cache_metric["max_tokens_clamp"]`` for the request_in log. When the
budget goes non-positive — meaning the input alone exceeds the upstream's
context window — the clamp raises ``TranslationError(400)`` with
diagnostic detail rather than serving a request that will
deterministically 400 upstream.

These tests cover:

* Curated context-window lookups (qwen.*, moonshotai.*, minimax.*, etc.).
* Clamp fires when budget < requested, leaving the diagnostic dict.
* Clamp is a no-op when budget >= requested.
* TranslationError on input-exceeds-context (not a silent floor).
* Anthropic upstreams are NOT in the map (intentional — their own limits
  enforce).
* Unknown prefixes are left alone (upstream rejects if it must).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ADAPTER_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anthropic_dial_adapter_app", ADAPTER_DIR / "app.py")
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
sys.modules["anthropic_dial_adapter_app"] = app
spec.loader.exec_module(app)


# ---------------------------------------------------------------------------
# _model_max_context_tokens — prefix map lookups
# ---------------------------------------------------------------------------


def test_qwen_exact_match_returns_131k() -> None:
    assert app._model_max_context_tokens("qwen.qwen3-coder-480b-a35b-v1:0") == 131_072


def test_qwen_prefix_fallback_returns_131k() -> None:
    # Any qwen.* deployment we don't have an exact match for falls back to the
    # prefix entry.
    assert app._model_max_context_tokens("qwen.qwen-some-future-model") == 131_072


def test_moonshotai_kimi_returns_128k() -> None:
    assert app._model_max_context_tokens("moonshotai.kimi-k2.5") == 128_000


def test_minimax_returns_245k() -> None:
    assert app._model_max_context_tokens("minimax.minimax-m2.5") == 245_760


def test_unknown_model_returns_none() -> None:
    # Unknown prefix → adapter leaves max_tokens alone; upstream enforces.
    assert app._model_max_context_tokens("unknown.model-v1") is None


def test_empty_model_returns_none() -> None:
    assert app._model_max_context_tokens("") is None


def test_anthropic_upstreams_not_in_map() -> None:
    # Intentional — Anthropic's own limits (200k+) handle themselves.
    assert app._model_max_context_tokens("anthropic.claude-sonnet-4-6") is None
    assert app._model_max_context_tokens("global.anthropic.claude-opus-4-7") is None


# ---------------------------------------------------------------------------
# _clamp_max_tokens_to_fit_context — behavior
# ---------------------------------------------------------------------------


def test_clamp_noop_when_input_is_small() -> None:
    body = {
        "model": "qwen.qwen3-coder-480b-a35b-v1:0",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
    }
    cache_metric: dict = {}
    app._clamp_max_tokens_to_fit_context(body, cache_metric)
    # Small input → plenty of room → no clamp
    assert body["max_tokens"] == 32000
    assert "max_tokens_clamp" not in cache_metric


def test_clamp_fires_when_input_crowds_context() -> None:
    """~95k-token-equivalent input vs 131k Qwen cap with max_tokens=32k →
    budget = 131k - 95k - 4k = 32k, just barely under requested → clamp fires."""
    body = {
        "model": "qwen.qwen3-coder-480b-a35b-v1:0",
        "max_tokens": 32000,
        # 380k chars / 4 ≈ 95k token estimate. Plus JSON envelope, will trip
        # the clamp but not exceed context.
        "messages": [{"role": "user", "content": "x" * (95_000 * 4)}],
    }
    cache_metric: dict = {}
    app._clamp_max_tokens_to_fit_context(body, cache_metric)
    assert body["max_tokens"] < 32000
    assert body["max_tokens"] > 0
    clamp = cache_metric["max_tokens_clamp"]
    assert clamp["original"] == 32000
    assert clamp["clamped"] == body["max_tokens"]
    assert clamp["max_context"] == 131_072
    assert clamp["model"] == "qwen.qwen3-coder-480b-a35b-v1:0"
    assert clamp["estimated_input_tokens"] > 0


def test_clamp_raises_on_input_exceeds_context() -> None:
    """When the input alone exceeds the upstream's context window, the adapter
    refuses to serve the request — raises ``TranslationError`` (400) with a
    diagnostic detail dict. Better than silently truncating to a useless floor
    and producing a doomed upstream call."""
    body = {
        "model": "qwen.qwen3-coder-480b-a35b-v1:0",
        "max_tokens": 32000,
        # Way over the 131k cap — input alone exceeds the model context.
        "messages": [{"role": "user", "content": "x" * (200_000 * 4)}],
    }
    cache_metric: dict = {}
    with pytest.raises(app.TranslationError) as excinfo:
        app._clamp_max_tokens_to_fit_context(body, cache_metric)
    # Status code is 400 (client must shorten the prompt).
    assert excinfo.value.status == 400
    # Diagnostic detail is structured for downstream logging.
    detail = excinfo.value.detail.get("input_exceeds_context")
    assert detail is not None
    assert detail["model"] == "qwen.qwen3-coder-480b-a35b-v1:0"
    assert detail["max_context"] == 131_072
    assert detail["estimated_input_tokens"] > 131_072
    assert detail["remaining_output_budget"] <= 0


def test_clamp_skipped_for_anthropic_upstream() -> None:
    """Anthropic models are not in _MODEL_MAX_CONTEXT → no clamp applied."""
    body = {
        "model": "anthropic.claude-sonnet-4-6",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "x" * (200_000 * 4)}],
    }
    cache_metric: dict = {}
    app._clamp_max_tokens_to_fit_context(body, cache_metric)
    assert body["max_tokens"] == 32000
    assert "max_tokens_clamp" not in cache_metric


def test_clamp_skipped_for_unknown_model() -> None:
    body = {
        "model": "unknown.experimental-v1",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "x" * (200_000 * 4)}],
    }
    cache_metric: dict = {}
    app._clamp_max_tokens_to_fit_context(body, cache_metric)
    assert body["max_tokens"] == 32000
    assert "max_tokens_clamp" not in cache_metric


def test_clamp_skipped_when_max_tokens_missing() -> None:
    body = {
        "model": "qwen.qwen3-coder-480b-a35b-v1:0",
        "messages": [{"role": "user", "content": "x" * (110_000 * 4)}],
    }
    cache_metric: dict = {}
    app._clamp_max_tokens_to_fit_context(body, cache_metric)
    # No max_tokens to clamp → no-op
    assert "max_tokens" not in body
    assert "max_tokens_clamp" not in cache_metric


def test_clamp_constants_are_sane() -> None:
    """Sanity-check the constants the clamp relies on. If someone tweaks them
    blindly, this test catches obviously-wrong values."""
    assert app._CONTEXT_SAFETY_MARGIN_TOKENS > 0
    # Safety margin should be smaller than every mapped context window.
    for prefix, ctx in app._MODEL_MAX_CONTEXT.items():
        assert app._CONTEXT_SAFETY_MARGIN_TOKENS < ctx, \
            f"safety margin too large for {prefix} ({ctx} tokens)"
