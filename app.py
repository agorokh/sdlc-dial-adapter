#!/usr/bin/env python3
"""Anthropic Messages API to OpenAI chat-completions translator.

Accepts Anthropic-shape POST /v1/messages requests from clients (e.g.
Claude Code CLI) and forwards them to an OpenAI-compatible gateway at:

    POST {UPSTREAM_BASE}/openai/deployments/{model}/chat/completions
         ?api-version={DIAL_API_VERSION}

The default target is EPAM AI DIAL (https://ai-proxy.lab.epam.com), but
any OpenAI chat-completions endpoint will work given a compatible
UPSTREAM_BASE and PROJECT_KEY.

Authentication: clients send any of {x-api-key, Authorization: Bearer ...};
the adapter substitutes `Api-Key: $PROJECT_KEY` read from the environment
at start-up. No credentials are persisted to disk.

Translation covers:
  - top-level `system` (string or content blocks) -> leading OpenAI system msg
  - text / image / tool_use / tool_result content blocks in both directions
  - tools[] (name/description/input_schema) <-> OpenAI tools (function)
  - tool_choice mapping (auto / any / tool / none)
  - cache_control (passthrough on Anthropic upstreams; stripped on others)
  - non-streaming JSON responses repackaged as Anthropic Messages responses
  - SSE event sequence: chat.completion.chunk -> message_start ->
    content_block_start / delta / stop -> message_delta -> message_stop
  - stable 1:1 tool_use_id mapping (Bedrock-style `toolu_bdrk_...` ids
    are preserved verbatim across the translation boundary)
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Coroutine, Sequence
from typing import Any, Union

import aiohttp
from aiohttp import web
from ccppm.dial_pricing import estimate_cost_usd, load_price_table_env
from ccppm.log_window import _family_from_model

UPSTREAM = os.environ.get("UPSTREAM_BASE", "https://ai-proxy.lab.epam.com").rstrip("/")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8092"))
DIAL_API_VERSION = os.environ.get("DIAL_API_VERSION", "2024-02-01")
PROJECT_KEY = os.environ.get("PROJECT_KEY", "")
LOG_PATH = os.environ.get(
    "ANTHROPIC_DIAL_ADAPTER_LOG",
    "/var/log/anthropic-dial-adapter/adapter.log",
)
# Shadow-mode dual-dispatch. When set, every primary response is
# mirrored to a second upstream deployment (typically an OSS Coder model
# like `qwen.qwen3-coder-480b-a35b-v1:0`) and the structural diff is logged
# to SHADOW_LOG_PATH. Fire-and-forget — does NOT block the client. Prep
# accumulating signal from real Claude Code sessions hitting the adapter.
SHADOW_MODEL = os.environ.get("ANTHROPIC_DIAL_SHADOW_MODEL", "").strip()
SHADOW_LOG_PATH = os.environ.get(
    "ANTHROPIC_DIAL_SHADOW_LOG",
    "/var/log/anthropic-dial-adapter/shadow.log",
)

# Live alias mechanism.
# JSON object mapping client-requested model id -> target DIAL deployment id.
# Example:
#   ANTHROPIC_DIAL_ALIASES_JSON='{"claude-sonnet-4-6":"qwen.qwen3-coder-480b-a35b-v1:0"}'
# Match happens AFTER model normalization, so keys can be either the short
# form ("claude-sonnet-4-6") or the namespaced form
# ("anthropic.claude-sonnet-4-6"). When matched, the body's model field is
# rewritten to the target before forwarding upstream, and the response's
# model field is rewritten back to the client-requested name so the calling
# tool (Claude Code) sees what it asked for.
#
# ``ANTHROPIC_DIAL_ALIASES_QWEN_JSON`` is an optional secondary alias var
# read when the primary env above is unset or blank. Useful in deployments
# where a primary alias map is shared and a secondary instance overrides a
# subset (e.g. routing a single picker slot to a different upstream).
def _load_aliases_map() -> dict[str, str]:
    raw = os.environ.get("ANTHROPIC_DIAL_ALIASES_JSON", "").strip()
    if not raw:
        raw = os.environ.get("ANTHROPIC_DIAL_ALIASES_QWEN_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as e:
        logger.warning(
            json.dumps({
                "event": "aliases_parse_failed",
                "adapter": "anthropic_dial_adapter",
                "error": str(e),
                "raw_len": len(raw),
            })
        )
        return {}
    if not isinstance(parsed, dict):
        logger.warning(
            json.dumps({
                "event": "aliases_not_a_dict",
                "adapter": "anthropic_dial_adapter",
                "type": type(parsed).__name__,
            })
        )
        return {}
    # Validate: keys + values must be non-empty strings, and the value must
    # pass the same upstream-id regex as a direct model selection (no path
    # traversal, no whitespace shenanigans).
    out: dict[str, str] = {}
    for k, v in parsed.items():
        if not isinstance(k, str) or not isinstance(v, str) or not k or not v:
            continue
        if not _DEPLOYMENT_ID_RE.fullmatch(v):
            logger.warning(
                json.dumps({
                    "event": "alias_target_rejected",
                    "adapter": "anthropic_dial_adapter",
                    "key": k,
                    "target": v,
                    "reason": "fails deployment-id regex",
                })
            )
            continue
        out[k] = v
    return out


_ALIASES_MAP: dict[str, str] = {}  # populated at startup via _load_aliases_map()
# Operator-configurable Bedrock list-price table (USD per 1M tokens).
_PRICE_TABLE: dict[str, dict[str, float]] = {}

# `logger` writes structured newline-delimited JSON suitable for any line-oriented
# log shipper (Vector, Fluent Bit, Promtail, etc.).
# stderr also receives a short human line for `docker logs`.
logger = logging.getLogger("anthropic_dial_adapter")
logger.setLevel(logging.INFO)


def _setup_logging() -> None:
    if logger.handlers:
        return  # idempotent — safe to call from both __main__ and on_startup
    fmt = logging.Formatter("%(message)s")
    # Best-effort file handler — falls back to stderr-only if path unwritable.
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[anthropic-dial-adapter] %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False


def emit(event: str, **fields: Any) -> None:
    """Single GFLog-shape JSON line. Vector tails `LOG_PATH`."""
    rec = {
        "@timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "adapter": "anthropic_dial_adapter",
        "event": event,
        **fields,
    }
    logger.info(json.dumps(rec, separators=(",", ":"), default=str))


def _gflog_client_name(request: web.Request) -> str:
    raw = (request.headers.get("User-Agent") or "").strip() or "unknown"
    raw = re.sub(r"\s+", " ", raw)
    return raw[:160]


def _gflog_target_model_family(model_id: str | None) -> str:
    """Upstream family for GFLog — same rules as :func:`ccppm.log_window._family_from_model`.

    Keeps ``adapter_metrics`` tags aligned with the exporter (Bedrock
    ``us.``/``eu.``/… regional inference ids map to the provider segment, not
    the region token).
    """
    return _family_from_model(model_id)


def _gflog_metrics_tags(request: web.Request, upstream_model_id: str | None) -> dict[str, str]:
    """Stable tag dimensions for the Influx ``adapter_metrics`` exporter."""
    return {
        "client_name": _gflog_client_name(request),
        "target_model_family": _gflog_target_model_family(upstream_model_id),
    }


def _cost_usd_estimate_kwargs(
    upstream_model_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> dict[str, Any]:
    """GFLog fields for ``response_out`` — omitted when model is not in the price table."""
    if not _PRICE_TABLE:
        return {}
    est = estimate_cost_usd(
        model_id=upstream_model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        table=_PRICE_TABLE,
    )
    if est is None:
        return {}
    return {"cost_usd_estimate": est}


# Shadow-mode log writer (separate file so a log shipper can tail
# /var/log/anthropic-dial-adapter/*.log and route by source if desired).
_shadow_logger = logging.getLogger("anthropic_dial_adapter_shadow")
_shadow_logger.setLevel(logging.INFO)


def _setup_shadow_logging() -> None:
    if _shadow_logger.handlers:
        return
    try:
        os.makedirs(os.path.dirname(SHADOW_LOG_PATH), exist_ok=True)
        fh = logging.FileHandler(SHADOW_LOG_PATH)
        fh.setFormatter(logging.Formatter("%(message)s"))
        _shadow_logger.addHandler(fh)
    except OSError:
        pass
    _shadow_logger.propagate = False


def shadow_emit(event: str, **fields: Any) -> None:
    rec = {
        "@timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "adapter": "anthropic_dial_adapter",
        "stream_source": "shadow",
        "event": event,
        **fields,
    }
    _shadow_logger.info(json.dumps(rec, separators=(",", ":"), default=str))


# ---------------------------------------------------------------------------
# Anthropic → OpenAI request translation
# ---------------------------------------------------------------------------

STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    # Upstream safety / moderation — not a user stop_sequence match.
    "content_filter": "refusal",
    "function_call": "tool_use",  # legacy
}

# ---------------------------------------------------------------------------
# Upstream cache_control support state.
# ---------------------------------------------------------------------------
# Anthropic Claude Code sends `cache_control: {"type":"ephemeral"}` markers on
# system blocks, the trailing user turn, and tool definitions. The Bedrock-
# Anthropic upstream supports caching via its native `cachePoint` shape, but
# DIAL's `ai-proxy.lab.epam.com` aiproxy adapter has a
# strict request validator that REJECTS every known cache_control shape with
# HTTP 400 "Extra inputs are not permitted" — inline content-block field,
# `custom_fields.cache_breakpoint`, and top-level message field all fail.
#
# Until the upstream aiproxy is patched, the adapter strips cache_control
# silently (current behavior) and tags every affected request with a
# `cache_control_strategy` log field so the AI-SDLC dashboard can surface the
# economic cost honestly. The startup probe runs once and caches the result.
_UPSTREAM_CACHE_CONTROL_SUPPORT: str | None = None  # supported|rejected|probe_failed|no_probe_model
_CACHE_PROBE_MODEL = os.environ.get(
    "ANTHROPIC_DIAL_CACHE_PROBE_MODEL",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
)


def _cache_control_strategy() -> str:
    """Map the upstream-probe result to a stable strategy tag used in metrics."""
    state = _UPSTREAM_CACHE_CONTROL_SUPPORT
    if state == "supported":
        return "passthrough"
    if state == "rejected":
        return "stripped_upstream_rejects"
    if state == "no_probe_model":
        return "stripped_no_probe_model"
    if state == "probe_failed":
        return "stripped_probe_failed"
    return "stripped_probe_pending"


async def _probe_upstream_cache_control_support(
    session: aiohttp.ClientSession,
) -> str:
    """One-shot startup probe.
    https://docs.dialx.ai/tutorials/developers/prompt-caching:

    DIAL expects `custom_fields.cache_breakpoint: {}` (empty marker, or with
    optional `expire_at`) on the message/tools entry — NOT Anthropic's
    `cache_control: {type: ephemeral}` inner shape. Our earlier probe was
    sending the Anthropic inner shape, which the strict DIAL validator
    correctly rejected as "Extra inputs are not permitted" — that was a bug
    in the probe, not a DIAL limitation.

    Returns one of: "supported" | "rejected" | "probe_failed" | "no_probe_model".

    "supported" only means the validator accepts the field. End-to-end
    caching ALSO requires `cacheSupported: true` on the deployment's DIAL
    config (see https://github.com/epam/ai-dial-core/blob/development/docs/
    dynamic-settings/models.md) and that the prompt exceeds the upstream's
    minimum cacheable size (Bedrock-Anthropic: 1024+ tokens). The probe
    cannot tell whether `cacheSupported` is set — only the request handler
    measures that, via `cache_read_input_tokens` in usage.
    """
    if not PROJECT_KEY:
        return "probe_failed"
    target = (
        f"{UPSTREAM}/openai/deployments/{_CACHE_PROBE_MODEL}/chat/completions"
        f"?api-version={DIAL_API_VERSION}"
    )
    # DIAL-shape per docs: custom_fields.cache_breakpoint is an empty marker
    # object placed on the system message (or any cache breakpoint position).
    body = {
        "messages": [
            {
                "role": "system",
                "content": "Be brief.",
                "custom_fields": {"cache_breakpoint": {}},
            },
            {"role": "user", "content": "ok"},
        ],
        "max_tokens": 4,
    }
    try:
        async with session.post(
            target,
            headers={"Api-Key": PROJECT_KEY, "Content-Type": "application/json"},
            data=json.dumps(body).encode(),
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                return "supported"
            err_body = (await resp.read()).decode("utf-8", "replace").lower()
            if resp.status == 404:
                return "no_probe_model"
            if resp.status >= 400 and (
                "cache_breakpoint" in err_body or "custom_fields" in err_body
            ):
                return "rejected"
            return "probe_failed"
    except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError):
        return "probe_failed"


class TranslationError(Exception):
    """Raised when an inbound Anthropic body cannot be translated."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


_DEPLOYMENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]+$")


def _validate_upstream_model_id(model: str) -> None:
    """Reject path/query injection in the Azure deployment segment."""
    if not _DEPLOYMENT_ID_RE.fullmatch(model):
        raise TranslationError(
            "model must be [A-Za-z0-9_.:.-]+ (safe deployment id for upstream URL)",
            status=400,
        )


# ---------------------------------------------------------------------------
# Tool-schema hash and MCP classification.
# ---------------------------------------------------------------------------


def _canonical_sha(obj: Any) -> str:
    """Stable SHA256 over a JSON-canonicalized object. Used to detect any
    drift between the inbound tool definition and the outbound shape we
    forward to the upstream. A correct adapter MUST preserve the
    name+description+schema triple byte-for-byte (modulo key renaming
    `input_schema` → `parameters`).
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]  # 64-bit prefix is plenty for drift detection


# Claude Code MCP-server convention: server-injected tools are prefixed
# `mcp__<server>__<tool>`. Native tools (Bash, Read, Write, Edit, …) are
# bare CapitalCase names. Anything else is either a user-defined plugin
# tool or, rarely, a custom subagent.
_NATIVE_TOOLS = {
    "Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep",
    "Task", "TodoWrite", "WebFetch", "WebSearch", "NotebookEdit",
    "KillBash", "BashOutput",
}


def _classify_tool_name(name: str) -> tuple[str, str | None]:
    """Returns (kind, mcp_server_or_none) where kind ∈ {native, mcp, other}."""
    if name in _NATIVE_TOOLS:
        return "native", None
    prefix = "mcp__"
    if name.startswith(prefix):
        rest = name[len(prefix):]
        sep = rest.find("__")
        if sep > 0:
            server = rest[:sep]
            if server:
                return "mcp", server
    return "other", None


def _flatten_text_blocks(blocks: Any) -> str:
    """Anthropic system can be string or list of {type:"text", text:"..."}."""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        parts: list[str] = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
        return "\n".join(parts)
    return ""


def _translate_content_to_openai(
    content: Any,
    *,
    cache_metric: dict[str, int],
) -> Any:
    """Convert an Anthropic message `content` field into OpenAI-shape parts.

    Strings stay strings. Lists of blocks become a list of OpenAI parts.
    Returns the value to drop into the OpenAI message dict (or None if the
    block list only contained tool_use / tool_result entries that the caller
    must promote to top-level OpenAI messages).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        # cache_control passthrough metric — DIAL/Bedrock may accept it inline;
        # adapter does not currently translate to custom_fields.cache_breakpoint.
        if "cache_control" in block:
            cache_metric["seen"] = cache_metric.get("seen", 0) + 1
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source") or {}
            if src.get("type") == "base64":
                media = src.get("media_type", "image/png")
                data = src.get("data", "")
                url = f"data:{media};base64,{data}"
            elif src.get("type") == "url":
                url = src.get("url", "")
            else:
                url = ""
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        # tool_use and tool_result are handled at the message level (see below).
    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        # Collapse single-text-part to string for upstream parsers that prefer it.
        return parts[0]["text"]
    return parts


def anthropic_to_openai(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a parsed Anthropic /v1/messages body into an OpenAI chat
    completions body. Returns (openai_body, cache_metric).

    cache_metric carries:
      - seen:        count of inbound `cache_control` markers
      - translated:  count of markers translated to DIAL `custom_fields.cache_breakpoint`
      - strategy:    one of "translated_dial_breakpoint"|"stripped_upstream_rejects"|
                     "stripped_probe_failed"|"stripped_no_probe_model"|
                     "stripped_probe_pending" — derived from startup probe verdict.
                     When supported, the adapter ACTIVELY TRANSLATES.
    """
    out: dict[str, Any] = {}
    strat = _cache_control_strategy()
    # When the upstream accepts DIAL-shape breakpoints, switch from "stripped"
    # reporting to "translated_dial_breakpoint" — this is the actual
    # operational mode.
    if strat == "passthrough":
        strat = "translated_dial_breakpoint"

    # Non-Anthropic DIAL deployments (qwen.*, mistral.*, deepseek.*) reject
    # requests carrying `custom_fields.cache_breakpoint` with a hard
    # `502 No route` at the DIAL gateway — Claude Code sets cache_control on
    # its system prompt and tool inventory, so every real Claude Code request
    # fails 502 against those upstreams. Drop translation for non-Anthropic
    # targets and report it via a distinct strategy tag so the AI-SDLC
    # dashboard surfaces the operational gap when this happens.
    
    target_model = str(body.get("model", "") or "")
    target_is_anthropic = target_model.startswith(("anthropic.", "global.anthropic."))
    if not target_is_anthropic and target_model:
        strat = "dropped_non_anthropic_upstream"

    cache_metric: dict[str, Any] = {"seen": 0, "translated": 0, "strategy": strat}
    _can_translate = (
        _UPSTREAM_CACHE_CONTROL_SUPPORT == "supported"
        and target_is_anthropic
    )

    if "model" in body:
        out["model"] = body["model"]
    if "max_tokens" in body and body["max_tokens"] is not None:
        out["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]
    if body.get("stream"):
        out["stream"] = True
        # Azure/OpenAI chat-completions only emit usage chunks when asked.
        out["stream_options"] = {"include_usage": True}

    messages: list[dict[str, Any]] = []
    known_tool_use_ids: set[str] = set()

    # 1. system prompt → leading system message
    if body.get("system"):
        sys_text = _flatten_text_blocks(body["system"])
        sys_has_cache = False
        if isinstance(body["system"], list):
            for b in body["system"]:
                if isinstance(b, dict) and "cache_control" in b:
                    cache_metric["seen"] += 1
                    sys_has_cache = True
        if sys_text:
            sys_msg: dict[str, Any] = {"role": "system", "content": sys_text}
            if sys_has_cache and _can_translate:
                # Translate Anthropic cache_control → DIAL cache_breakpoint
                # (https://docs.dialx.ai/tutorials/developers/prompt-caching)
                sys_msg["custom_fields"] = {"cache_breakpoint": {}}
                cache_metric["translated"] += 1
            messages.append(sys_msg)

    # 2. messages[] — walk and split assistant tool_use / user tool_result
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, list):
            tool_use_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            tool_result_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            text_image_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") in ("text", "image")
            ]

            if role == "assistant" and tool_use_blocks:
                # Assistant message that includes tool calls.
                tool_calls = []
                for tu in tool_use_blocks:
                    if "cache_control" in tu:
                        cache_metric["seen"] += 1
                    call_id = tu.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                    known_tool_use_ids.add(str(call_id))
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": json.dumps(tu.get("input") or {}, separators=(",", ":")),
                        },
                    })
                msg_out: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
                if text_image_blocks:
                    msg_out["content"] = _translate_content_to_openai(
                        text_image_blocks, cache_metric=cache_metric
                    )
                else:
                    msg_out["content"] = None
                messages.append(msg_out)
                continue

            if role == "user" and tool_result_blocks:
                # Each Anthropic tool_result becomes its own OpenAI `tool` message
                # (OpenAI does not allow combining tool results with user text in
                # one message). Any sibling text/image blocks become a separate
                # user message AFTER the tool messages.
                for tr in tool_result_blocks:
                    tool_use_id = tr.get("tool_use_id")
                    if not tool_use_id:
                        raise TranslationError("tool_result block is missing 'tool_use_id'")
                    sid = str(tool_use_id)
                    if sid not in known_tool_use_ids:
                        raise TranslationError(
                            f"tool_result references unknown tool_use_id {sid!r}",
                            status=502,
                        )
                    tr_raw = tr.get("content")
                    if isinstance(tr_raw, list):
                        tr_openai = _translate_content_to_openai(
                            tr_raw, cache_metric=cache_metric
                        )
                    elif isinstance(tr_raw, str):
                        tr_openai = tr_raw
                    elif tr_raw is None:
                        tr_openai = ""
                    else:
                        tr_openai = str(tr_raw)
                    # Non-Anthropic DIAL deployments route through Bedrock
                    # Converse. Two observed failure modes for tool-message
                    # content on the OSS path:
                    #
                    # (a) Bare string that happens to parse as JSON
                    #     (e.g. ``gh pr list --json`` returns
                    #     ``[{"number":42},…]``) → DIAL auto-parses, stuffs
                    #     the value into ``toolResult.content[0].json``,
                    #     Bedrock 400s because ``json`` must be an OBJECT.
                    # (b) OpenAI content-parts list
                    #     ``[{"type":"text","text":"…"}]`` → DIAL gateway
                    #     returns 502 "No route" because the chat-completions
                    #     contract requires bare-string content for ``tool``.
                    #
                    # Workaround: keep bare-string but wrap payload in a
                    # guaranteed JSON OBJECT ``{"output": <text>}``. DIAL
                    # auto-parse now lands on a valid object, Bedrock
                    # accepts, the receiving model unwraps transparently.
                    # Anthropic-on-DIAL keeps its existing path untouched.
                    target_model = str(body.get("model", "") or "")
                    target_is_anthropic = target_model.startswith(
                        ("anthropic.", "global.anthropic.")
                    )
                    if not target_is_anthropic:
                        if isinstance(tr_openai, list):
                            parts: list[str] = []
                            for part in tr_openai:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    parts.append(part.get("text") or "")
                                elif isinstance(part, dict) and part.get("type") == "image_url":
                                    url = (part.get("image_url") or {}).get("url") or ""
                                    parts.append(f"[image: {url}]" if url else "[image]")
                                elif isinstance(part, dict):
                                    parts.append(json.dumps(part, ensure_ascii=False))
                                else:
                                    parts.append(str(part))
                            tr_openai = "\n".join(s for s in parts if s)
                        elif not isinstance(tr_openai, str):
                            tr_openai = str(tr_openai)
                        tr_openai = json.dumps(
                            {"output": tr_openai}, ensure_ascii=False
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": sid,
                        "content": tr_openai,
                    })
                if text_image_blocks:
                    messages.append({
                        "role": "user",
                        "content": _translate_content_to_openai(
                            text_image_blocks, cache_metric=cache_metric
                        ),
                    })
                continue

        # Default path: plain user/assistant message with text/image content.
        translatable_cc_blocks = (
            sum(
                1
                for b in content
                if isinstance(b, dict)
                and b.get("type") in ("text", "image")
                and "cache_control" in b
            )
            if isinstance(content, list) else 0
        )
        tr_content = _translate_content_to_openai(content, cache_metric=cache_metric)
        om: dict[str, Any] = {"role": role, "content": tr_content}
        if translatable_cc_blocks and _can_translate:
            om["custom_fields"] = {"cache_breakpoint": {}}
            cache_metric["translated"] += translatable_cc_blocks
        messages.append(om)

    out["messages"] = messages

    # 3. tools[] -> OpenAI tools (function), plus schema-drift detection and MCP classification
    tools_in = body.get("tools") or []
    tool_inventory: list[dict[str, Any]] = []
    if tools_in:
        out_tools: list[dict[str, Any]] = []
        for t in tools_in:
            if not isinstance(t, dict):
                continue
            t_has_cache = "cache_control" in t
            if t_has_cache:
                cache_metric["seen"] += 1
            raw_name = t.get("name", "")
            name = raw_name if isinstance(raw_name, str) else ""
            schema_in = t.get("input_schema") or {"type": "object", "properties": {}}
            description = t.get("description", "")
            schema_sha_in = _canonical_sha({"n": name, "d": description, "s": schema_in})
            out_entry = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema_in,  # passthrough
                },
            }
            if t_has_cache and _can_translate:
                # Translate Anthropic tools[].cache_control → DIAL
                # custom_fields.cache_breakpoint on the tools entry (per
                # DIAL prompt-caching docs example).
                out_entry["custom_fields"] = {"cache_breakpoint": {}}
                cache_metric["translated"] += 1
            schema_sha_out = _canonical_sha({
                "n": out_entry["function"]["name"],
                "d": out_entry["function"]["description"],
                "s": out_entry["function"]["parameters"],
            })
            kind, mcp_server = _classify_tool_name(name)
            tool_inventory.append({
                "name": name,
                "kind": kind,
                "mcp_server": mcp_server,
                "schema_sha_in": schema_sha_in,
                "schema_sha_out": schema_sha_out,
                "drift": schema_sha_in != schema_sha_out,
            })
            out_tools.append(out_entry)
        out["tools"] = out_tools
    cache_metric["tool_inventory"] = tool_inventory
    cache_metric["tool_inventory_hash"] = _canonical_sha(
        [(t["name"], t["schema_sha_in"]) for t in tool_inventory]
    ) if tool_inventory else None
    cache_metric["tools_native_count"] = sum(1 for t in tool_inventory if t["kind"] == "native")
    cache_metric["tools_mcp_count"] = sum(1 for t in tool_inventory if t["kind"] == "mcp")
    cache_metric["tools_other_count"] = sum(1 for t in tool_inventory if t["kind"] == "other")
    cache_metric["tools_drift_count"] = sum(1 for t in tool_inventory if t["drift"])
    cache_metric["mcp_servers_seen"] = sorted(set(
        t["mcp_server"] for t in tool_inventory if t["mcp_server"]
    ))

    # 4. tool_choice mapping
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        ttype = tc.get("type")
        if ttype == "auto":
            out["tool_choice"] = "auto"
        elif ttype == "any":
            out["tool_choice"] = "required"
        elif ttype == "tool":
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name", "")},
            }
        elif ttype == "none":
            out["tool_choice"] = "none"
    elif isinstance(tc, str) and tc in ("auto", "any", "none"):
        out["tool_choice"] = {"any": "required"}.get(tc, tc)

    # Strip request fields the target upstream is known to reject so the
    # request returns 200 with degraded capability rather than 422/400.
    # See `_PREFIX_INCAPABLE` above for the per-prefix list.
    stripped = _strip_unsupported_features_for_upstream(out, out.get("model", ""))
    cache_metric["features_stripped"] = stripped

    return out, cache_metric


# ---------------------------------------------------------------------------
# OpenAI → Anthropic response translation (non-streaming)
# ---------------------------------------------------------------------------


def openai_to_anthropic_response(
    upstream: dict[str, Any], requested_model: str
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build an Anthropic Messages response from a non-streaming OpenAI body.
    Returns (anthropic_body, tool_calls_by_name) — the second tuple element
    powers the dashboard's per-tool success-rate panel.
    """
    choice = (upstream.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason") or "stop"

    content_blocks: list[dict[str, Any]] = []
    tool_calls_by_name: dict[str, int] = {}
    text = msg.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                content_blocks.append({"type": "text", "text": part.get("text", "")})

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            tool_input = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            tool_input = {}
        raw_name = fn.get("name")
        tname = raw_name if isinstance(raw_name, str) else ""
        tool_calls_by_name[tname] = tool_calls_by_name.get(tname, 0) + 1
        raw_id = tc.get("id")
        tool_id = raw_id if isinstance(raw_id, str) and raw_id else (
            f"toolu_{uuid.uuid4().hex[:24]}"
        )
        content_blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": tname,
            "input": tool_input,
        })

    usage = upstream.get("usage") or {}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    anthropic_usage = {
        "input_tokens": max(0, int(usage.get("prompt_tokens") or 0) - int(cached)),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cache_read_input_tokens": int(cached),
        "cache_creation_input_tokens": 0,
    }

    return (
        {
            "id": upstream.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": upstream.get("model") or requested_model,
            "content": content_blocks,
            "stop_reason": STOP_REASON_MAP.get(finish, finish),
            "stop_sequence": None,
            "usage": anthropic_usage,
        },
        tool_calls_by_name,
    )


# ---------------------------------------------------------------------------
# OpenAI SSE → Anthropic SSE event stream
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _estimate_streaming_input_hint(body: dict[str, Any]) -> int:
    """Order-of-magnitude input-token hint for `message_start` before upstream usage.

    Streaming chunks usually carry ``usage`` only on the final event even with
    ``stream_options.include_usage``; Claude Code's context meter still benefits
    from a non-zero hint derived from the inbound Anthropic body.
    """
    parts: list[str] = []
    parts.append(json.dumps(body.get("messages", []), separators=(",", ":"), default=str))
    sys = body.get("system")
    if isinstance(sys, str):
        parts.append(sys)
    elif isinstance(sys, list):
        parts.append(json.dumps(sys, separators=(",", ":"), default=str))
    return max(0, len("".join(parts)) // 4)


async def stream_openai_to_anthropic(
    upstream_resp: aiohttp.ClientResponse,
    *,
    requested_model: str,
    response: web.StreamResponse,
    trace: dict[str, Any],
    estimated_input_tokens: int = 0,
    force_model_override: str | None = None,
) -> None:
    """Read OpenAI SSE chunks from `upstream_resp`, emit Anthropic events into
    `response`. Tracks event counts in `trace` for the success-criteria metric.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = False
    text_open = False
    text_index = 0
    # tool_calls indexed by their OpenAI index (typically 0..N); each maps to a
    # content_block index assigned in stream order so the indices stay stable
    # across deltas.
    tool_block_index_by_oai_index: dict[int, int] = {}
    open_tool_oai_indices: set[int] = set()
    tool_calls_by_name: dict[str, int] = {}  # streaming-path per-tool counter
    tool_stopped_oai_indices: set[int] = set()
    next_block_index = 0
    accumulated_usage: dict[str, int] = {}
    final_stop_reason = "end_turn"
    saw_finish_reason = False
    consumed = 0
    emitted = 0

    async def emit_bytes(b: bytes) -> None:
        nonlocal emitted
        emitted += 1
        await response.write(b)

    async def close_open_tool_blocks() -> None:
        nonlocal open_tool_oai_indices, tool_stopped_oai_indices
        for prev_oai in sorted(open_tool_oai_indices):
            await emit_bytes(_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": tool_block_index_by_oai_index[prev_oai],
            }))
            tool_stopped_oai_indices.add(prev_oai)
        open_tool_oai_indices.clear()

    async def process_data_line(line_str: str) -> None:
        nonlocal started, text_open, text_index, next_block_index
        nonlocal tool_block_index_by_oai_index, open_tool_oai_indices, tool_stopped_oai_indices
        nonlocal final_stop_reason, saw_finish_reason, accumulated_usage, consumed
        if line_str.startswith(":") or not line_str.startswith("data:"):
            return
        payload = line_str[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            ev = json.loads(payload)
        except (ValueError, TypeError):
            return
        consumed += 1

        if not started:
            started = True
            u0 = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
            real_in = 0
            if u0:
                cached0 = int((u0.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                prompt0 = int(u0.get("prompt_tokens") or 0)
                real_in = max(0, prompt0 - cached0)
            input_hint = max(estimated_input_tokens, real_in)
            # When the caller forces an override (alias mode),
            # use it regardless of what the upstream stream reports as `model`.
            # Otherwise prefer upstream's value, falling back to requested.
            start_model = (
                force_model_override
                if force_model_override is not None
                else (ev.get("model") or requested_model)
            )
            started_msg = {
                "type": "message_start",
                "message": {
                    "id": ev.get("id") or message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": start_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_hint,
                        "output_tokens": 0,
                        "cache_read_input_tokens": int(
                            (u0.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                        ),
                        "cache_creation_input_tokens": 0,
                    },
                },
            }
            await emit_bytes(_sse("message_start", started_msg))

        choice = (ev.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}

        # --- Text deltas ---
        text_piece = delta.get("content")
        if isinstance(text_piece, str) and text_piece:
            await close_open_tool_blocks()
            if not text_open:
                text_index = next_block_index
                next_block_index += 1
                await emit_bytes(_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                }))
                text_open = True
            await emit_bytes(_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": text_piece},
            }))

        # --- Tool call deltas ---
        tool_calls_delta = delta.get("tool_calls") or []
        if tool_calls_delta and text_open:
            await emit_bytes(_sse("content_block_stop", {
                "type": "content_block_stop", "index": text_index,
            }))
            text_open = False
        for tc in tool_calls_delta:
            oai_idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            if oai_idx not in tool_block_index_by_oai_index:
                # Anthropic requires each tool block stopped before the next starts.
                await close_open_tool_blocks()
                # First chunk for this tool call — emit content_block_start.
                block_idx = next_block_index
                next_block_index += 1
                tool_block_index_by_oai_index[oai_idx] = block_idx
                open_tool_oai_indices.add(oai_idx)
                raw_name = fn.get("name")
                tname = raw_name if isinstance(raw_name, str) else ""
                tool_calls_by_name[tname] = tool_calls_by_name.get(tname, 0) + 1
                raw_id = tc.get("id")
                tool_id = raw_id if isinstance(raw_id, str) and raw_id else (
                    f"toolu_{uuid.uuid4().hex[:24]}"
                )
                await emit_bytes(_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tname,
                        "input": {},
                    },
                }))
            args_delta = fn.get("arguments")
            if isinstance(args_delta, str) and args_delta:
                await emit_bytes(_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": tool_block_index_by_oai_index[oai_idx],
                    "delta": {"type": "input_json_delta", "partial_json": args_delta},
                }))

        # --- Final chunk: finish_reason + usage ---
        if choice.get("finish_reason"):
            saw_finish_reason = True
            final_stop_reason = STOP_REASON_MAP.get(
                choice["finish_reason"], choice["finish_reason"]
            )
        usage = ev.get("usage")
        if isinstance(usage, dict):
            accumulated_usage = usage

    buf = b""
    async for chunk in upstream_resp.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line_str = line.decode("utf-8", errors="replace").rstrip("\r")
            await process_data_line(line_str)
    if buf:
        await process_data_line(buf.decode("utf-8", errors="replace").rstrip("\r"))

    if not started:
        started_msg = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": requested_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
        await emit_bytes(_sse("message_start", started_msg))
        started = True

    stream_truncated = started and consumed > 0 and not saw_finish_reason
    trace["stream_truncated"] = stream_truncated
    if stream_truncated:
        await emit_bytes(_sse("error", {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "upstream closed the SSE stream before sending finish_reason",
            },
        }))
        final_stop_reason = "max_tokens"

    # Close any open blocks.
    if text_open:
        await emit_bytes(_sse("content_block_stop", {
            "type": "content_block_stop", "index": text_index,
        }))
    for oai_idx, block_idx in sorted(tool_block_index_by_oai_index.items()):
        if oai_idx in tool_stopped_oai_indices:
            continue
        await emit_bytes(_sse("content_block_stop", {
            "type": "content_block_stop", "index": block_idx,
        }))

    cached = ((accumulated_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    msg_delta_usage = {
        "input_tokens": max(0, int(accumulated_usage.get("prompt_tokens") or 0) - int(cached)),
        "output_tokens": int(accumulated_usage.get("completion_tokens") or 0),
        "cache_read_input_tokens": int(cached),
        "cache_creation_input_tokens": 0,
    }
    await emit_bytes(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
        "usage": msg_delta_usage,
    }))
    await emit_bytes(_sse("message_stop", {"type": "message_stop"}))

    trace["sse_events_consumed"] = consumed
    trace["sse_events_emitted"] = emitted
    trace["sse_event_emission_ratio"] = (emitted / consumed) if consumed else 0.0
    trace["final_stop_reason"] = final_stop_reason
    # Surface the same token-usage fields the non-streaming path emits so
    # CCPPM cost metrics (input/output/cache_read) work for streaming too.
    trace["input_tokens"] = int(msg_delta_usage.get("input_tokens") or 0)
    trace["output_tokens"] = int(msg_delta_usage.get("output_tokens") or 0)
    trace["cache_read_input_tokens"] = int(msg_delta_usage.get("cache_read_input_tokens") or 0)
    trace["cache_creation_input_tokens"] = int(msg_delta_usage.get("cache_creation_input_tokens") or 0)
    trace["tool_calls_by_name"] = tool_calls_by_name  # streaming-path counter


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


async def health(_r: web.Request) -> web.Response:
    return web.Response(text="ok")


async def count_tokens_stub(_r: web.Request) -> web.Response:
    return web.Response(
        status=501,
        body=json.dumps({
            "type": "error",
            "error": {
                "type": "not_implemented",
                "message": "count_tokens is not supported by the adapter.",
            },
        }).encode(),
        content_type="application/json",
        headers={"x-anthropic-dial-adapter-reason": "count_tokens_not_implemented"},
    )


# Models list — Claude Desktop's Dev-Mode "Gateway" connection test calls
# GET /v1/models on profile-apply and rejects the gateway if it 404s.
# We fetch ai-proxy's /openai/models, filter to anthropic.* deployments,
# and reshape into Anthropic Messages /v1/models response.
#
# Cached in-process for MODELS_TTL_SEC so the test isn't a stampede on the
# upstream. Falls back to a curated static list when the upstream is
# unreachable (off-VPN, 5xx, etc.) so the adapter still passes the Desktop
# connection probe.
#
# These strings are **OpenAI deployment ids** from the gateway's
# ``GET /openai/models`` listing (``anthropic.*``, ``qwen.*``, …). They are
# not the same namespace as DIAL Core's internal ``models`` config object
# (DIAL-side ids like ``claude-sonnet-4-5`` may alias to entirely different
# upstream deployments). Keeping a small offline pilot list here is
# intentional; the live path still prefers the upstream inventory when the
# gateway is reachable (see ``models()``).
MODELS_TTL_SEC = 300
_MODELS_FALLBACK = [
    "anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    # Surface non-Anthropic DIAL deployments in the
    # Claude Code `/model` picker so the operator can swap models without
    # spinning up a sibling adapter per provider. Pilot list — may be
    # trimmed once we learn which deployments survive the OpenAI-shape
    # translation end-to-end.
    "qwen.qwen3-coder-480b-a35b-v1:0",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "moonshotai.kimi-k2.5",
    "minimax.minimax-m2.5",
    "mistral.devstral-2-123b",
    "deepseek.v3.2",
    # Google Gemma 3 instruction-tuned.
    # 27B is the most capable; 12B exposed for low-latency pilots. PT
    # (pretrained / base) variants are not advertised — Claude Code's
    # instruction-following surface requires IT.
    "google.gemma-3-27b-it",
    "google.gemma-3-12b-it",
]

# Which deployment-id namespaces /v1/models advertises
# back to the Anthropic client. Comma-separated env override lets operators
# narrow (anthropic-only for a paranoid demo) or widen without a rebuild.
_DEFAULT_ADVERTISED_PREFIXES: tuple[str, ...] = (
    "anthropic.",
    "qwen.",
    "moonshotai.",
    "minimax.",
    "mistral.",
    "deepseek.",
    "google.",
)


def _parsed_advertised_prefixes() -> tuple[str, ...]:
    raw = os.environ.get(
        "ANTHROPIC_DIAL_ADVERTISE_PREFIXES",
        ",".join(_DEFAULT_ADVERTISED_PREFIXES),
    )
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    # Blank / comma-only env would otherwise filter *every* upstream id.
    return parts if parts else _DEFAULT_ADVERTISED_PREFIXES


_ADVERTISED_PREFIXES: tuple[str, ...] = _parsed_advertised_prefixes()


# Per-prefix capability gaps at DIAL. Each value lists Anthropic-shape request
# fields that the upstream gateway rejects for any deployment matching the
# prefix. The adapter strips these BEFORE forwarding, emits a structured
# `features_stripped` count on the request_in event so the AI-SDLC dashboard
# can quantify the gap, and lets the request through with degraded
# functionality rather than a hard 422/400.
#
# Observed upstream behaviour by deployment family (from live probes against
# the gateway). The strings below are upstream-error literals the adapter
# pattern-matches when deciding to strip a feature:
#
#   - deepseek.v3.2        -> "Tools are not supported" (HTTP 422)
#                          → "doesn't support the stopSequences field" (HTTP 400)
#   - google.gemma-3-*-it  → "doesn't support the stopSequences field" (HTTP 400);
#                            tools pass at the gateway but Gemma 3 ignores
#                            them silently and replies in text — strip so
#                            Claude Code stops waiting for tool_use blocks
#                            that will never come.
#
# Tools-incapable upstreams still serve the safety-classifier path (it does
# not need tools to return safe/risky), so stripping rather than blacklisting
# the deployment keeps the Haiku-slot picker mapping useful.
_PREFIX_INCAPABLE: dict[str, frozenset[str]] = {
    "deepseek.": frozenset({"tools", "tool_choice", "stop_sequences", "stop"}),
    "google.": frozenset({"tools", "tool_choice", "stop_sequences", "stop"}),
    # Qwen / Kimi / MiniMax on Bedrock support native tool_use (these are the
    # OSS bake-off winners precisely because of that), but their Converse API
    # rejects the ``stopSequences`` field that Claude Code unconditionally
    # sends on every auto-mode classifier request. Without stripping, the
    # classifier 400s every Bash/Skill safety check and fails closed — which
    # the Claude Code UI surfaces as the misleading "<model> is temporarily
    # unavailable" message. Strip only stop_sequences/stop so native tools
    # still flow. Empirically reproduced 2026-05-15 against
    # qwen.qwen3-coder-480b-a35b-v1:0 on DIAL/Bedrock.
    "qwen.": frozenset({"stop_sequences", "stop"}),
    "moonshotai.": frozenset({"stop_sequences", "stop"}),
    "minimax.": frozenset({"stop_sequences", "stop"}),
}


def _strip_unsupported_features_for_upstream(
    openai_body: dict[str, Any], target_model: str
) -> list[str]:
    """Remove fields that the target upstream's DIAL route rejects.

    Mutates ``openai_body`` in-place and returns the list of stripped field
    names (empty when the target is fully capable).
    """
    if not target_model:
        return []
    for prefix, incompat in _PREFIX_INCAPABLE.items():
        if target_model.startswith(prefix):
            stripped: list[str] = []
            # ``stop_sequences`` is Anthropic; ``stop`` is its OpenAI mirror —
            # ``anthropic_to_openai`` rewrites the former into the latter, so
            # we strip ``stop`` here (after translation) but list both in the
            # incapable set for documentation.
            for field in ("tools", "tool_choice", "stop"):
                if field in incompat and field in openai_body:
                    del openai_body[field]
                    stripped.append(field)
            return stripped
    return []


def _filter_model_ids_for_advertise(ids: Sequence[str]) -> list[str]:
    """Keep only deployment ids allowed by ``_ADVERTISED_PREFIXES``.

    Used for the static ``/v1/models`` fallback so it matches the live-upstream
    filter when ``ANTHROPIC_DIAL_ADVERTISE_PREFIXES`` is narrowed.

    If the operator's prefix list excludes every curated pilot id, fall back to
    ``anthropic.*`` entries only so the picker (and Claude Desktop's gateway
    probe) never goes completely empty.
    """
    kept = [mid for mid in ids if any(mid.startswith(p) for p in _ADVERTISED_PREFIXES)]
    if kept:
        return kept
    return [mid for mid in ids if mid.startswith("anthropic.")]


def _strip_model_display_tag(model: str) -> str:
    """Strip trailing ``[1m]``-style display tags some clients append to picker ids."""
    return re.sub(r"\[[^\]]+\]\s*$", "", model).strip()


def _normalize_requested_model(
    raw: str,
    models_cache: dict[str, Any],
    request_id: str,
    body: dict[str, Any],
) -> str:
    """Map a bare Claude Code model name to a DIAL deployment id.

    Claude Code (2.1+) defaults to short names like ``claude-sonnet-4-6``
    with no ``anthropic.`` prefix and sometimes a ``[1m]``-style display tag.
    This function strips the tag unconditionally, then—if the name still lacks
    the ``anthropic.`` namespace—tries three well-known transforms against the
    live models cache (or the static fallback when the cache is cold).

    The original *raw* name is preserved for ``model_normalized`` telemetry so
    the log shows exactly what the client sent.

    Args:
        raw:          The model string from ``body["model"]``.
        models_cache: ``request.app["models_cache"]`` (may be empty/missing).
        request_id:   Current request id forwarded to ``emit()``.
        body:         The parsed request body; updated in-place if normalised.

    Returns:
        The normalised deployment id (or *bare* if no match was found).
    """
    # Defensive type check: body["model"] should always be a string from a
    # well-behaved JSON client, but a malformed body could carry an int,
    # dict, or null. Return as-is so the existing _validate_upstream_model_id
    # below produces the canonical 400 instead of a regex crash here.
    if not isinstance(raw, str):
        return raw  # type: ignore[return-value]

    # Strip any client-side display tag like "[1m]" unconditionally so it
    # never reaches _validate_upstream_model_id regardless of which branch
    # below is taken (e.g. already-namespaced "anthropic.claude-sonnet-4-6[1m]"
    # would otherwise fail the deployment-id regex).
    bare = _strip_model_display_tag(raw)

    if bare.startswith("anthropic.") or "/" in bare:
        # Already namespaced or contains a path separator — skip candidate
        # expansion but still reflect tag-stripping back into body.
        if bare != raw:
            body["model"] = bare
        return bare

    # Build the set of valid deployment ids from the live cache (or the
    # static fallback if the cache hasn't been warmed yet).
    envelope = models_cache.get("envelope")
    if not isinstance(envelope, dict):
        envelope = {}
    known_ids: set[str] = {
        m.get("id")
        for m in (envelope.get("data") or [])
        if isinstance(m, dict) and m.get("id")
    }
    if not known_ids:
        known_ids = set(_MODELS_FALLBACK)

    # Claude Code's `/model` picker stores the user's pick in a stripped form
    # (e.g. "sonnet-4-5-20250929" — no "claude-" prefix). When the underlying
    # DIAL deployment id is "anthropic.claude-sonnet-4-5-20250929-v1:0", the
    # bare-only candidate set produces a 404. Try both with and without an
    # inserted "claude-" so either form survives. The original four
    # bare-prefix candidates remain first so existing inputs are unchanged.
    candidates: list[str] = [
        f"anthropic.{bare}",
        f"anthropic.{bare}-v1:0",
        f"anthropic.{bare}-v1",
    ]
    if not bare.startswith("claude-"):
        candidates.extend([
            f"anthropic.claude-{bare}",
            f"anthropic.claude-{bare}-v1:0",
            f"anthropic.claude-{bare}-v1",
        ])
    for candidate in candidates:
        if candidate in known_ids:
            emit("model_normalized", request_id=request_id,
                 from_model=raw, to_model=candidate)
            body["model"] = candidate
            return candidate

    # No match — return bare (tag-stripped) so validation has a clean string.
    if bare != raw:
        body["model"] = bare
    return bare


def _alias_lookup_candidate_keys(requested_model: str, raw_client_model: str) -> list[str]:
    """Keys to try against ``_ALIASES_MAP``, most specific first.

    Operators often configure short Claude Code ids (``claude-sonnet-4-6``) while
    :func:`_normalize_requested_model` rewrites the active deployment id to
    ``anthropic.*`` before alias lookup — so we must fall back to the raw client
    string (and tag-stripped raw) when the normalized id is not a map key.

    Non-string values are coerced to ``""`` so a malformed JSON ``model`` field
    cannot raise ``TypeError`` here before :func:`_validate_upstream_model_id`
    returns a 400 (see messages() type guard).
    """
    if not isinstance(requested_model, str):
        requested_model = ""
    if not isinstance(raw_client_model, str):
        raw_client_model = ""

    keys: list[str] = []
    seen: set[str] = set()

    def add(k: str) -> None:
        if not k or k in seen:
            return
        seen.add(k)
        keys.append(k)

    add(requested_model)
    add(raw_client_model)
    bare_tag = _strip_model_display_tag(raw_client_model)
    add(bare_tag)
    # Alias maps often key bare ``claude-*`` while normalization already
    # expanded to ``anthropic.claude-*``.
    if requested_model.startswith("anthropic."):
        add(requested_model.removeprefix("anthropic."))
    # Picker may send bare ``sonnet-4-5-20250929`` while alias maps use
    # ``claude-sonnet-4-5-20250929`` — mirror ``_normalize_requested_model``.
    if bare_tag and not bare_tag.startswith(("claude-", "anthropic.")):
        add(f"claude-{bare_tag}")
    return keys


def _select_alias_mapping(
    aliases_map: dict[str, str],
    requested_model: str,
    raw_client_model: str,
) -> tuple[str, str] | None:
    """Return ``(source_key, target)`` for the first matching alias key, else ``None``."""
    if not aliases_map:
        return None
    for key in _alias_lookup_candidate_keys(requested_model, raw_client_model):
        if key in aliases_map:
            return (key, aliases_map[key])
    return None


# Display-name prefix → human family label. Drives the `/model` picker copy
# for the non-Anthropic deployments we advertise.
_PROVIDER_LABELS: dict[str, str] = {
    "anthropic.": "Claude",
    "qwen.": "Qwen",
    "moonshotai.": "Kimi",
    "minimax.": "MiniMax",
    "mistral.": "Mistral",
    "deepseek.": "DeepSeek",
    "google.": "Gemma",
}


def _humanize_model_id(model_id: str) -> str:
    """anthropic.claude-opus-4-7 → "Claude Opus 4 7 (DIAL)".

    Non-Anthropic deployments get a sensible family label too, e.g.
    ``qwen.qwen3-coder-480b-a35b-v1:0`` → ``Qwen Qwen3 Coder 480b A35b (DIAL)``
    (each whitespace-delimited token uses :func:`str.capitalize`, so mixed
    alpha/digit segments like ``a35b`` become ``A35b``, not ``A35B``).
    """
    family = "Claude"
    base = model_id.removeprefix("anthropic.")
    for prefix, label in _PROVIDER_LABELS.items():
        if model_id.startswith(prefix):
            family = label
            base = model_id[len(prefix):]
            break
    # Strip the long Bedrock date+version tail when present so the dropdown
    # stays readable: "claude-haiku-4-5-20251001-v1:0" → "claude-haiku-4-5".
    short = re.sub(r"-\d{8}(-v\d+:\d+)?(-with-thinking)?$", "", base)
    # Also strip a trailing bare "-vN:N" tail (no date), e.g. Qwen ids.
    short = re.sub(r"-v\d+:\d+$", "", short)
    short = short.removeprefix("claude-").replace("-v", " v").replace("-", " ")
    pretty = " ".join(p.capitalize() if p[:1].isalpha() else p for p in short.split())
    suffix = " (with thinking)" if base.endswith("-with-thinking") else ""
    return f"{family} {pretty}{suffix} (DIAL)"


def _owned_by_for_model_id(mid: str) -> str:
    """Stable ``owned_by`` for /v1/models entries (Anthropic + advertised OSS)."""
    if mid.startswith("anthropic."):
        return "anthropic-via-dial"
    if mid.startswith("qwen."):
        return "qwen-via-dial"
    if mid.startswith("moonshotai."):
        return "moonshot-via-dial"
    if mid.startswith("minimax."):
        return "minimax-via-dial"
    if mid.startswith("mistral."):
        return "mistral-via-dial"
    if mid.startswith("deepseek."):
        return "deepseek-via-dial"
    if mid.startswith("google."):
        return "google-via-dial"
    return "via-dial"


def _split_openai_prompt_usage(usage: dict[str, Any]) -> tuple[int, int, int]:
    """Mirror Anthropic-path accounting: non-cached prompt, completion, cache-read.

    Summing input_tokens + cache_read across responses must not double-count
    cached prompt tokens present in ``prompt_tokens``.
    """
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    inp = max(0, prompt - cached)
    out = int(usage.get("completion_tokens") or 0)
    return inp, out, cached


def _anthropic_models_envelope(ids: list[str]) -> dict:
    """Hybrid envelope satisfying both Anthropic /v1/models clients
    (Claude Code, Claude Desktop) and OpenAI /v1/models clients
    (Cursor, Zed, Continue, Aider, Goose). Each entry carries both
    `type: model` (Anthropic) and `object: model` (OpenAI); envelope
    carries both `object: list` (OpenAI) and `has_more`/`first_id`/
    `last_id` (Anthropic). Both clients are lenient about extra fields.
    """
    data = []
    for mid in ids:
        entry = {
            "id": mid,
            "type": "model",
            "object": "model",
            "display_name": _humanize_model_id(mid),
            "created_at": "2025-01-01T00:00:00Z",
            "created": 1735689600,  # 2025-01-01T00:00:00Z epoch — OpenAI shape
            "owned_by": _owned_by_for_model_id(mid),
        }
        # Honest routing in the picker. Match the same
        # short-key / namespaced-key logic as messages() so ANTHROPIC_DIAL_ALIASES_JSON
        # entries keyed by bare Claude Code ids still surface the "(→ target)" suffix.
        alias_match = _select_alias_mapping(
            _ALIASES_MAP,
            mid,
            mid.removeprefix("anthropic."),
        )
        if alias_match:
            _, alias_target = alias_match
            entry["display_name"] = (
                f"{entry['display_name']} (→ {alias_target})"
            )
        data.append(entry)
    return {
        "object": "list",        # OpenAI envelope
        "data": data,
        "has_more": False,       # Anthropic envelope
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


async def models(request: web.Request) -> web.Response:
    app = request.app
    cache = app.setdefault("models_cache", {"at": 0.0, "envelope": None})
    now = time.monotonic()
    if cache["envelope"] is not None and (now - cache["at"]) < MODELS_TTL_SEC:
        return web.json_response(cache["envelope"])

    # None → upstream fetch/parse failed (use static fallback). A successful
    # probe that returns zero anthropic.* deployments is an empty list, not a
    # fallback trigger.
    live_ids: list[str] | None = None
    if PROJECT_KEY:
        session: aiohttp.ClientSession = app["client_session"]
        try:
            async with session.get(
                f"{UPSTREAM}/openai/models",
                headers={"Api-Key": PROJECT_KEY},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as up:
                if up.status == 200:
                    try:
                        raw = await up.text()
                    except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as e:
                        emit("error", reason="models_upstream_read", message=str(e))
                    else:
                        try:
                            body = json.loads(raw) if raw.strip() else {}
                        except (ValueError, TypeError):
                            emit(
                                "error",
                                reason="models_upstream_non_json",
                                snippet=(raw[:256] if raw else ""),
                            )
                        else:
                            if isinstance(body, dict):
                                data = body.get("data")
                                if isinstance(data, list):
                                    live_ids = [
                                        str(mid)
                                        for m in data
                                        if isinstance(m, dict)
                                        for mid in [m.get("id")]
                                        if mid is not None
                                        and any(
                                            str(mid).startswith(p)
                                            for p in _ADVERTISED_PREFIXES
                                        )
                                    ]
                else:
                    emit("error", reason="models_upstream_status", status=up.status)
        except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as e:
            emit("error", reason="models_upstream_connect", message=str(e))

    if live_ids is None:
        ids = _filter_model_ids_for_advertise(_MODELS_FALLBACK)
        emit("models_fallback", count=len(ids), reason="upstream_unavailable")
    elif not live_ids:
        # Upstream returned 200 but every id was filtered out by
        # ``_ADVERTISED_PREFIXES`` (mis-set env, empty allow-list, etc.). Treat
        # like a failed probe so Claude Desktop's gateway check still sees ids.
        ids = _filter_model_ids_for_advertise(_MODELS_FALLBACK)
        emit("models_fallback", count=len(ids), reason="upstream_filtered_empty")
    else:
        emit("models_live", count=len(live_ids))
        ids = live_ids

    envelope = _anthropic_models_envelope(sorted(set(ids)))
    cache["at"] = now
    cache["envelope"] = envelope
    return web.json_response(envelope)


async def on_startup(app: web.Application) -> None:
    global _UPSTREAM_CACHE_CONTROL_SUPPORT, _ALIASES_MAP, _PRICE_TABLE
    _setup_logging()
    _setup_shadow_logging()  # shadow log handler (idempotent)
    app["client_session"] = aiohttp.ClientSession()
    app["background_tasks"] = set()
    # Parse alias map at startup. Stored in module-level
    # _ALIASES_MAP so per-request handler can look up without re-parsing.
    _ALIASES_MAP = _load_aliases_map()
    raw_price = os.environ.get("ANTHROPIC_DIAL_PRICE_TABLE_JSON", "")
    _pt, pt_err = load_price_table_env(raw_price)
    if pt_err:
        emit(
            "price_table_load_failed",
            error=pt_err,
            raw_len=len(raw_price),
        )
    _PRICE_TABLE = _pt
    emit("startup", upstream=UPSTREAM, listen_port=LISTEN_PORT,
         api_version=DIAL_API_VERSION, project_key_present=bool(PROJECT_KEY),
         shadow_model=SHADOW_MODEL or None,
         aliases_count=len(_ALIASES_MAP),
         alias_keys=sorted(_ALIASES_MAP) if _ALIASES_MAP else [],
         price_table_models=len(_PRICE_TABLE))
    # Probe the upstream's cache_control acceptance once at startup.
    # Result drives `cache_control_strategy` reporting for the process lifetime.
    _UPSTREAM_CACHE_CONTROL_SUPPORT = await _probe_upstream_cache_control_support(
        app["client_session"]
    )
    emit("upstream_cache_probe",
         result=_UPSTREAM_CACHE_CONTROL_SUPPORT,
         probe_model=_CACHE_PROBE_MODEL)


async def on_cleanup(app: web.Application) -> None:
    tasks = tuple(app.get("background_tasks", set()))
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await app["client_session"].close()
    emit("shutdown")


def _track_background_task(
    app: web.Application,
    coro: Coroutine[Any, Any, None],
) -> asyncio.Task[None]:
    tasks = app["background_tasks"]
    task = asyncio.create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


async def messages(request: web.Request) -> Union[web.Response, web.StreamResponse]:
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
    started_at = time.monotonic()
    if not PROJECT_KEY:
        emit(
            "error",
            request_id=request_id,
            reason="missing_project_key",
            **_gflog_metrics_tags(request, None),
        )
        return web.json_response(
            {"type": "error", "error": {"type": "authentication_error",
                                         "message": "PROJECT_KEY env not set"}},
            status=500,
        )

    raw = await request.read()
    try:
        body = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error",
                                         "message": "body is not valid JSON"}},
            status=400,
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error",
                                         "message": "body must be a JSON object"}},
            status=400,
        )

    model_field = body.get("model")
    if model_field is None or model_field == "":
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error",
                                         "message": "model is required"}},
            status=400,
        )
    if not isinstance(model_field, str):
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error",
                                         "message": "model must be a string"}},
            status=400,
        )
    requested_model = model_field

    # Capture the raw client name before normalization. For alias-disguised
    # responses we surface the same logical id the client sent (e.g.
    # ``claude-sonnet-4-6``), not the post-normalization ``anthropic.*``
    # deployment id — with display tags like ``[1m]`` stripped so the API
    # response matches the non-alias path.
    raw_client_model = requested_model

    # Normalise short Claude Code model names → DIAL deployment ids.
    # Tag-stripping ("[1m]" etc.) is also applied unconditionally here so the
    # model string passed to _validate_upstream_model_id is always clean.
    requested_model = _normalize_requested_model(
        requested_model,
        request.app.get("models_cache") or {},
        request_id,
        body,
    )

    # Apply alias map. Prefer the normalized deployment id,
    # then the raw client model (short Claude Code ids / display-tagged names),
    # so ANTHROPIC_DIAL_ALIASES_JSON keys like ``claude-sonnet-4-6`` still match
    # after ``_normalize_requested_model`` expands to ``anthropic.*``.
    client_view_model = requested_model
    alias_active = False
    alias_match = _select_alias_mapping(_ALIASES_MAP, requested_model, raw_client_model)
    if alias_match:
        alias_source_key, alias_target = alias_match
        emit(
            "model_aliased",
            request_id=request_id,
            from_model=alias_source_key,
            to_model=alias_target,
            **_gflog_metrics_tags(request, alias_target),
        )
        requested_model = alias_target
        body["model"] = alias_target
        # Mirror the non-alias path: strip ``[1m]``-style display tags so the
        # response ``model`` field never leaks terminal formatting artifacts.
        client_view_model = _strip_model_display_tag(raw_client_model)
        alias_active = True

    _mt = _gflog_metrics_tags(request, str(requested_model))

    try:
        _validate_upstream_model_id(requested_model)
    except TranslationError as e:
        emit("error", request_id=request_id, reason="invalid_model", message=str(e), **_mt)
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
            status=e.status,
        )

    try:
        openai_body, cache_metric = anthropic_to_openai(body)
    except TranslationError as e:
        emit("error", request_id=request_id, reason="translation_failed", message=str(e), **_mt)
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
            status=e.status,
        )

    stream = bool(body.get("stream"))
    target = (
        f"{UPSTREAM}/openai/deployments/"
        f"{requested_model}/chat/completions?api-version={DIAL_API_VERSION}"
    )
    upstream_headers = {
        "Api-Key": PROJECT_KEY,
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    # Preserve trace propagation headers if the client sent them.
    for h in ("Traceparent", "X-Conversation-Id", "X-DIAL-CHAT-ID"):
        if h in request.headers:
            upstream_headers[h] = request.headers[h]
    # Route cache-aware requests toward DIAL upstream nodes holding prompt cache
    # (paired with translated custom_fields.cache_breakpoint bodies).
    if cache_metric.get("translated"):
        upstream_headers["X-CACHE-POLICY"] = "cache-priority"

    emit("request_in", request_id=request_id, model=requested_model, stream=stream,
         message_count=len(body.get("messages") or []),
         tool_count=len(body.get("tools") or []),
         cache_control_seen=cache_metric.get("seen", 0),
         cache_control_translated=cache_metric.get("translated", 0),
         cache_control_strategy=cache_metric.get("strategy"),
         # Per-upstream capability gap (deepseek.* etc.)
         features_stripped=cache_metric.get("features_stripped") or [],
         # schema integrity canary
         tool_inventory_hash=cache_metric.get("tool_inventory_hash"),
         tool_inventory=cache_metric.get("tool_inventory") or [],
         tools_drift_count=cache_metric.get("tools_drift_count", 0),
         # per-class tool composition
         tools_native_count=cache_metric.get("tools_native_count", 0),
         tools_mcp_count=cache_metric.get("tools_mcp_count", 0),
         tools_other_count=cache_metric.get("tools_other_count", 0),
         mcp_servers_seen=cache_metric.get("mcp_servers_seen") or [],
         **_mt)

    session: aiohttp.ClientSession = request.app["client_session"]
    try:
        upstream_resp = await session.post(
            target,
            headers=upstream_headers,
            data=json.dumps(openai_body).encode(),
            timeout=aiohttp.ClientTimeout(total=600),
        )
    except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as e:
        emit("error", request_id=request_id, reason="upstream_connect", message=str(e), **_mt)
        return web.json_response(
            {"type": "error", "error": {"type": "api_error", "message": f"upstream: {e}"}},
            status=502,
        )

    if upstream_resp.status >= 400:
        err_body = await upstream_resp.read()
        emit("error", request_id=request_id, reason="upstream_status",
             status=upstream_resp.status, body_snippet=err_body[:512].decode("utf-8", "replace"),
             **_mt)
        upstream_resp.release()
        if err_body and err_body.startswith(b"{"):
            try:
                upstream_error: Any = json.loads(err_body)
            except (ValueError, TypeError):
                upstream_error = err_body.decode("utf-8", "replace")
        else:
            upstream_error = err_body.decode("utf-8", "replace") if err_body else ""
        return web.json_response(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"upstream returned {upstream_resp.status}",
                    "upstream": upstream_error,
                },
            },
            status=upstream_resp.status,
        )

    if not stream:
        try:
            upstream_text = await upstream_resp.text()
        finally:
            upstream_resp.release()
        try:
            upstream_json = json.loads(upstream_text) if upstream_text else {}
        except (ValueError, TypeError):
            emit("error", request_id=request_id, reason="upstream_non_json",
                 snippet=upstream_text[:512], **_mt)
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "upstream returned 200 with a non-JSON body",
                        "upstream": upstream_text[:4096],
                    },
                },
                status=502,
            )
        if not isinstance(upstream_json, dict):
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "upstream JSON was not an object",
                    },
                },
                status=502,
            )
        anthropic_body, tool_calls_by_name = openai_to_anthropic_response(
            upstream_json, client_view_model
        )
        # When alias was applied, force the response.model to the client-view
        # name regardless of what upstream returned. openai_to_anthropic_response
        # normally preserves the upstream model (e.g. for cross-region
        # inference profiles like "global.anthropic.*"), but in alias mode
        # the upstream id is the swap target and would leak the disguise.
        if alias_active:
            anthropic_body["model"] = client_view_model
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        _usage = anthropic_body["usage"]
        emit("response_out", request_id=request_id, stream=False, status=200,
             elapsed_ms=elapsed_ms, stop_reason=anthropic_body["stop_reason"],
             input_tokens=_usage["input_tokens"],
             output_tokens=_usage["output_tokens"],
             cache_read_input_tokens=_usage["cache_read_input_tokens"],
             cache_creation_input_tokens=int(_usage.get("cache_creation_input_tokens") or 0),
             tool_calls_by_name=tool_calls_by_name,
             shadow_dispatched=bool(SHADOW_MODEL),
             **_cost_usd_estimate_kwargs(
                 str(requested_model),
                 input_tokens=int(_usage["input_tokens"] or 0),
                 output_tokens=int(_usage["output_tokens"] or 0),
                 cache_read_input_tokens=int(_usage.get("cache_read_input_tokens") or 0),
                 cache_creation_input_tokens=int(_usage.get("cache_creation_input_tokens") or 0),
             ),
             **_mt)
        # Fire-and-forget shadow dispatch. Does not block the client.
        if SHADOW_MODEL:
            primary_summary = _summarize_response_shape(upstream_json)
            _track_background_task(request.app, _run_shadow_dispatch(
                session, openai_body, primary_summary, request_id, requested_model,
            ))
        return web.json_response(anthropic_body)

    # Streaming path.
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await response.prepare(request)
    trace: dict[str, Any] = {}
    stream_exc: Exception | None = None
    try:
        await stream_openai_to_anthropic(
            upstream_resp,
            requested_model=client_view_model,
            response=response,
            trace=trace,
            estimated_input_tokens=_estimate_streaming_input_hint(body),
            # When alias is active, force the streamed message_start.model
            # to the client-view name (overrides whatever the upstream
            # alias target reported).
            force_model_override=client_view_model if alias_active else None,
        )
    except Exception as e:
        stream_exc = e
        emit(
            "error",
            request_id=request_id,
            reason="streaming_failed",
            message=f"{type(e).__name__}: {e}",
            **_mt,
        )
        trace["stream_exception"] = f"{type(e).__name__}"
        try:
            await response.write(_sse("error", {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "adapter failed while translating the upstream SSE stream",
                },
            }))
        except Exception:
            pass
    finally:
        upstream_resp.release()
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    emit(
        "response_out",
        request_id=request_id,
        stream=True,
        status=200,
        elapsed_ms=elapsed_ms,
        stream_transform_failed=bool(stream_exc),
        **trace,
        **_cost_usd_estimate_kwargs(
            str(requested_model),
            input_tokens=int(trace.get("input_tokens") or 0),
            output_tokens=int(trace.get("output_tokens") or 0),
            cache_read_input_tokens=int(trace.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(trace.get("cache_creation_input_tokens") or 0),
        ),
        **_mt,
    )
    try:
        await response.write_eof()
    except Exception:
        pass
    return response


# ---------------------------------------------------------------------------
# Shadow-mode helpers.
# ---------------------------------------------------------------------------


def _summarize_response_shape(body: dict) -> dict:
    """Extract just the structural fingerprint of a response — what the
    shadow-diff cares about. Avoids retaining content verbatim in logs.
    """
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content")
    text_len = len(text) if isinstance(text, str) else (
        sum(len(p.get("text", "")) for p in text if isinstance(p, dict))
        if isinstance(text, list) else 0
    )
    tcbn: dict[str, int] = {}
    arg_shape_sha: dict[str, str] = {}
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        tcbn[name] = tcbn.get(name, 0) + 1
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {"_parse_error": True}
        arg_shape_sha[name] = _canonical_sha(sorted(args.keys()) if isinstance(args, dict) else [])
    return {
        "stop_reason": choice.get("finish_reason"),
        "text_length": text_len,
        "tool_calls_by_name": tcbn,
        "tool_arg_shape_sha": arg_shape_sha,
        "usage_completion_tokens": int((body.get("usage") or {}).get("completion_tokens") or 0),
    }


def _diff_categories(primary: dict, shadow: dict) -> list[str]:
    """Categorise the structural delta between primary and shadow."""
    cats: list[str] = []
    if primary.get("stop_reason") != shadow.get("stop_reason"):
        cats.append("stop_reason_mismatch")
    if bool(primary.get("tool_calls_by_name")) != bool(shadow.get("tool_calls_by_name")):
        cats.append("tool_presence_mismatch")
    elif primary.get("tool_calls_by_name") != shadow.get("tool_calls_by_name"):
        cats.append("tool_set_mismatch")
    else:
        # Same tool names — check the argument-shape SHA agreement.
        for name in primary.get("tool_calls_by_name", {}):
            ps = (primary.get("tool_arg_shape_sha") or {}).get(name)
            ss = (shadow.get("tool_arg_shape_sha") or {}).get(name)
            if ps and ss and ps != ss:
                cats.append("tool_args_shape_mismatch")
                break
    p_len = primary.get("text_length") or 0
    s_len = shadow.get("text_length") or 0
    if p_len and s_len:
        ratio = s_len / p_len
        if ratio < 0.5 or ratio > 2.0:
            cats.append("text_length_drift")
    elif (p_len > 0) != (s_len > 0):
        cats.append("text_presence_mismatch")
    if not cats:
        cats.append("structurally_aligned")
    return cats


async def _run_shadow_dispatch(
    session: aiohttp.ClientSession,
    base_openai_body: dict,
    primary_summary: dict,
    request_id: str,
    primary_model: str,
) -> None:
    """Fire-and-forget shadow call. Posts the same prompt to SHADOW_MODEL,
    structurally diffs against primary, writes one shadow.log line.
    """
    if not (SHADOW_MODEL and PROJECT_KEY):
        return
    try:
        _validate_upstream_model_id(SHADOW_MODEL)
    except TranslationError as e:
        shadow_emit(
            "shadow_error",
            request_id=request_id,
            shadow_model=SHADOW_MODEL,
            primary_model=primary_model,
            reason="invalid_shadow_model",
            message=str(e),
        )
        return
    shadow_body = copy.deepcopy(base_openai_body)
    shadow_body["model"] = SHADOW_MODEL
    # Shadow is never streamed — single-shot, cheap to compare.
    shadow_body.pop("stream", None)
    shadow_body.pop("stream_options", None)
    target = (
        f"{UPSTREAM}/openai/deployments/"
        f"{SHADOW_MODEL}/chat/completions?api-version={DIAL_API_VERSION}"
    )
    started = time.monotonic()
    elapsed_ms = 0
    try:
        async with session.post(
            target,
            headers={"Api-Key": PROJECT_KEY, "Content-Type": "application/json"},
            data=json.dumps(shadow_body).encode(),
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status >= 400:
                err = (await resp.read())[:512].decode("utf-8", "replace")
                elapsed_ms = int((time.monotonic() - started) * 1000)
                shadow_emit("shadow_error",
                            request_id=request_id,
                            shadow_model=SHADOW_MODEL,
                            primary_model=primary_model,
                            status=resp.status,
                            elapsed_ms=elapsed_ms,
                            body_snippet=err)
                return
            shadow_json = await resp.json()
            elapsed_ms = int((time.monotonic() - started) * 1000)
    except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError, ValueError) as e:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        shadow_emit("shadow_error",
                    request_id=request_id,
                    shadow_model=SHADOW_MODEL,
                    primary_model=primary_model,
                    elapsed_ms=elapsed_ms,
                    reason="connect_or_parse",
                    message=str(e))
        return
    if not isinstance(shadow_json, dict):
        shadow_emit(
            "shadow_error",
            request_id=request_id,
            shadow_model=SHADOW_MODEL,
            primary_model=primary_model,
            reason="shadow_non_object_json",
            message=type(shadow_json).__name__,
        )
        return
    try:
        shadow_summary = _summarize_response_shape(shadow_json)
    except Exception as e:
        shadow_emit(
            "shadow_error",
            request_id=request_id,
            shadow_model=SHADOW_MODEL,
            primary_model=primary_model,
            reason="shadow_summarize_failed",
            message=str(e),
        )
        return
    cats = _diff_categories(primary_summary, shadow_summary)
    shadow_emit("shadow_response",
                request_id=request_id,
                primary_model=primary_model,
                shadow_model=SHADOW_MODEL,
                elapsed_ms=elapsed_ms,
                primary=primary_summary,
                shadow=shadow_summary,
                diff_categories=cats)


# ---------------------------------------------------------------------------
# OpenAI-shape sibling routes for multi-client use (Cursor, Zed,
# Continue, Aider, Goose, JetBrains AI Assistant). These editors don't allow
# overriding the Anthropic base URL but DO allow overriding the OpenAI base
# URL — pointing them at `http://127.0.0.1:8092/v1` gives the gateway a
# multi-client governance story without touching the existing Anthropic
# ingress.
#
# Implementation: passthrough. The upstream already speaks Azure-OpenAI
# Chat Completions, so an OpenAI client's body is almost wire-compatible
# with `/openai/deployments/{model}/chat/completions`. We:
#   - Look up `body["model"]`, validate as a deployment id
#   - Build the upstream URL + Api-Key header
#   - Forward verbatim; stream SSE through with the same keepalive stripping
#   - Reuse the existing tool-inventory tagging from `cache_metric` shape
# ---------------------------------------------------------------------------


def _openai_tool_inventory(body: dict[str, Any]) -> dict[str, Any]:
    """Mirror `anthropic_to_openai`'s tool-schema-hash and MCP-classification tagging for an OpenAI-shape
    body, since the upstream-call code path doesn't go through it.
    """
    tools_in = body.get("tools") or []
    tool_inventory: list[dict[str, Any]] = []
    for t in tools_in:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict):
            continue
        name_raw = fn.get("name")
        name = name_raw if isinstance(name_raw, str) else ""
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
        description = fn.get("description", "")
        sha = _canonical_sha({"n": name, "d": description, "s": schema})
        kind, mcp_server = _classify_tool_name(name)
        tool_inventory.append({
            "name": name,
            "kind": kind,
            "mcp_server": mcp_server,
            "schema_sha_in": sha,
            "schema_sha_out": sha,  # passthrough; no rename
            "drift": False,
        })
    return {
        "tool_inventory": tool_inventory,
        "tool_inventory_hash": _canonical_sha(
            [(t["name"], t["schema_sha_in"]) for t in tool_inventory]
        ) if tool_inventory else None,
        "tools_native_count": sum(1 for t in tool_inventory if t["kind"] == "native"),
        "tools_mcp_count": sum(1 for t in tool_inventory if t["kind"] == "mcp"),
        "tools_other_count": sum(1 for t in tool_inventory if t["kind"] == "other"),
        "tools_drift_count": 0,
        "mcp_servers_seen": sorted(set(
            t["mcp_server"] for t in tool_inventory if t["mcp_server"]
        )),
    }


async def chat_completions(request: web.Request) -> Union[web.Response, web.StreamResponse]:
    """OpenAI Chat Completions ingress. Translates only the headers + URL;
    passes the body through to the upstream and streams the response back
    in OpenAI's `chat.completion.chunk` shape (which is what the upstream
    natively emits).
    """
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
    started_at = time.monotonic()
    if not PROJECT_KEY:
        emit("error", request_id=request_id, reason="missing_project_key",
             client_protocol="openai", **_gflog_metrics_tags(request, None))
        return web.json_response(
            {"error": {"type": "authentication_error", "message": "PROJECT_KEY env not set"}},
            status=500,
        )

    raw = await request.read()
    try:
        body = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return web.json_response(
            {"error": {"type": "invalid_request_error",
                       "message": "body is not valid JSON"}},
            status=400,
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"error": {"type": "invalid_request_error",
                       "message": "body must be a JSON object"}},
            status=400,
        )

    requested_model = body.get("model") or ""
    if not requested_model:
        return web.json_response(
            {"error": {"type": "invalid_request_error", "message": "model is required"}},
            status=400,
        )
    try:
        _validate_upstream_model_id(requested_model)
    except TranslationError as e:
        return web.json_response(
            {"error": {"type": "invalid_request_error", "message": str(e)}},
            status=e.status,
        )

    _o_mt = _gflog_metrics_tags(request, str(requested_model))

    stream = bool(body.get("stream"))
    if stream:
        # Many OpenAI gateways require this for token usage on streams.
        body.setdefault("stream_options", {})
        if isinstance(body["stream_options"], dict):
            body["stream_options"].setdefault("include_usage", True)

    target = (
        f"{UPSTREAM}/openai/deployments/"
        f"{requested_model}/chat/completions?api-version={DIAL_API_VERSION}"
    )
    upstream_headers = {
        "Api-Key": PROJECT_KEY,
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    for h in ("Traceparent", "X-Conversation-Id", "X-DIAL-CHAT-ID"):
        if h in request.headers:
            upstream_headers[h] = request.headers[h]

    inv = _openai_tool_inventory(body)
    emit("request_in", request_id=request_id, model=requested_model, stream=stream,
         client_protocol="openai",
         message_count=len(body.get("messages") or []),
         tool_count=len(body.get("tools") or []),
         cache_control_seen=0,
         cache_control_strategy=_cache_control_strategy(),
         tool_inventory_hash=inv["tool_inventory_hash"],
         tool_inventory=inv["tool_inventory"],
         tools_drift_count=inv["tools_drift_count"],
         tools_native_count=inv["tools_native_count"],
         tools_mcp_count=inv["tools_mcp_count"],
         tools_other_count=inv["tools_other_count"],
         mcp_servers_seen=inv["mcp_servers_seen"],
         **_o_mt)

    session: aiohttp.ClientSession = request.app["client_session"]
    try:
        upstream_resp = await session.post(
            target,
            headers=upstream_headers,
            data=json.dumps(body).encode(),
            timeout=aiohttp.ClientTimeout(total=600),
        )
    except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as e:
        emit("error", request_id=request_id, reason="upstream_connect",
             client_protocol="openai", message=str(e), **_o_mt)
        return web.json_response(
            {"error": {"type": "api_error", "message": f"upstream: {e}"}},
            status=502,
        )

    if upstream_resp.status >= 400:
        err_body = await upstream_resp.read()
        emit("error", request_id=request_id, reason="upstream_status",
             client_protocol="openai",
             status=upstream_resp.status,
             body_snippet=err_body[:512].decode("utf-8", "replace"),
             **_o_mt)
        upstream_resp.release()
        return web.Response(
            status=upstream_resp.status,
            body=err_body,
            content_type="application/json",
        )

    if not stream:
        try:
            upstream_text = await upstream_resp.text()
        finally:
            upstream_resp.release()
        try:
            upstream_json = json.loads(upstream_text) if upstream_text else {}
        except (ValueError, TypeError):
            emit("error", request_id=request_id, reason="upstream_non_json",
                 client_protocol="openai",
                 snippet=upstream_text[:512] if upstream_text else "",
                 **_o_mt)
            return web.json_response(
                {
                    "error": {
                        "type": "api_error",
                        "message": "upstream returned 200 with a non-JSON body",
                        "upstream": (upstream_text[:4096] if upstream_text else ""),
                    },
                },
                status=502,
            )
        if not isinstance(upstream_json, dict):
            return web.json_response(
                {
                    "error": {
                        "type": "api_error",
                        "message": "upstream JSON was not an object",
                    },
                },
                status=502,
            )
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        # Per-tool counter from the upstream response shape directly.
        tcbn: dict[str, int] = {}
        for tc in ((upstream_json.get("choices") or [{}])[0].get("message") or {}).get(
            "tool_calls"
        ) or []:
            tname = (tc.get("function") or {}).get("name", "")
            if tname:
                tcbn[tname] = tcbn.get(tname, 0) + 1
        usage = upstream_json.get("usage") or {}
        inp_tok, out_tok, cache_rd = _split_openai_prompt_usage(
            usage if isinstance(usage, dict) else {},
        )
        emit("response_out", request_id=request_id, stream=False, status=200,
             client_protocol="openai", elapsed_ms=elapsed_ms,
             stop_reason=(upstream_json.get("choices") or [{}])[0].get("finish_reason"),
             input_tokens=inp_tok,
             output_tokens=out_tok,
             cache_read_input_tokens=cache_rd,
             cache_creation_input_tokens=0,
             tool_calls_by_name=tcbn,
             shadow_dispatched=bool(SHADOW_MODEL),
             **_cost_usd_estimate_kwargs(
                 str(requested_model),
                 input_tokens=inp_tok,
                 output_tokens=out_tok,
                 cache_read_input_tokens=cache_rd,
                 cache_creation_input_tokens=0,
             ),
             **_o_mt)
        # Shadow dispatch on the OpenAI sibling route too.
        if SHADOW_MODEL:
            _track_background_task(request.app, _run_shadow_dispatch(
                session, body, _summarize_response_shape(upstream_json),
                request_id, requested_model,
            ))
        return web.json_response(upstream_json)

    # Streaming — passthrough with SSE comment-line stripping (mirrors the
    # standard outbound-adapter pattern).
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await response.prepare(request)
    consumed = 0
    emitted = 0
    accumulated_usage: dict[str, Any] = {}
    tcbn: dict[str, int] = {}
    saw_tool_idx: set[int] = set()
    stream_read_exc: BaseException | None = None
    try:
        buf = b""
        async for chunk in upstream_resp.content.iter_any():
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_str = line.decode("utf-8", errors="replace").rstrip("\r")
                # SSE comment lines are skipped; blank lines are event terminators
                # and MUST be forwarded so clients can delimit events.
                if line_str.startswith(":"):
                    continue
                if line_str == "":
                    await response.write(b"\n")
                    continue
                if line_str.startswith("data:"):
                    payload = line_str[len("data:"):].strip()
                    if payload and payload != "[DONE]":
                        try:
                            ev = json.loads(payload)
                            consumed += 1
                            for tc in (
                                (ev.get("choices") or [{}])[0].get("delta") or {}
                            ).get("tool_calls") or []:
                                idx = tc.get("index", 0)
                                if idx not in saw_tool_idx:
                                    saw_tool_idx.add(idx)
                                    tname = (tc.get("function") or {}).get("name", "")
                                    if tname:
                                        tcbn[tname] = tcbn.get(tname, 0) + 1
                            u = ev.get("usage")
                            if isinstance(u, dict):
                                accumulated_usage = u
                        except (ValueError, TypeError):
                            pass
                await response.write(line + b"\n")
                emitted += 1
        if buf:
            await response.write(buf)
            emitted += 1
    except (
        aiohttp.ClientPayloadError,
        asyncio.TimeoutError,
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        OSError,
    ) as e:
        stream_read_exc = e
        emit(
            "error",
            request_id=request_id,
            reason="upstream_stream_read",
            client_protocol="openai",
            message=f"{type(e).__name__}: {e}",
            **_o_mt,
        )
        try:
            error_frame = json.dumps(
                {
                    "error": {
                        "type": "api_error",
                        "message": "upstream stream read failed",
                    }
                },
                separators=(",", ":"),
            ).encode()
            await response.write(b"data: " + error_frame + b"\n\n")
            emitted += 1
        except Exception:
            pass
    finally:
        upstream_resp.release()
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    inp_tok, out_tok, cache_rd = _split_openai_prompt_usage(accumulated_usage)
    emit("response_out", request_id=request_id, stream=True, status=200,
         client_protocol="openai", elapsed_ms=elapsed_ms,
         sse_events_consumed=consumed, sse_events_emitted=emitted,
         sse_event_emission_ratio=(emitted / consumed) if consumed else 0.0,
         input_tokens=inp_tok,
         output_tokens=out_tok,
         cache_read_input_tokens=cache_rd,
         cache_creation_input_tokens=0,
         tool_calls_by_name=tcbn,
         upstream_stream_aborted=bool(stream_read_exc),
         stream_error_type=(type(stream_read_exc).__name__ if stream_read_exc else None),
         **_cost_usd_estimate_kwargs(
             str(requested_model),
             input_tokens=inp_tok,
             output_tokens=out_tok,
             cache_read_input_tokens=cache_rd,
             cache_creation_input_tokens=0,
         ),
         **_o_mt)
    try:
        await response.write_eof()
    except Exception:
        pass
    return response


def build_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/v1/messages/count_tokens", count_tokens_stub)
    # OpenAI-shape sibling routes for Cursor, Zed, Continue, Aider,
    # Goose, JetBrains AI. Multi-client governance story.
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


if __name__ == "__main__":
    _setup_logging()
    emit("boot", upstream=UPSTREAM, listen_port=LISTEN_PORT)
    web.run_app(build_app(), host=os.environ.get("BIND", "127.0.0.1"), port=LISTEN_PORT)
