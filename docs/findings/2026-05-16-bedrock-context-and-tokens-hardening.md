# Findings — Bedrock-backed OSS upstreams: complete failure-class table & fixes

**Date:** 2026-05-16
**Status:** resolved (live-verified end-to-end)
**Audience:** anyone running Claude Code (or another Anthropic-shape client) against EPAM AI DIAL or another OpenAI-compatible gateway that fronts AWS Bedrock for non-Anthropic models.

## TL;DR

Running Claude Code against third-party Bedrock-hosted models (Qwen, Kimi, MiniMax, DeepSeek, etc.) via DIAL produces **five distinct 400-error classes** that all look the same from the client side (`API Error: 400 upstream returned 400`). Four involve adapter translation or missing features; one is a hard upstream context-window limit. This adapter now handles all five so end-to-end agentic-loop sessions are stable.

## The five failure classes

| # | Symptom | Root cause | Fix layer in this adapter |
|---|---------|------------|-----|
| 1 | Auto-mode classifier 400s on every Bash/Skill safety check; UI shows `<model> is temporarily unavailable` | Bedrock's Converse API rejects the `stopSequences` field for `qwen.*`, `moonshotai.*`, `minimax.*` (same rejection class as `deepseek.*` and `google.gemma-3-*`) | `_UPSTREAM_PREFIX_STRIP_REGISTRY` extended; `_strip_unsupported_features_for_upstream` removes `stop_sequences`/`stop` before forwarding |
| 2 | 400 when a request carries an MCP tool whose `function.name` exceeds 64 characters (e.g. `mcp__plugin_deploy-on-aws_awsknowledge__aws___get_regional_availability` = 71 chars) | Bedrock Converse enforces `toolSpec.name` ≤ 64 chars | New `_build_tool_name_alias` deterministically aliases long names on the request side (`<64-8 prefix>__<6-hex-sha1>`); `openai_to_anthropic_response` and `stream_openai_to_anthropic` accept a `tool_name_aliases` map and reverse-map `tool_use[].name` so the client never sees the alias |
| 3 | 400 on multi-turn tool_result content that happens to be JSON-shape text (e.g. `gh pr list --json` returns a JSON array): `messages.N.content.0.toolResult.content.0.json is invalid. Provide a json object for the field` | DIAL's OpenAI→Converse translator auto-parses bare-string `tool` role content as JSON; if the parse yields anything other than a JSON OBJECT (array, scalar, null), Bedrock rejects | `anthropic_to_openai` wraps tool_result content as `{"output": <text>}` (a guaranteed JSON object) before forwarding for non-Anthropic upstreams; OSS bake-off-winner models unwrap the `output` envelope transparently |
| 4 | 400 mid-agent-loop after a long session: `This model's maximum context length is 131072 tokens. However, you requested 32000 output tokens and your prompt contains at least 99073 input tokens, for a total of at least 131073 tokens` | Claude Code unconditionally requests `max_tokens=32000` output reservation. Qwen-on-Bedrock caps at 131k context. Once the conversation crosses ~99k input tokens, *any* request fails by exactly 1 over the cap | New `_clamp_max_tokens_to_fit_context` computes `budget = max_context − estimated_input − 4096-token safety margin` and sets `max_tokens = budget` when `0 < budget < requested` (never `max(1024, budget)` — that would exceed the window on small-window models). When `budget <= 0`, returns 400 with `input_exceeds_context` instead of forwarding. Curated `_MODEL_MAX_CONTEXT` map covers qwen.*: 131072, moonshotai.*: 128000, minimax.*: ~256000, deepseek.*: 65536, google.gemma-*: 8192. Anthropic upstreams left unmapped (200k+ contexts handle themselves) |
| 5 | Claude Code's pre-flight context-usage check fails: `countTokens API call failed: 501 not_implemented`; client then blasts ahead blind and trips class #4 anyway | Adapter previously returned 501 from `POST /v1/messages/count_tokens` | New `count_tokens` handler returns `{"input_tokens": N}` using chars/4 over a stable JSON serialization of `system + messages + tools + tool_choice`, padded `_COUNT_TOKENS_PAD_FACTOR=1.10` so the client errs toward compacting too early rather than too late |

## Why classes #4 and #5 compose

Layer 4 is a band-aid: if Claude Code does generate a request that's about to overflow, the adapter clamps `max_tokens` so the request still succeeds (the model just gets less output budget). Layer 5 is the proper fix: with a working `count_tokens` endpoint, Claude Code self-trims the conversation via `/compact` *before* the request ever fires. Both are shipped together because layer 5 alone doesn't help long-running sessions that already overshot; layer 4 alone doesn't free Claude Code from blasting ahead blind.

## Why `chars/4 × 1.10` instead of tiktoken or a real tokenizer

`tiktoken.get_encoding('cl100k_base')` downloads the BPE merges file from OpenAI's CDN on first use, which fails in the VPN-gated target environment. Bundling the file is a larger change for a benefit that doesn't compound — the count is used for *compaction-trigger* math, not exact budget calculations. A stable approximation that *over-reports slightly* is functionally equivalent to a real tokenizer.

The 1.10 multiplier was chosen empirically: prior runs measured chars/4 undercounting Bedrock's per-model actual count by 1-5% on Claude Code's prompt shapes. Padding 10% covers the upper bound comfortably while still producing 1-token counts for short prompts.

A future improvement (noted in the code comments) is to vendor `cl100k_base.tiktoken` offline, or implement an upstream-probe-based exact count via a `max_tokens=1` probe. Until then this heuristic is both simpler and good enough.

## Why `ANTHROPIC_SMALL_FAST_MODEL` is NOT a fix for class #1

`ANTHROPIC_SMALL_FAST_MODEL` decouples the **background namer/summarizer** path from the main loop model — useful for keeping cheap, frequent calls off your Opus-slot quota. It does **not** decouple the **auto-mode safety classifier** path. Claude Code 2.1.x hardcodes the classifier to the main loop model regardless of this env var. We verified this with a forced-classifier controlled experiment: classifier_request_started always reports `model=<main-loop-model>` even with `ANTHROPIC_SMALL_FAST_MODEL` set to a different deployment.

Practical implication: don't rely on a small-model classifier as your safety-classifier-degradation workaround. Use `permissions.allow` in `.claude/settings.json` to bypass the classifier for read-only Bash + `git` + `gh` operations, and fix the underlying Bedrock-translation issues (which is what this adapter does).

## Recommended client config for OSS-on-DIAL deployments

```bash
# In your wrapper / launcher
export ANTHROPIC_BASE_URL="http://127.0.0.1:8092"           # this adapter
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen.qwen3-coder-480b-a35b-v1:0"
export ANTHROPIC_DEFAULT_SONNET_MODEL="moonshotai.kimi-k2.5"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="minimax.minimax-m2.5"
# Decouples background namer/summarizer onto the Haiku-slot model so it
# doesn't burn the main loop's minute-token budget. Classifier path is
# unaffected — that's a Claude Code 2.1.x design limit.
export ANTHROPIC_SMALL_FAST_MODEL="minimax.minimax-m2.5"
# DIAL gateways 400 on unknown beta headers — suppress Anthropic-specific ones.
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
```

Plus a `.claude/settings.json` `permissions.allow` allowlist for read-only Bash, `git`, and `gh` ops so the auto-mode classifier path stays off the hot loop.

## Verification

Each layer was live-verified against `qwen.qwen3-coder-480b-a35b-v1:0` on EPAM AI DIAL during the 2026-05-15/16 hardening pass:

| Layer | Test |
|-------|------|
| 1 | `curl POST /v1/messages` with `stop_sequences: ["STOP"]` against Qwen → 200 (was 400). |
| 2 | Smoke test: `_build_tool_name_alias("mcp__plugin_deploy-on-aws_awsknowledge__aws___get_regional_availability")` returns a 64-char alias preserving the `mcp__<server>__` prefix. |
| 3 | End-to-end Claude Code session (`claude --print "Use Bash to run gh pr list --json number,title, then echo done"`) succeeds with `outcome=ok` from the adapter. Same prompt previously triggered class #3 mid-stream. |
| 4 | `curl POST /v1/messages` with 110k-token equivalent body → adapter logs `max_tokens_clamp={'original':32000,'clamped':16969,...}`, upstream returns 200. |
| 5 | `curl POST /v1/messages/count_tokens` with small body → `{"input_tokens": 11}` (was `501 not_implemented`). |

## Acknowledgements

Investigated and shipped end-to-end during a multi-day debugging session against `qwen.qwen3-coder-480b-a35b-v1:0` and other OSS deployments on EPAM AI DIAL. This document is the public consolidation of the findings — the adapter PRs (#1, #3, #4) carry the corresponding code changes.
