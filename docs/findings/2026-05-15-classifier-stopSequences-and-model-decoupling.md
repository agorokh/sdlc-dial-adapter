# Findings — Classifier brown-outs on third-party Bedrock upstreams + model-decoupling limits

**Date:** 2026-05-15
**Status:** resolved-partial — adapter fix shipped; one Claude Code design limit documented

## Symptom

Claude Code 2.1.142 running against an OSS Bedrock-backed model on DIAL (Qwen, Kimi, MiniMax) shows the auto-mode classifier failing every Bash/Skill call with:

```
<model> is temporarily unavailable, so auto mode cannot determine the safety of Bash right now.
```

That message is misleading — it implies a transient gateway outage. The debug log (`~/.claude/debug/*.txt`) reveals a deterministic 400 every single time:

```
classifier_request_started ... model=qwen.qwen3-coder-480b-a35b-v1:0 stage=xml_s1
classifier_request_finished outcome=error errorKind=Error:400
"This model doesn't support the stopSequences field. Remove stopSequences and try again."
Auto mode classifier unavailable, denying with retry guidance (fail closed)
```

## Root cause

Claude Code sends `stopSequences` on every classifier request. The AWS Bedrock Converse API rejects this field for several third-party model families — most notably `qwen.*`, `moonshotai.*`, and `minimax.*` — the same failure mode this adapter already handles for `deepseek.*` and `google.gemma-3-*`.

## Fix (in this commit)

Extended `_PREFIX_INCAPABLE` in `app.py` to add `qwen.`, `moonshotai.`, `minimax.` with `{stop_sequences, stop}` stripped on outbound requests. These three families support native tool_use, so `tools` and `tool_choice` are left intact.

After the adapter restart, the same classifier_request returns `outcome=ok` in ~1200ms.

## Secondary finding — Claude Code model decoupling limits

While debugging, also tested the two-model decoupling pattern (analogous to cloud Claude Code's Sonnet/Opus main + Haiku classifier+namer split). Controlled experiment used `ANTHROPIC_DEFAULT_OPUS_MODEL=qwen...` and `ANTHROPIC_SMALL_FAST_MODEL=minimax...`:

| Path | Model actually used | Decoupled from main? |
|------|---------------------|----------------------|
| Main loop | qwen.qwen3-coder-480b | n/a |
| Background namer / summarizer | minimax.minimax-m2.5 | ✅ **YES** (17 parallel MiniMax calls vs 17 Qwen-main calls observed in adapter request_in log) |
| Auto-mode safety classifier | qwen.qwen3-coder-480b | ❌ **NO** — Claude Code 2.1.142 hardcodes the classifier path to the main-loop model; `ANTHROPIC_SMALL_FAST_MODEL` is ignored for this path |

So **`ANTHROPIC_SMALL_FAST_MODEL` does decouple the namer/summarizer path but not the auto-mode classifier path.** The cloud Claude Code pattern of Haiku-as-classifier is not (yet?) available against DIAL via this env var. Until a future Claude Code release exposes a classifier-model knob, the practical mitigation for hot classifier load is `permissions.allow` in `.claude/settings.json` — pattern-matched commands skip the classifier entirely.

## Recommended client config for OSS-on-DIAL deployments

```bash
# In your wrapper / launcher
export ANTHROPIC_BASE_URL="http://127.0.0.1:8092"            # this adapter
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen.qwen3-coder-480b-a35b-v1:0"
export ANTHROPIC_DEFAULT_SONNET_MODEL="moonshotai.kimi-k2.5"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="minimax.minimax-m2.5"
# Decouples namer/summarizer (the only useful split today). The classifier
# stays on the main model regardless of this var.
export ANTHROPIC_SMALL_FAST_MODEL="minimax.minimax-m2.5"
# DIAL gateways 400 on unknown beta headers
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
```

Plus a `.claude/settings.json` `permissions.allow` allowlist for read-only Bash, `git`, and `gh` ops so the classifier path stays out of the critical loop.

## Follow-ups (not in this PR)

- **64-char tool-name aliasing.** Bedrock Converse rejects `toolSpec.name` > 64 chars. MCP tool names from third-party plugins can easily exceed this (`mcp__plugin_deploy-on-aws_awsknowledge__aws___get_regional_availability` is 71 chars). The dial-sandbox internal fork has this fix shipped (`agorokh/dial-sandbox@0267363`); cherry-pick is queued for this repo separately.
- **`autoMode.model` settings key.** If/when Claude Code exposes a way to route the classifier to a different model, validate the cloud-Claude-Code-style decoupling.
