# Portability, running `anthropic-dial-adapter` on another machine

Short note for EPAM colleagues who saw the research report and asked how to
reuse the adapter on their own machine.

## What it is

A single Python file, [`app.py`](app.py), that accepts Claude Code's
Anthropic-style `POST /v1/messages` requests and translates them to
EPAM DIAL's OpenAI-shape `POST /openai/deployments/{model}/chat/completions`.
Streaming, tool calls, and prompt caching translate too. One dependency:
`aiohttp`. No database, no message queue, no auth service.

## What you need

- A DIAL API key with access to your project on `ai-proxy.lab.epam.com`
- Either Docker (recommended) or Python 3.12+
- Claude Code 2.1+ on your `$PATH` as `claude`

## Run it

```bash
# Docker, preferred
docker build -t anthropic-dial-adapter:local .
docker run --rm -d --name anthropic-dial-adapter \
  -e PROJECT_KEY="$DIAL_API_KEY_PROJECT" \
  -e UPSTREAM_BASE="https://ai-proxy.lab.epam.com" \
  -p 127.0.0.1:8092:8092 \
  anthropic-dial-adapter:local

# or Python directly
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PROJECT_KEY="$DIAL_API_KEY_PROJECT" python3 app.py
```

Smoke check: `curl -sS http://127.0.0.1:8092/health` returns `ok`.

## Point Claude Code at it

Save as `~/.local/bin/claude_dial` and `chmod +x`:

```bash
#!/usr/bin/env bash
exec env \
  CLAUDE_CONFIG_DIR="$HOME/.claude_dial" \
  ANTHROPIC_BASE_URL="http://127.0.0.1:8092" \
  ANTHROPIC_AUTH_TOKEN="placeholder-not-validated-on-loopback" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="anthropic.claude-opus-4-7" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="anthropic.claude-sonnet-4-6" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="anthropic.claude-haiku-4-5-20251001-v1:0" \
  CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
  claude "$@"
```

Then run `claude_dial` instead of `claude`. Same harness, traffic now goes
through DIAL on your enterprise project key. To target an OSS model behind
DIAL, swap the picker-slot env values for any deployment id from
`GET /openai/deployments` (e.g. `qwen.qwen3-coder-480b-a35b-v1:0`).

## What it does, wire-level summary

`anthropic_to_openai` and `openai_to_anthropic_response` in `app.py`
translate request and response bodies in both directions:
`tool_use` ⇄ `tool_calls`, `stop_reason` ⇄ `finish_reason`, Anthropic
content blocks ⇄ OpenAI messages, SSE stream ⇄ Anthropic SSE.
`cache_control` markers pass through on Anthropic upstreams; they are
stripped for non-Anthropic upstreams (DIAL gateway gap, not a model
property). Per-request log fields land in
`/var/log/anthropic-dial-adapter/adapter.log` as one JSON line each.

## What this doesn't include

The InfluxDB / Vector / Grafana sidecars from the in-repo Compose stack
are independent of the bridge. They turn the adapter log into measurements
and dashboards; useful if you want the telemetry analysis from the
report, not required to operate Claude Code through DIAL.

## Anything else

The longer in-repo notes ([`README.md`](README.md)) cover Doppler-based
secret handling and the optional observability sidecars. The research
report at `docs/research/2026-05-14_management-summary/THE_FULL_STORY.html`
covers the motivation and the 192-trial evaluation. Wire-level questions
or new error codes: each `error` event in the adapter log carries a
`reason` field and a `request_id` you can quote.
