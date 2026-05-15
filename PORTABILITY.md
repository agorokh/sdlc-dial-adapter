# Portability, running `sdlc-dial-adapter` on another machine

Short engineer note for anyone who saw the research report or the repo and
wants to run the adapter locally.

## What it is

A single Python file, [`app.py`](app.py), that accepts Anthropic-shape
`POST /v1/messages` requests (the request shape Claude Code emits) and
translates them to an OpenAI-style `POST /openai/deployments/{model}/chat/completions`
upstream. Streaming, tool calls, and prompt caching translate too. One
runtime dependency: `aiohttp`. No database, no message queue, no auth
service.

## What you need

- A DIAL API key with access to your project (default upstream is
  `ai-proxy.lab.epam.com`; any OpenAI-compatible chat-completions
  endpoint works if you set `UPSTREAM_BASE`)
- Either Docker (recommended) or Python 3.12+
- Claude Code 2.1+ on your `$PATH` as `claude`

## Run it

```bash
# Docker, preferred
docker build -t sdlc-dial-adapter:local .
docker run --rm -d --name sdlc-dial-adapter \
  -e PROJECT_KEY="$YOUR_DIAL_API_KEY" \
  -e UPSTREAM_BASE="https://ai-proxy.lab.epam.com" \
  -e BIND=0.0.0.0 \
  -p 127.0.0.1:8092:8092 \
  sdlc-dial-adapter:local

# or Python directly (binds to 127.0.0.1 by default)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PROJECT_KEY="$YOUR_DIAL_API_KEY" python3 app.py
```

Note: when running under Docker, the `BIND=0.0.0.0` override is required so
the container's port forward reaches the listener inside. Running on the
host directly, the default `BIND=127.0.0.1` is the right choice. Do not
bind to a routable interface without a reverse proxy in front: the adapter
substitutes its own `PROJECT_KEY` upstream and does not validate any
caller-supplied auth header.

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

Then run `claude_dial` instead of `claude`. Same harness, traffic now
goes through the configured upstream on your project key. To target an
OSS model brokered by the gateway, swap the picker-slot env values for
any deployment id from `GET /openai/deployments` (e.g.
`qwen.qwen3-coder-480b-a35b-v1:0`).

## What it does, wire-level summary

`anthropic_to_openai` and `openai_to_anthropic_response` in `app.py`
translate request and response bodies in both directions:
`tool_use` <-> `tool_calls`, `stop_reason` <-> `finish_reason`, Anthropic
content blocks <-> OpenAI messages, SSE stream <-> Anthropic SSE.
`cache_control` markers pass through on Anthropic upstreams; they are
stripped for non-Anthropic upstreams (gateway-side gap, not a model
property). Per-request log fields land at the path given by
`ANTHROPIC_DIAL_ADAPTER_LOG` (default `/var/log/anthropic-dial-adapter/adapter.log`
inside the container; falls back to stderr otherwise), one JSON line
per `request_in` / `response_out` / `error` event.

## What this repo doesn't ship

This repo is the minimal subset needed to run the adapter standalone.
The observability sidecars (Influx, Vector, Grafana, the metrics
exporter) and the larger evaluation tooling live in the parent
research sandbox and are not part of this artefact. The adapter
emits structured JSON logs on its own, so any line-oriented log
shipper will work without those sidecars.

## Common issues

- **`error` event with `reason: missing_project_key`.** The adapter
  booted with an empty `PROJECT_KEY`. Set the env var and restart.
- **`502 No route` from the upstream.** The gateway does not know
  the deployment id you sent. Check available ids with
  `GET /openai/deployments`.
- **Long `tool_use` calls time out.** Streaming responses use a
  shared `ClientSession`; the default total timeout is generous,
  but a stalled upstream will still hold a request open. Watch the
  `error` events for `upstream_stream_read` or
  `streaming_failed` reasons; both carry a `request_id` you can
  quote.

Each `error` event in the log carries a `reason` field and a
`request_id`. Quote both when reporting an issue.
