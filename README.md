# sdlc-dial-adapter

A small translation adapter that lets [Anthropic Claude Code](https://github.com/anthropics/claude-code)
and other Anthropic Messages API clients run against
[EPAM AI DIAL](https://github.com/epam/ai-dial) or any other gateway that
speaks the OpenAI chat-completions shape.

Part of EPAM's DIAL ecosystem of SDLC productivity experiments. This repo
is one of the export paths from a larger internal sandbox: when an
experiment proves it should be reusable, we extract the minimum subset
needed to run it standalone and publish here.

## What it does

`POST /v1/messages` in, `POST /openai/deployments/{model}/chat/completions`
out, in both directions, including:

- `tool_use` ⇄ OpenAI `tool_calls` translation
- `stop_reason` ⇄ `finish_reason` mapping
- Anthropic content blocks ⇄ OpenAI messages
- SSE streaming bridge in both directions
- `cache_control` passthrough on Anthropic upstreams; stripped on others
  (gateway gap, not a model gap)
- Per-request structured JSON log line with token counts, tool inventory,
  cache strategy, latency, and an optional Bedrock-priced cost estimate

One dependency: `aiohttp`. No database. No auth service. No background
processes.

## Quick start

```bash
# Build and run
docker build -t sdlc-dial-adapter:local .
docker run --rm -d --name sdlc-dial-adapter \
  -e PROJECT_KEY="$YOUR_DIAL_API_KEY" \
  -e UPSTREAM_BASE="https://ai-proxy.lab.epam.com" \
  -e BIND=0.0.0.0 \
  -p 127.0.0.1:8092:8092 \
  sdlc-dial-adapter:local

# Smoke check
curl -sS http://127.0.0.1:8092/health
# -> ok
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8092
export ANTHROPIC_AUTH_TOKEN=placeholder-not-validated-on-loopback
export ANTHROPIC_DEFAULT_OPUS_MODEL=anthropic.claude-opus-4-7
export ANTHROPIC_DEFAULT_SONNET_MODEL=anthropic.claude-sonnet-4-6
export ANTHROPIC_DEFAULT_HAIKU_MODEL=anthropic.claude-haiku-4-5-20251001-v1:0
claude
```

Full instructions, env reference, wire-level translation notes, and
limitations are in [PORTABILITY.md](PORTABILITY.md).

## What you need

| Requirement | Why |
|---|---|
| A DIAL API key with project access | Sent as `Api-Key:` header on every upstream call |
| Docker or Python 3.12+ | Two equivalent ways to run `app.py` |
| Claude Code 2.1+ on `$PATH` as `claude` | The Anthropic-shape client the adapter targets |

DIAL keys for EPAM colleagues come through the standard enterprise
subscription. For external users, point `UPSTREAM_BASE` at any OpenAI
chat-completions endpoint you have credentials for.

## Configuration reference

All configuration is via environment variables. Defaults work for a
single-user loopback deployment against EPAM DIAL.

| Variable | Default | Purpose |
|---|---|---|
| `PROJECT_KEY` | _(unset; required)_ | Upstream API key. Sent as `Api-Key:` on every forwarded request. |
| `UPSTREAM_BASE` | `https://ai-proxy.lab.epam.com` | Base URL of the OpenAI-compatible gateway. |
| `DIAL_API_VERSION` | `2024-02-01` | Appended as `?api-version=` query string on each upstream call. |
| `BIND` | `127.0.0.1` | Listen address. Set to `0.0.0.0` inside Docker so the host port-forward reaches the listener. Do not bind to a routable interface without a reverse proxy in front. |
| `LISTEN_PORT` | `8092` | TCP port. |
| `ANTHROPIC_DIAL_ADAPTER_LOG` | `/var/log/anthropic-dial-adapter/adapter.log` | Path for the structured JSON log. Falls back to stderr if the directory is unwritable. |
| `ANTHROPIC_DIAL_PRICE_TABLE_JSON` | _(empty)_ | Optional operator price table. When set, each `response_out` event carries a `cost_usd_estimate` field. |
| `ANTHROPIC_DIAL_ALIASES_JSON` | _(empty)_ | Optional model alias map. Rewrites the `model` field in requests to a different upstream deployment id. |
| `ANTHROPIC_DIAL_SHADOW_MODEL` | _(empty)_ | Optional shadow-dispatch target. When set, every primary response triggers a second upstream call for comparison; the shadow response is written to a separate log and never returned to the client. Doubles upstream load and cost. |
| `ANTHROPIC_DIAL_CACHE_PROBE_MODEL` | _(auto)_ | Override for the model used at startup to detect cache_control support upstream. |

## What this isn't

- Not a proxy. There is no `api.anthropic.com` fallback. Setting
  `ANTHROPIC_BASE_URL` to this adapter routes every request through the
  configured `UPSTREAM_BASE`.
- Not a billing system. The optional `cost_usd_estimate` field is
  computed from a configurable price table; treat it as a sanity-check
  number, not an invoice.
- Not a multi-tenant gateway. One process per project key. The
  adapter substitutes its own `PROJECT_KEY` upstream and does not
  validate any caller-supplied auth header. Bind to loopback only or
  put a reverse proxy with auth in front before exposing it.

## Status

Proof of concept extracted from a larger internal evaluation
(192 trials of Claude Code 2.1 across eight upstream models routed
through DIAL). The translation core is stable; the operational
sidecars (observability, telemetry exporters) live in the parent
sandbox and are out of scope for this repo.

## License

Apache 2.0, matching EPAM AI DIAL itself. See [LICENSE](LICENSE).
