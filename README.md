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

## What this isn't

- Not a proxy. There is no `api.anthropic.com` fallback. Setting
  `ANTHROPIC_BASE_URL` to this adapter routes every request through the
  configured `UPSTREAM_BASE`.
- Not a billing system. The optional `cost_usd_estimate` field is
  computed from a configurable price table; treat it as a sanity-check
  number, not an invoice.
- Not a multi-tenant gateway. One process per project key. Run multiple
  instances behind a reverse proxy if you need tenant isolation.

## Status

Proof of concept extracted from a larger internal evaluation
(192 trials of Claude Code 2.1 across eight upstream models routed
through DIAL). The translation core is stable; the operational
sidecars (observability, telemetry exporters) live in the parent
sandbox and are out of scope for this repo.

## License

Apache 2.0, matching EPAM AI DIAL itself. See [LICENSE](LICENSE).
