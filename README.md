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
through DIAL). The translation core is stable.

The repo also ships optional reference observability artifacts:
Grafana dashboards under [`observability/`](observability/) and an
InfluxDB exporter under [`ccppm/`](ccppm/) — so adopters who want a
working metrics-and-dashboards pipeline can stand one up in minutes.
Adopters using their own observability stack should read
[`observability/EVENT_SCHEMA.md`](observability/EVENT_SCHEMA.md) —
the vendor-neutral contract for JSON events written to
`ANTHROPIC_DIAL_ADAPTER_LOG` (or stderr when that path is unwritable).

## Code layout

`app.py` is a single Python module organized by clearly-labeled
section dividers. The most-touched code paths:

| Section | Lines | What lives here |
|---|---:|---|
| Anthropic → OpenAI request translation | ~250–800 | `anthropic_to_openai()`. Where Bedrock-quirk workarounds apply on the request side (stop_sequences strip via `_strip_unsupported_features_for_upstream`, 64-char tool name aliasing via `_alias_long_tool_names`, `max_tokens` clamp via `_clamp_max_tokens_to_fit_context`, `{"output": ...}` tool_result wrap). |
| OpenAI → Anthropic response translation | ~810–890 | `openai_to_anthropic_response()`. Reverse-maps aliased tool names back so the client never sees them. |
| OpenAI SSE → Anthropic SSE | ~890–1180 | `stream_openai_to_anthropic()`. Streaming bridge. |
| HTTP plumbing | ~1180–2410 | `/health`, `/v1/models`, `/v1/messages`, `/v1/messages/count_tokens`. The `count_tokens` handler at line ~1239 implements the heuristic that tells Claude Code when to auto-compact. |
| Shadow-mode helpers | ~2410–2570 | Optional parallel-dispatch mode for comparison testing. |
| OpenAI-shape sibling routes | ~2570–end | `/v1/chat/completions` passthrough for editors that override the OpenAI base URL (Cursor, Zed, etc.). |

The request/response translators (`anthropic_to_openai`,
`openai_to_anthropic_response`) and Bedrock workaround helpers are
pure and covered under `tests/`. The streaming bridge
(`stream_openai_to_anthropic`) is not in that unit suite.
See [`docs/findings/`](docs/findings) for the engineering write-ups
that explain each Bedrock workaround's failure mode and fix.

## Development

```bash
# Set up
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest

# Run the test suite
python -m pytest tests/ -v
```

CI runs `pytest` plus an `ast.parse` sweep on every Python file across
Python 3.11 and 3.12. See the [CI workflow](.github/workflows/ci.yml).

## License

Apache 2.0, matching EPAM AI DIAL itself. See [LICENSE](LICENSE).
