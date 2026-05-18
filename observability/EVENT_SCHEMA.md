# Event schema

The adapter emits structured JSON events to stdout. Each line is one event. This document is the **contract**: every field listed here is supported across releases; anything not listed is internal and may change.

If you're building your own OTEL/Prometheus/Datadog pipeline against this adapter, use the field list below as your starting point. The reference InfluxDB exporter at `ccppm/claude_code_events_exporter.py` shows one mapping; your pipeline can be different.

## Envelope (every event)

Each line emitted to stdout has this skeleton:

```json
{
  "@timestamp": "2026-05-16T12:34:56-0700",
  "adapter": "anthropic_dial_adapter",
  "event": "<event_name>",
  ...
}
```

Beyond `@timestamp`, `adapter`, and `event`, each event has its own field set. Below.

## Lifecycle events

### `startup`
Emitted once on adapter boot.
- `upstream` (str): the OpenAI-compatible gateway URL the adapter forwards to.
- `listen_port` (int): port the adapter binds.
- `api_version` (str): Anthropic API version header sent to clients.
- `project_key_present` (bool): whether `PROJECT_KEY` is set (auth presence; no value logged).
- `shadow_model` (str | null): non-null when shadow mirroring is configured.
- `aliases_count` (int): number of `ANTHROPIC_DIAL_ALIASES_JSON` entries.
- `alias_keys` (list[str]): the aliased client model names (NOT their upstream targets).
- `price_table_models` (int): count of entries in the optional price table.

### `shutdown`
Emitted once on adapter shutdown. No fields beyond envelope.

### `upstream_cache_probe`
Emitted after startup once the cache-control compatibility probe completes.
- `result` (`"supported"` | `"unsupported"` | `"probe_failed"`).
- `probe_model` (str): which model id was used for the probe.

## Per-request events

These all share `request_id` (str, unique per request) and `client_name` (str, parsed from User-Agent), plus client/family classification tags.

### `model_normalized` *(when client sent a short model name like `claude-sonnet-4-6` and adapter mapped it to a DIAL deployment id)*
- `request_id` (str)
- `raw` (str): what the client sent.
- `normalized` (str): the DIAL deployment id used upstream.
- `strategy` (str): which normalization rule fired.

### `request_in`
Emitted after the request body is parsed and translated but before the upstream POST.

| Field | Type | Meaning |
|---|---|---|
| `request_id` | str | unique per request |
| `model` | str | the upstream deployment id after translation |
| `stream` | bool | streaming or single-response |
| `client_protocol` | `"anthropic"` \| `"openai"` | which client surface |
| `message_count` | int | length of the messages array |
| `tool_count` | int | number of tools declared on the request |
| `cache_control_seen` | int | Anthropic cache_control blocks observed |
| `cache_control_translated` | int | cache_control blocks forwarded as DIAL breakpoints |
| `cache_control_strategy` | str | one of `passthrough`, `translated_dial_breakpoint`, `stripped`, `dropped_non_anthropic_upstream` |
| `features_stripped` | list[str] | request fields dropped to satisfy upstream capability gaps (e.g. `["tools","stop"]` for deepseek/google.gemma) |
| `tool_inventory_hash` | str \| null | 16-char SHA-256 prefix of the tool inventory — drift canary |
| `tool_inventory` | list[obj] | per-tool `{name, kind, mcp_server, schema_sha_in, schema_sha_out, drift}` |
| `tools_drift_count` | int | how many tools' schemas mutated through translation (>0 = bug) |
| `tools_native_count` | int | tools tagged as Anthropic-native |
| `tools_mcp_count` | int | tools routed through MCP servers |
| `tools_other_count` | int | tools that didn't match native/MCP heuristics |
| `mcp_servers_seen` | list[str] | distinct MCP server names referenced in this request |
| `max_tokens_clamp` | obj \| null | non-null when the adapter clamped `max_tokens` to fit context. Fields: `{original, clamped, estimated_input_tokens, max_context, model}` |
| `target_model_family` | str | derived family tag (`anthropic`, `qwen`, `moonshotai`, `minimax`, etc.) |
| `client_name` | str | parsed User-Agent |

### `response_out`
Emitted after a successful upstream response.

| Field | Type | Meaning |
|---|---|---|
| `request_id` | str | matches the corresponding `request_in` |
| `stream` | bool | how the response was delivered |
| `status` | int | HTTP status returned to client |
| `elapsed_ms` | int | wall time from request_in to first byte of response |
| `stop_reason` | str | Anthropic-shape stop_reason (`end_turn`, `tool_use`, `max_tokens`, ...) |
| `input_tokens` | int | reported by upstream |
| `output_tokens` | int | reported by upstream |
| `cache_read_input_tokens` | int | upstream-reported cache-hit tokens |
| `cache_creation_input_tokens` | int | upstream-reported cache-miss tokens written |
| `tool_calls_by_name` | dict[str,int] | per-tool invocation counts in this response. Keys are the **client's original** tool names, even if the adapter aliased them on the wire (see 64-char Bedrock fix). |
| `cost_usd_estimate` | float | optional, present when the price table is loaded |
| `shadow_dispatched` | bool | true when a parallel shadow request was mirrored |
| `target_model_family` | str | as above |
| `client_name` | str | as above |

### `error`
Emitted on any failure path. The `reason` field discriminates between failure classes.

| Field | Type | Notes |
|---|---|---|
| `request_id` | str | when scoped to a single request |
| `reason` | str | one of `invalid_model`, `translation_failed`, `upstream_connect`, `upstream_status`, `upstream_non_json`, `models_upstream_read`, `models_upstream_status`, `models_upstream_connect` |
| `status` | int | when `reason == upstream_status` |
| `body_snippet` | str | first ~300 chars of upstream body |
| `message` | str | brief human-readable description |
| `target_model_family` | str | when known |
| `client_name` | str | when known |

## Cardinality notes for metrics pipelines

When wiring this into a metrics backend, treat these fields as **high-cardinality dimensions** (don't blindly bucket them as labels):

- `request_id` — unique per request. Never a label; use it for tracing.
- `tool_inventory_hash` — bounded but large (one per distinct tool set).
- `tool_calls_by_name` keys — bounded by your tool inventory (~50 typical).
- `body_snippet` — free text; log only, never label.

These are safe label dimensions:

- `event`, `reason`, `status`, `stream`, `stop_reason`
- `client_name`, `target_model_family`, `cache_control_strategy`
- `features_stripped` (sort+join before labeling)

## How the reference exporter maps this

`ccppm/claude_code_events_exporter.py` tails the JSON stream and writes InfluxDB line protocol. It exists for two reasons:

1. **Reference implementation** — read the source to understand one workable mapping into a TSDB.
2. **Local-dev fast path** — pair with the bundled Grafana dashboards for an out-of-the-box observability stack.

If you're shipping this adapter into a Prometheus or OpenTelemetry environment, you don't need the InfluxDB exporter. Tail the stdout JSON yourself (e.g. via a Vector / Fluentbit / OTEL Collector pipeline) and emit your own metric/log records using this schema as the source of truth.

## Compatibility

Events listed here are stable across the `0.x` adapter line. New fields may be added to existing events; existing fields will not be removed or renamed without a major version bump and a documented migration path.
