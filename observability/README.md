# Observability bundle (optional)

This directory ships a reference observability stack for the adapter. Everything here is **optional** — the adapter runs fine without any of it. Use this bundle when you want a one-command Grafana stack out of the box, or as documentation for what to measure if you have your own observability backend.

## What's in here

| File | Purpose |
|---|---|
| `EVENT_SCHEMA.md` | **The contract.** What JSON events the adapter emits on stdout, what fields each carries, and which fields are safe to use as metric labels vs. high-cardinality. **Start here** if you're wiring your own pipeline. |
| `grafana/dial-bridge-overview.json` | Grafana dashboard: *Claude Code — Anthropic vs DIAL routing comparison*. Side-by-side panels for request/error/latency/token-usage on Anthropic-on-DIAL vs OSS-on-DIAL deployments. The business-logic dashboard. |
| `grafana/dial-overview.json` | Grafana dashboard: *DIAL Sandbox Overview*. Adapter-level health: error rates, capability-strip activity, tool drift, cache_control strategy distribution. |
| `../ccppm/exporter.py` | InfluxDB exporter daemon. Tails the adapter's stdout JSON stream, computes rolling metrics, writes InfluxDB line protocol. |
| `../ccppm/claude_code_events_exporter.py` | Companion exporter that pushes Claude Code's own internal event stream (when telemetry is enabled). |
| `../ccppm/metrics_from_log.py` | Pure computation module: given a window of JSON event lines, produce a metrics dict. Used by the exporter daemon and also unit-testable. |

## Architectural model

```
                                        ┌──────────────────────┐
   Claude Code  ──HTTP──>  adapter  ──┬──>  upstream (DIAL)    │
                  (stdout JSON event)  │                       │
                                       │                        │
                                       v                        │
                              ccppm/exporter.py                  │
                              (tail JSON events)                 │
                                       │                        │
                                       v                        │
                                  InfluxDB                       │
                                       │                        │
                                       v                        │
                                  Grafana (dashboards)           │
                                       │                        │
                                       v                        │
                                  Operator / SRE                 │
```

The adapter never speaks to InfluxDB or Grafana directly. It writes one structured JSON line per event to stdout. The exporter daemon is what makes those events readable as metrics; the dashboards are what makes them readable as a story.

## Quick start (reference stack — InfluxDB + Grafana)

1. **Capture adapter stdout** to a log file or directly to a pipe.
   ```bash
   python app.py 2>&1 | tee /var/log/adapter.jsonl
   ```

2. **Run the exporter daemon** pointing at that log:
   ```bash
   export INFLUXDB_URL=http://localhost:8086
   export INFLUXDB_ORG=<your-org>
   export INFLUXDB_BUCKET=dial-bridge
   export INFLUXDB_TOKEN=<your-token>
   python -m ccppm.exporter --log /var/log/adapter.jsonl
   ```
   The exporter tails the file and writes InfluxDB line protocol. See `ccppm/exporter.py` for the full env var contract.

3. **Import the Grafana dashboards.** In Grafana, *Dashboards → Import*, upload the JSON files in `grafana/`. When prompted, select your InfluxDB datasource (the dashboards reference it as `${DS_INFLUXDB}` and Grafana's Import UI will ask you to map it).

## Using a different observability backend

The reference stack uses InfluxDB because that's what we run in our development sandbox. **Your stack can be anything.** Read `EVENT_SCHEMA.md` for the field contract and wire up whichever pipeline you prefer:

- **OpenTelemetry Collector**: tail stdout with the [`filelog` receiver](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/receiver/filelogreceiver), parse with the JSON operator, route to a [logs / metrics / traces pipeline](https://opentelemetry.io/docs/collector/configuration/) of your choice.
- **Prometheus + Loki**: ship logs to Loki, run a `count_over_time` rate; for actual histograms, use Vector to transform stdout JSON into Prometheus exposition format and scrape it.
- **Datadog**: their [agent](https://docs.datadoghq.com/logs/log_collection/) tails JSON natively; map `event`, `target_model_family`, `client_name` to Datadog facets.
- **Custom**: any tail-and-parse pipeline works. The events are pure newline-delimited JSON.

The reference InfluxDB exporter is ~1100 lines across `exporter.py + claude_code_events_exporter.py + metrics_from_log.py`. The pure-math separation in `metrics_from_log.py` is intentional — that's the piece worth reading even if you build your own pipeline, because it documents *how to derive each panel's value* from the raw events.

## Sanitization notes

The shipped dashboards use a Grafana datasource template variable (`${DS_INFLUXDB}`) — Grafana's Import UI will ask you to pick your actual datasource. No internal hostnames, organization names, or environment-specific URLs are baked into any of these artifacts. If you find one, please open an issue.
