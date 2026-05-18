# Observability bundle (optional)

This directory ships a reference observability stack for the adapter. Everything here is **optional** — the adapter runs fine without any of it. Use this bundle when you want a one-command Grafana stack out of the box, or as documentation for what to measure if you have your own observability backend.

## What's in here

| File | Purpose |
|---|---|
| `EVENT_SCHEMA.md` | **The contract.** What JSON events the adapter emits, what fields each carries, and which fields are safe to use as metric labels vs. high-cardinality. **Start here** if you're wiring your own pipeline. |
| `grafana/dial-bridge-overview.json` | Grafana dashboard: *Claude Code — Anthropic vs DIAL routing comparison*. Side-by-side panels for request/error/latency/token-usage on Anthropic-on-DIAL vs OSS-on-DIAL deployments. The business-logic dashboard. |
| `grafana/dial-overview.json` | Grafana dashboard: *Adapter metrics overview*. Rolling CCPPM metrics from `ccppm/exporter.py` (`adapter_metrics` measurement). |
| `../ccppm/exporter.py` | InfluxDB exporter daemon. Tails the adapter GFLog file(s), computes rolling metrics, writes InfluxDB line protocol. |
| `../ccppm/claude_code_events_exporter.py` | Optional sidecar: restructures Claude Code OTel events already in Influx (`logs` measurement) into queryable `claude_code_events` records. |
| `../ccppm/metrics_from_log.py` | Pure computation module: given a window of JSON event lines, produce a metrics dict. Used by the exporter daemon and also unit-testable. |

## Architectural model

```
                                        ┌──────────────────────┐
   Claude Code  ──HTTP──>  adapter  ──┬──>  upstream (DIAL)    │
                  (GFLog JSON events)  │                       │
                                       │                        │
                                       v                        │
                              ccppm/exporter.py                  │
                              (tail log file)                    │
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

The adapter never speaks to InfluxDB or Grafana directly. It writes one structured JSON line per event to **`ANTHROPIC_DIAL_ADAPTER_LOG`** (default `/var/log/anthropic-dial-adapter/adapter.log`) and mirrors a short prefixed line to **stderr** for `docker logs`. The exporter daemon tails the log file and writes metrics; the dashboards visualize them.

## Quick start (reference stack — InfluxDB + Grafana)

1. **Run the adapter** with a writable log path (default works in Docker when `/var/log/anthropic-dial-adapter` exists):
   ```bash
   export ANTHROPIC_DIAL_ADAPTER_LOG=/var/log/anthropic-dial-adapter/adapter.log
   python app.py
   ```

2. **Run the exporter daemon** (tails the same log path via env):
   ```bash
   export INFLUX_URL=http://localhost:8086
   export INFLUX_ORG=dial-sandbox
   export INFLUX_BUCKET=dial-metrics
   export INFLUX_TOKEN=<your-token>
   # Or use INFLUXDB_URL / INFLUXDB_ORG / INFLUXDB_BUCKET / INFLUXDB_TOKEN (aliases).
   python -m ccppm.exporter --once --log /var/log/anthropic-dial-adapter/adapter.log
   # Or set ANTHROPIC_DIAL_ADAPTER_LOG instead of --log; omit --once for the 30s loop.
   ```
   See `ccppm/exporter.py` for the full env var contract (`ADAPTER_LOG_DIR`, `ADAPTER_LOG_FILES`, `ADAPTER_METRICS_WINDOW_MINUTES`, healthcheck knobs, etc.).

3. **Import the Grafana dashboards.** In Grafana, *Dashboards → Import*, upload the JSON files in `grafana/`. When prompted, select your InfluxDB datasource (the dashboards reference it as `${DS_INFLUXDB}` and Grafana's Import UI will ask you to map it).

## Using a different observability backend

The reference stack uses InfluxDB because that's what we run in our development sandbox. **Your stack can be anything.** Read `EVENT_SCHEMA.md` for the field contract and wire up whichever pipeline you prefer:

- **OpenTelemetry Collector**: tail the log file with the [`filelog` receiver](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/receiver/filelogreceiver), parse with the JSON operator, route to a [logs / metrics / traces pipeline](https://opentelemetry.io/docs/collector/configuration/) of your choice.
- **Prometheus + Loki**: ship logs to Loki, run a `count_over_time` rate; for actual histograms, use Vector to transform GFLog JSON into Prometheus exposition format and scrape it.
- **Datadog**: their [agent](https://docs.datadoghq.com/logs/log_collection/) tails JSON natively; map `event`, `target_model_family`, `client_name` to Datadog facets.
- **Custom**: any tail-and-parse pipeline works. The events are pure newline-delimited JSON.

The reference InfluxDB exporter is ~1100 lines across `exporter.py + claude_code_events_exporter.py + metrics_from_log.py`. The pure-math separation in `metrics_from_log.py` is intentional — that's the piece worth reading even if you build your own pipeline, because it documents *how to derive each panel's value* from the raw events.

## Sanitization notes

The shipped dashboards use a Grafana datasource template variable (`${DS_INFLUXDB}`) — Grafana's Import UI will ask you to pick your actual datasource. No internal hostnames, organization names, or environment-specific URLs are baked into any of these artifacts. If you find one, please open an issue.
