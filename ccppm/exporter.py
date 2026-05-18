"""Sidecar: rolling CCPPM snapshot from adapter GFLog files → InfluxDB (Issue #79)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .log_window import (
    _strip_log_line,
    annotate_and_partition,
    events_in_window,
    merged_events_sorted,
)
from .metrics_from_log import compute
from .influx_escape import (
    influx_escape_measurement as _influx_escape_measurement,
    influx_escape_string_field as _influx_escape_string_field,
    influx_escape_tag as _influx_escape_tag,
)


def metrics_to_line_protocol(
    measurement: str,
    tags: dict[str, str],
    metrics: dict[str, Any],
    ts_ns: int,
) -> str:
    # Tag keys follow the same escaping rules as tag values (including '=').
    tag_parts = [
        f"{_influx_escape_tag(k)}={_influx_escape_tag(v)}" for k, v in sorted(tags.items())
    ]
    tag_str = ("," + ",".join(tag_parts)) if tag_parts else ""

    fields: list[str] = []
    m = metrics

    def add_float(name: str, val: Any) -> None:
        if val is None:
            return
        if isinstance(val, (int, float)):
            fields.append(f"{name}={float(val)}")

    def add_int(name: str, val: Any) -> None:
        if isinstance(val, bool):
            fields.append(f"{name}={1 if val else 0}i")
        elif isinstance(val, int):
            fields.append(f"{name}={val}i")

    add_float("sse_event_emission_ratio_mean", m.get("sse_event_emission_ratio_mean"))
    add_float("tool_use_round_trip_success_rate", m.get("tool_use_round_trip_success_rate"))
    add_float("partial_message_error_rate", m.get("partial_message_error_rate"))
    add_float("tool_use_id_stability", m.get("tool_use_id_stability"))
    add_int("cache_control_seen_total", m.get("cache_control_seen_total"))
    add_int("cache_control_passthrough_total", m.get("cache_control_passthrough_total"))
    add_int("cache_control_dropped_total", m.get("cache_control_dropped_total"))
    add_float("end_to_end_p50_ms", m.get("end_to_end_p50_ms"))
    add_float("end_to_end_p95_ms", m.get("end_to_end_p95_ms"))
    add_int("input_tokens_total", m.get("input_tokens_total"))
    add_int("output_tokens_total", m.get("output_tokens_total"))
    add_int("cache_read_input_tokens_total", m.get("cache_read_input_tokens_total"))
    add_float("cost_usd_estimate_total", m.get("cost_usd_estimate_total"))
    mu = m.get("mcp_path_uninterrupted")
    if isinstance(mu, bool):
        fields.append(f"mcp_path_uninterrupted={1 if mu else 0}i")

    errs = m.get("errors_by_reason") or {}
    if isinstance(errs, dict):
        r429 = errs.get("upstream_rate_limit_429")
        if isinstance(r429, (int, float)):
            fields.append(f"errors_upstream_rate_limit_429={int(r429)}i")

    dist = m.get("cache_control_strategy_distribution") or {}
    if isinstance(dist, dict):
        for k, v in dist.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, (int, float)):
                continue
            safe = "".join(ch if ch.isalnum() else "_" for ch in k).strip("_") or "unknown"
            fields.append(f"cache_control_strat_{safe}={int(v)}i")

    probe = m.get("upstream_cache_probe_result")
    if isinstance(probe, str) and probe:
        fields.append(f'upstream_cache_probe_result="{_influx_escape_string_field(probe)}"')

    if not fields:
        fields.append("heartbeat=1i")

    return f"{_influx_escape_measurement(measurement)}{tag_str} {','.join(fields)} {ts_ns}"


def write_lines(
    influx_url: str,
    org: str,
    bucket: str,
    token: str,
    lines: list[str],
    *,
    precision: str = "ns",
) -> None:
    if not lines:
        return
    from urllib.parse import quote

    url = (
        f"{influx_url.rstrip('/')}/api/v2/write?"
        f"org={quote(org)}&bucket={quote(bucket)}&precision={quote(precision)}"
    )
    body = ("\n".join(lines) + "\n").encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "text/plain; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"Influx write HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2048]
        raise RuntimeError(f"Influx write failed: HTTP {e.code}: {detail}") from e


def _state_path() -> Path:
    return Path(os.environ.get("ADAPTER_METRICS_STATE_PATH", "/tmp/adapter_metrics_exporter_state"))


def _record_influx_write_ok() -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(int(time.time())), encoding="utf-8")


def _last_influx_write_within(seconds: float) -> bool:
    p = _state_path()
    if not p.exists():
        return False
    try:
        written = float(p.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    return (time.time() - written) <= seconds


def _influx_ready(influx_url: str) -> bool:
    try:
        url = f"{influx_url.rstrip('/')}/ready"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status in (200, 204)
    except OSError:
        return False


def _tail_nonempty_stripped_lines(path: Path, max_bytes: int, maxlen: int) -> list[str]:
    if not path.exists():
        return []
    if max_bytes <= 0:
        return []
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    dq: deque[str] = deque(maxlen=maxlen)
    with path.open(encoding="utf-8", errors="replace") as f:
        if start > 0:
            f.seek(start - 1)
            prev = f.read(1)
            if prev not in ("\n", "\r"):
                f.readline()
            else:
                f.seek(start)
        for raw in f:
            s = raw.strip()
            if s:
                dq.append(s)
    return list(dq)


def run_healthcheck(log_paths: list[Path]) -> int:
    from .log_window import iter_parsed_lines, parse_adapter_timestamp

    tail_bytes = int(os.environ.get("ADAPTER_METRICS_HEALTH_TAIL_BYTES", str(2 * 1024 * 1024)))
    max_lines = int(os.environ.get("ADAPTER_METRICS_HEALTH_MAX_LINES", "500"))
    any_nonempty_log = False
    found_json = False
    for p in log_paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        any_nonempty_log = True
        for raw in _tail_nonempty_stripped_lines(p, tail_bytes, max_lines):
            st = _strip_log_line(raw)
            if not st or not st.startswith("{"):
                continue
            try:
                obj = json.loads(st)
            except (ValueError, TypeError):
                return 1
            if not isinstance(obj, dict):
                return 1
            found_json = True
            ts_raw = obj.get("@timestamp")
            if not isinstance(ts_raw, str) or parse_adapter_timestamp(ts_raw) is None:
                return 1
            ev = obj.get("event")
            if not isinstance(ev, str) or not ev:
                return 1
    if any_nonempty_log and not found_json:
        return 1

    # Strict JSON parse over the same tail window as metrics so a bad line
    # cannot scroll out of the sampled 500-line heuristic above.
    try:
        for p in log_paths:
            if not p.exists() or p.stat().st_size == 0:
                continue
            for _ in iter_parsed_lines([p], strict=True, max_tail_bytes=tail_bytes):
                pass
    except ValueError:
        return 1

    require = os.environ.get("ADAPTER_METRICS_HEALTH_REQUIRE_INFLUX", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if require and os.environ.get("INFLUX_TOKEN"):
        influx_url = os.environ.get("INFLUX_URL", "http://influxdb:8086")
        if not _influx_ready(influx_url):
            return 1
        max_age = float(os.environ.get("ADAPTER_METRICS_HEALTH_MAX_WRITE_AGE_SEC", "120"))
        if not _last_influx_write_within(max_age):
            return 1
    return 0


def _log_paths_from_env() -> list[Path]:
    log_path = os.environ.get("ANTHROPIC_DIAL_ADAPTER_LOG", "").strip()
    if log_path:
        return [Path(log_path)]
    base = Path(os.environ.get("ADAPTER_LOG_DIR", "/var/log/anthropic-dial-adapter"))
    files = os.environ.get("ADAPTER_LOG_FILES", "adapter.log")
    return [base / name.strip() for name in files.split(",") if name.strip()]


def run_once() -> list[str]:
    paths = _log_paths_from_env()
    window_minutes = float(os.environ.get("ADAPTER_METRICS_WINDOW_MINUTES", "5"))
    window_seconds = window_minutes * 60.0
    measurement = os.environ.get("INFLUX_MEASUREMENT", "adapter_metrics")
    influx_url = os.environ.get("INFLUX_URL", "http://influxdb:8086")
    org = os.environ.get("INFLUX_ORG", "dial-sandbox")
    bucket = os.environ.get("INFLUX_BUCKET", "dial-metrics")
    token = os.environ.get("INFLUX_TOKEN", "")
    if not token:
        raise RuntimeError("INFLUX_TOKEN is required for Influx writes")

    merged = merged_events_sorted(paths, strict=False)
    now = datetime.now(timezone.utc)
    window_events = events_in_window(merged, window_end=now, window_seconds=window_seconds)
    parts = annotate_and_partition(window_events)
    ts_ns = time.time_ns()
    lines: list[str] = []
    for (client_name, family), evs in parts.items():
        if not evs:
            continue
        m = compute(evs)
        tags = {"client.name": client_name, "target_model_family": family}
        lines.append(metrics_to_line_protocol(measurement, tags, m, ts_ns))
    if not lines:
        scrape = os.environ.get("INFLUX_SCRAPE_MEASUREMENT", "ccppm_scrape")
        lines.append(f"{_influx_escape_measurement(scrape)} empty_window=1i {ts_ns}")
    write_lines(influx_url, org, bucket, token, lines)
    _record_influx_write_ok()
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--healthcheck",
        action="store_true",
        help=(
            "Validate recent GFLog JSON and timestamps; optionally Influx "
            "reachability and last successful write (see env docs)."
        ),
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Compute one snapshot and write to Influx, then exit.",
    )
    args = p.parse_args(argv)
    paths = _log_paths_from_env()
    if args.healthcheck:
        return run_healthcheck(paths)

    if not os.environ.get("INFLUX_TOKEN", "").strip():
        print("[adapter-metrics-exporter] FATAL: INFLUX_TOKEN is required", file=sys.stderr)
        return 1

    if args.once:
        run_once()
        return 0

    interval = float(os.environ.get("ADAPTER_METRICS_INTERVAL_SECONDS", "30"))
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[adapter-metrics-exporter] ERROR: {e}", file=sys.stderr)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
