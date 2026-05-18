"""Claude Code OTel events → structured InfluxDB records.

Anthropic's claude_code.* OTel events (api_request, tool_result, tool_decision,
api_retries_exhausted, internal_error, user_prompt, mcp_server_connection, etc.)
land in our InfluxDB under measurement="logs" with all attributes packed into
a single JSON-stringified field. That makes them effectively un-queryable from
Grafana — Flux's json.parse on every panel render is slow and awkward.

This sidecar reads the logs measurement on a rolling window, parses the
attributes JSON, and writes structured records to measurement="claude_code_events"
with the dimensions we actually want to slice on as InfluxDB tags (event_name,
model, client_name, target_model_family, tool_name, success, status_code) and
the metrics as fields (duration_ms, input_tokens, output_tokens, cost_usd, ...).

Designed mirror of ccppm/exporter.py — same Influx write helpers, same run loop
pattern, same healthcheck contract.

Run continuously every INTERVAL_SECONDS (default 30s) or once with --once.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from .influx_escape import (
    influx_escape_measurement as _esc_measurement,
    influx_escape_string_field as _esc_string_field,
    influx_escape_tag as _esc_tag,
)
from .log_window import _family_from_model

# Influx bucket names are interpolated into Flux; reject metacharacters that
# could break out of the string literal (Bugbot / injection hardening).
_SAFE_INFLUX_BUCKET = re.compile(r"^[A-Za-z0-9_.-]+$")


def _env(*keys: str, default: str = "") -> str:
    for key in keys:
        val = os.environ.get(key)
        if val is not None and str(val).strip() != "":
            return val
    return default


def _validate_influx_bucket(name: str) -> str:
    if not name or _SAFE_INFLUX_BUCKET.fullmatch(name) is None:
        raise ValueError(
            "INFLUX_BUCKET must match ^[A-Za-z0-9_.-]+$ (no quotes or spaces); "
            f"got {name!r}"
        )
    return name


# ---------------------------------------------------------------------------
# Influx write helpers (mirror ccppm/exporter.py — keep identical so we can
# consolidate later)
# ---------------------------------------------------------------------------


def _rfc3339_utc_z_to_epoch_ns(t: str) -> int:
    """Parse Flux/Grafana ``...Z`` timestamps with nanosecond precision (int ns)."""
    s = t.strip()
    if not s.endswith("Z"):
        raise ValueError("expected UTC Zulu timestamp")
    s = s[:-1]
    if "." in s:
        main, frac = s.rsplit(".", 1)
        digits = re.sub(r"[^0-9]", "", frac)
        if not digits:
            frac_ns = 0
        else:
            digits = (digits + "000000000")[:9]
            frac_ns = int(digits)
        dt = datetime.fromisoformat(main).replace(tzinfo=timezone.utc)
    else:
        frac_ns = 0
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    sec = int((dt - epoch).total_seconds())
    return sec * 1_000_000_000 + frac_ns


# ---------------------------------------------------------------------------
# Event → structured record extraction
# ---------------------------------------------------------------------------


# Per-event-name extractors. Each returns (tags, fields) given the event
# attributes dict. Tags are LOW-CARDINALITY dimensions for grouping;
# fields are numeric/string values to plot.
def _extract_api_request(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "api_request",
        "model": str(attrs.get("model", "unknown")),
        "client_name": str(attrs.get("client.name", "unknown")),
        "target_model_family": _family_from_model(attrs.get("model")),
        "query_source": str(attrs.get("query_source", "")),
        "speed": str(attrs.get("speed", "")),
        "effort": str(attrs.get("effort", "")),
    }
    fields: dict[str, Any] = {"count": 1}
    for k_src, k_dst in (
        ("duration_ms", "duration_ms"),
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_read_tokens", "cache_read_tokens"),
        ("cache_creation_tokens", "cache_creation_tokens"),
        ("cost_usd", "cost_usd"),
    ):
        fv = _finite_float(attrs.get(k_src))
        if fv is not None:
            fields[k_dst] = fv
    # stop_reason occasionally on api_request events — newer SDK versions
    stop_reason = attrs.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason:
        fields["stop_reason_s"] = stop_reason  # string field — surfaces on tables
    return tags, fields


def _extract_tool_result(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "tool_result",
        "tool_name": str(attrs.get("tool_name", "unknown")),
        "client_name": str(attrs.get("client.name", "unknown")),
        "success": "true" if _truthy_attr(attrs.get("success")) else "false",
        "mcp_server_scope": str(attrs.get("mcp_server_scope", "")),
        "error_type": str(attrs.get("error_type", "")),
    }
    fields: dict[str, Any] = {}
    dur = _finite_float(attrs.get("duration_ms"))
    if dur is not None:
        fields["duration_ms"] = dur
    for k in ("tool_input_size_bytes", "tool_result_size_bytes"):
        fv = _finite_float(attrs.get(k))
        if fv is not None:
            fields[k] = fv
    # Heartbeat for counting — every emitted row is one tool_result
    fields["count"] = 1
    return tags, fields


def _extract_api_retries_exhausted(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "api_retries_exhausted",
        "model": str(attrs.get("model", "unknown")),
        "client_name": str(attrs.get("client.name", "unknown")),
        "target_model_family": _family_from_model(attrs.get("model")),
        "status_code": str(attrs.get("status_code", "")),
    }
    fields: dict[str, Any] = {"count": 1}
    for k in ("total_attempts", "total_retry_duration_ms"):
        fv = _finite_float(attrs.get(k))
        if fv is not None:
            fields[k] = fv
    return tags, fields


def _extract_api_error(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "api_error",
        "model": str(attrs.get("model", "unknown")),
        "client_name": str(attrs.get("client.name", "unknown")),
        "target_model_family": _family_from_model(attrs.get("model")),
        "status_code": str(attrs.get("status_code", "")),
    }
    fields: dict[str, Any] = {"count": 1}
    for k in ("duration_ms", "attempt"):
        fv = _finite_float(attrs.get(k))
        if fv is not None:
            fields[k] = fv
    return tags, fields


def _extract_internal_error(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "internal_error",
        "client_name": str(attrs.get("client.name", "unknown")),
        "error_name": str(attrs.get("error_name", "unknown"))[:64],
    }
    return tags, {"count": 1}


def _extract_user_prompt(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "user_prompt",
        "client_name": str(attrs.get("client.name", "unknown")),
        "command_name": str(attrs.get("command_name", ""))[:64],
        "command_source": str(attrs.get("command_source", "")),
    }
    fields: dict[str, Any] = {"count": 1}
    pl = _finite_float(attrs.get("prompt_length"))
    if pl is not None:
        fields["prompt_length"] = pl
    return tags, fields


def _finite_float(v: Any) -> float | None:
    """Return a finite float for line-protocol fields, or ``None`` to omit."""
    if isinstance(v, bool):
        return None
    if not isinstance(v, (int, float)):
        return None
    x = float(v)
    if not math.isfinite(x):
        return None
    return x


def _parse_non_negative_int(v: Any) -> int:
    """Parse hook-style counters; accept int/float/numeric strings (Sourcery)."""
    if v is None or v == "":
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return max(0, v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return 0
        try:
            return max(0, int(v))
        except OverflowError:
            return 0
    try:
        x = float(str(v).strip())
        if not math.isfinite(x):
            return 0
        return max(0, int(x))
    except (ValueError, OverflowError):
        return 0


def _truthy_attr(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)


def _extract_compaction(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "compaction",
        "client_name": str(attrs.get("client.name", "unknown")),
        "trigger": str(attrs.get("trigger", "")),
        "success": "true" if _truthy_attr(attrs.get("success")) else "false",
    }
    fields: dict[str, Any] = {"count": 1}
    pre: float | None = None
    post: float | None = None
    for k in ("duration_ms", "pre_tokens", "post_tokens"):
        fv = _finite_float(attrs.get(k))
        if fv is not None:
            fields[k] = fv
            if k == "pre_tokens":
                pre = fv
            elif k == "post_tokens":
                post = fv
    if pre is not None and post is not None:
        fields["token_delta"] = pre - post
    return tags, fields


def _extract_hook_execution_complete(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    tags = {
        "event_name": "hook_execution_complete",
        "client_name": str(attrs.get("client.name", "unknown")),
        "hook_event": str(attrs.get("hook_event", "unknown")),
        "hook_source": str(attrs.get("hook_source", "")),
    }
    n_block = _parse_non_negative_int(attrs.get("num_blocking"))
    n_err = _parse_non_negative_int(attrs.get("num_non_blocking_error"))
    fields: dict[str, Any] = {
        "count": 1,
        "hook_errors": float(n_block + n_err),
        "num_blocking": float(n_block),
        "num_non_blocking_error": float(n_err),
    }
    td = _finite_float(attrs.get("total_duration_ms"))
    if td is not None:
        fields["total_duration_ms"] = td
    return tags, fields


def _extract_permission_mode_changed(attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    """Emit every transition; Grafana filters to bypassPermissions for the audit stat."""
    tags = {
        "event_name": "permission_mode_changed",
        "client_name": str(attrs.get("client.name", "unknown")),
        "from_mode": str(attrs.get("from_mode", "")),
        "to_mode": str(attrs.get("to_mode", "")),
        "trigger": str(attrs.get("trigger", "")),
    }
    return tags, {"count": 1}


# Dispatch table. Add new event handlers here — anything not in the map is
# counted as a generic event (event_name tag + count=1) so we can still
# see emission rate.
EXTRACTORS = {
    "claude_code.api_request": _extract_api_request,
    "claude_code.tool_result": _extract_tool_result,
    "claude_code.api_retries_exhausted": _extract_api_retries_exhausted,
    "claude_code.api_error": _extract_api_error,
    "claude_code.internal_error": _extract_internal_error,
    "claude_code.user_prompt": _extract_user_prompt,
    "claude_code.compaction": _extract_compaction,
    "claude_code.hook_execution_complete": _extract_hook_execution_complete,
    "claude_code.permission_mode_changed": _extract_permission_mode_changed,
}


def _generic_extract(event_name: str, attrs: dict) -> tuple[dict[str, str], dict[str, Any]]:
    """Fallback for event types we don't have a dedicated handler for —
    captures count so emission rate is still visible."""
    tags = {
        "event_name": event_name.replace("claude_code.", ""),
        "client_name": str(attrs.get("client.name", "unknown")),
    }
    return tags, {"count": 1}


def to_line_protocol(event_name: str, attrs: dict, ts_ns: int,
                     measurement: str = "claude_code_events") -> str | None:
    """Emit a single InfluxDB line-protocol record. Returns None if no fields."""
    extractor = EXTRACTORS.get(event_name)
    if extractor:
        tags, fields = extractor(attrs)
    else:
        tags, fields = _generic_extract(event_name, attrs)

    # Drop empty tag values so they don't pollute the index.
    tags = {k: v for k, v in tags.items() if v}
    if not fields:
        return None

    tag_str = ""
    if tags:
        tag_str = "," + ",".join(
            f"{_esc_tag(k)}={_esc_tag(v)}" for k, v in sorted(tags.items())
        )

    field_parts: list[str] = []
    for k, v in fields.items():
        if isinstance(v, bool):
            field_parts.append(f"{_esc_tag(k)}={'true' if v else 'false'}")
        elif isinstance(v, int):
            field_parts.append(f"{_esc_tag(k)}={v}i")
        elif isinstance(v, float) and math.isfinite(v):
            field_parts.append(f"{_esc_tag(k)}={v}")
        elif isinstance(v, str):
            field_parts.append(f'{_esc_tag(k)}="{_esc_string_field(v)}"')

    if not field_parts:
        return None
    return f"{_esc_measurement(measurement)}{tag_str} {','.join(field_parts)} {ts_ns}"


# ---------------------------------------------------------------------------
# InfluxDB I/O
# ---------------------------------------------------------------------------

def influx_query(influx_url: str, org: str, token: str, flux: str) -> list[dict[str, Any]]:
    """Run a Flux query, return rows as list of dicts (one per record).
    Uses the /api/v2/query endpoint (CSV response)."""
    url = f"{influx_url.rstrip('/')}/api/v2/query?org={quote(org)}"
    body = json.dumps({"query": flux, "type": "flux"}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Accept": "application/csv",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode()

    # Parse Flux CSV. The `attributes` field value contains JSON with embedded
    # commas and quotes — naive split(",") corrupts it. csv.reader respects
    # RFC 4180 quoting which Flux CSV uses.
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    reader = csv.reader(io.StringIO(text))
    for cols in reader:
        if not cols or all(c == "" for c in cols):
            header = None
            continue
        # Annotation lines start with `#datatype` / `#group` / `#default`.
        if cols[0].startswith("#"):
            header = None
            continue
        if header is None:
            header = cols
            continue
        if len(cols) != len(header):
            continue
        rows.append(dict(zip(header, cols)))
    return rows


def influx_write(influx_url: str, org: str, bucket: str, token: str,
                 lines: list[str], *, precision: str = "ns") -> None:
    if not lines:
        return
    url = (
        f"{influx_url.rstrip('/')}/api/v2/write?"
        f"org={quote(org)}&bucket={quote(bucket)}&precision={quote(precision)}"
    )
    body = ("\n".join(lines) + "\n").encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Token {token}",
        "Content-Type": "text/plain; charset=utf-8",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2048]
        sys.stderr.write(f"influx_write failed: {e.code} {detail}\n")
        raise RuntimeError(f"Influx write rejected: HTTP {e.code}") from e


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _log_row_pair_key(r: dict[str, Any]) -> tuple[str, str]:
    """Stable join key for OTel log rows: wall time + session (when present)."""
    t = str(r.get("_time") or "")
    sid = str(r.get("session.id") or "")
    return (t, sid)


def fetch_recent_events(influx_url: str, org: str, token: str,
                        bucket: str, window_seconds: int) -> list[tuple[str, dict, int]]:
    """Pull recent log records where _field='body' AND _field='attributes'.

    Rows are paired on ``(_time, session.id)`` so concurrent events sharing the
    same timestamp do not collide in a single-key dict.
    """
    bucket = _validate_influx_bucket(bucket)
    keep_cols = '["_time","_value","session.id"]'
    body_q = (
        f'from(bucket: "{bucket}")'
        f' |> range(start: -{window_seconds}s)'
        f' |> filter(fn: (r) => r._measurement == "logs" and r._field == "body")'
        f' |> filter(fn: (r) => r._value =~ /^claude_code\\./)'
        f" |> keep(columns: {keep_cols})"
    )
    attrs_q = (
        f'from(bucket: "{bucket}")'
        f' |> range(start: -{window_seconds}s)'
        f' |> filter(fn: (r) => r._measurement == "logs" and r._field == "attributes")'
        f" |> keep(columns: {keep_cols})"
    )

    body_rows = influx_query(influx_url, org, token, body_q)
    attrs_rows = influx_query(influx_url, org, token, attrs_q)

    body_queues: dict[tuple[str, str], collections.deque[str]] = collections.defaultdict(
        collections.deque
    )
    for r in body_rows:
        if "_time" not in r:
            continue
        v = r.get("_value")
        if not isinstance(v, str) or not v.startswith("claude_code."):
            continue
        body_queues[_log_row_pair_key(r)].append(v)

    out: list[tuple[str, dict, int]] = []
    for r in attrs_rows:
        t = r.get("_time")
        if not t:
            continue
        key = _log_row_pair_key(r)
        q = body_queues.get(key)
        if not q:
            continue
        ev = q.popleft()
        if not ev.startswith("claude_code."):
            continue
        attrs_str = r.get("_value", "")
        try:
            attrs = json.loads(attrs_str)
        except (ValueError, TypeError):
            continue
        try:
            ts_ns = _rfc3339_utc_z_to_epoch_ns(str(t))
        except ValueError:
            continue
        out.append((ev, attrs, ts_ns))
    return out


def run_once(influx_url: str, org: str, bucket: str, token: str,
             window_seconds: int = 60,
             measurement: str = "claude_code_events") -> int:
    """One scrape — fetch events from the last window_seconds, write structured records.
    Returns number of lines written."""
    events = fetch_recent_events(influx_url, org, token, bucket, window_seconds)
    lines: list[str] = []
    for ev_name, attrs, ts_ns in events:
        lp = to_line_protocol(ev_name, attrs, ts_ns, measurement)
        if lp:
            lines.append(lp)
    influx_write(influx_url, org, bucket, token, lines)
    return len(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--healthcheck", action="store_true")
    args = p.parse_args(argv)

    influx_url = _env("INFLUX_URL", "INFLUXDB_URL", default="http://influxdb:8086")
    org = _env("INFLUX_ORG", "INFLUXDB_ORG", default="default")
    bucket = _validate_influx_bucket(_env("INFLUX_BUCKET", "INFLUXDB_BUCKET", default="dial-metrics"))
    token = _env("INFLUX_TOKEN", "INFLUXDB_TOKEN")
    if not token:
        sys.stderr.write("INFLUX_TOKEN is required\n")
        return 2

    interval = int(os.environ.get("INTERVAL_SECONDS", "30"))
    # Use a slight over-window so we don't lose events at boundary edges.
    window = max(int(os.environ.get("WINDOW_SECONDS", str(interval * 2))), 10)

    if args.healthcheck:
        # Smoke test — just confirm we can query InfluxDB
        try:
            influx_query(influx_url, org, token,
                         f'from(bucket: "{bucket}") |> range(start: -1m) |> limit(n: 1)')
            return 0
        except Exception as e:
            sys.stderr.write(f"healthcheck failed: {e}\n")
            return 1

    if args.once:
        n = run_once(influx_url, org, bucket, token, window_seconds=window)
        sys.stdout.write(f"wrote {n} structured records\n")
        return 0

    sys.stdout.write(f"claude-code-events-exporter: interval={interval}s window={window}s\n")
    sys.stdout.flush()
    while True:
        try:
            n = run_once(influx_url, org, bucket, token, window_seconds=window)
            sys.stdout.write(f"[{datetime.now(timezone.utc).isoformat()}] wrote {n} records\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"[{datetime.now(timezone.utc).isoformat()}] error: {e}\n")
            sys.stderr.flush()
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
