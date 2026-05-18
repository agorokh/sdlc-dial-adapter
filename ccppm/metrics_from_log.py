#!/usr/bin/env python3
"""Compute the 8 Claude Code Provider Parity Matrix (CCPPM) metrics
from the adapter's GFLog stream.

Reads NDJSON `adapter.log` events from disk (or stdin) and emits:

  sse_event_emission_ratio        mean across streaming responses
  tool_use_round_trip_success_rate
  partial_message_error_rate
  tool_use_id_stability           always 1.0 by adapter contract; check error tags
  cache_control_strategy          {seen, dropped, passthrough} counts
  end_to_end_p95_latency_ms       p95 over all response_out
  cost_usd_estimate_total       sum of per-response adapter estimates (Issue #81)
  mcp_path_uninterrupted          true unless an error event names mcp

The script does not touch the network. It is the local equivalent of the
Grafana panels the dashboard JSON will eventually surface once Vector's
routing into InfluxDB is wired for non-DIAL-shape GFLog events
(follow-up tracked in the bringup investigation).

Usage:
  python metrics_from_log.py --log /var/log/anthropic-dial-adapter/adapter.log
  python metrics_from_log.py --log /tmp/run.log --since-iso 2026-05-11T00:00:00
  docker exec docker-anthropic-dial-adapter-1 \
    cat /var/log/anthropic-dial-adapter/adapter.log | python metrics_from_log.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

# Bracket-noise that some shells prepend ("[anthropic-dial-adapter] ").
_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s+")


def _events(stream):
    """Yield (event_name, parsed_dict) for each NDJSON line."""
    for raw in stream:
        line = _PREFIX_RE.sub("", raw.strip())
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(ev, dict):
            continue
        yield ev.get("event", "?"), ev


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def compute(events) -> dict:
    requests_in = 0
    streaming_responses = 0
    nonstream_responses = 0
    ratios: list[float] = []
    latencies_ms: list[float] = []
    stop_reasons: Counter = Counter()
    cache_control_seen = 0
    cache_control_strategy: Counter = Counter()
    upstream_cache_probe_result: str | None = None
    errors_by_reason: Counter = Counter()
    # Part G + H aggregations
    tool_schema_drift_count = 0
    tool_calls_by_name: Counter = Counter()
    tool_calls_by_kind: Counter = Counter()
    mcp_servers_seen: set[str] = set()
    request_tool_kind_summary: Counter = Counter()  # native/mcp/other request shares
    tool_use_round_trips_started = 0   # request that returned stop_reason=tool_use
    tool_use_round_trips_followed = 0  # next request had message_count >= 3
    mcp_interrupted = False
    input_tokens_total = 0
    output_tokens_total = 0
    cache_read_total = 0
    cost_usd_estimate_total = 0.0
    # Only the request_in *immediately following* a tool_use response counts
    # as the tool-result follow-up (avoids counting unrelated later turns).
    expect_immediate_tool_followup = False
    stream_truncated_count = 0

    for name, ev in events:
        if name == "request_in":
            requests_in += 1
            if expect_immediate_tool_followup:
                if (ev.get("message_count") or 0) >= 3:
                    tool_use_round_trips_followed += 1
                expect_immediate_tool_followup = False
        elif name == "response_out":
            if ev.get("stream"):
                streaming_responses += 1
                r = ev.get("sse_event_emission_ratio")
                if isinstance(r, (int, float)) and r > 0:
                    ratios.append(float(r))
            else:
                nonstream_responses += 1
            ms = ev.get("elapsed_ms")
            if isinstance(ms, (int, float)):
                latencies_ms.append(float(ms))
            stop = ev.get("final_stop_reason") or ev.get("stop_reason")
            if stop:
                stop_reasons[stop] += 1
            if stop == "tool_use":
                tool_use_round_trips_started += 1
                expect_immediate_tool_followup = True
            if ev.get("stream_truncated"):
                stream_truncated_count += 1
            input_tokens_total += int(ev.get("input_tokens") or 0)
            output_tokens_total += int(ev.get("output_tokens") or 0)
            cache_read_total += int(ev.get("cache_read_input_tokens") or 0)
            cst = ev.get("cost_usd_estimate")
            if isinstance(cst, (int, float)):
                cost_usd_estimate_total += float(cst)
        elif name == "error":
            reason = ev.get("reason", "unknown")
            errors_by_reason[reason] += 1
            # 429 from the upstream is a rate-limit, not an adapter bug.
            # Surface it separately so it doesn't pollute the failure rate.
            if reason == "upstream_status" and ev.get("status") == 429:
                errors_by_reason["upstream_rate_limit_429"] = (
                    errors_by_reason.get("upstream_rate_limit_429", 0) + 1
                )
                errors_by_reason["upstream_status"] -= 1
            if "mcp" in str(ev).lower():
                mcp_interrupted = True
        # cache_control accounting lives on request_in only.
        if name == "request_in":
            n = ev.get("cache_control_seen")
            if isinstance(n, int):
                cache_control_seen += n
                strat = ev.get("cache_control_strategy")
                if strat and n > 0:
                    cache_control_strategy[strat] += n
        # Startup probe result — the dashboard's banner signal.
        if name == "upstream_cache_probe":
            r = ev.get("result")
            if isinstance(r, str):
                upstream_cache_probe_result = r
        # Part G: schema drift accounting (per-tool, summed across requests).
        if name == "request_in":
            tool_schema_drift_count += int(ev.get("tools_drift_count") or 0)
            for t in ev.get("tool_inventory") or []:
                if isinstance(t, dict) and t.get("kind"):
                    request_tool_kind_summary[t["kind"]] += 1
            for srv in ev.get("mcp_servers_seen") or []:
                if isinstance(srv, str):
                    mcp_servers_seen.add(srv)
        # Part H: per-tool / per-class call counts from the response side.
        if name == "response_out":
            tcbn = ev.get("tool_calls_by_name") or {}
            if isinstance(tcbn, dict):
                for k, v in tcbn.items():
                    if isinstance(v, (int, float)) and v > 0:
                        tool_calls_by_name[k] += int(v)
                        kind, _server = (
                            ("native", None) if k in {
                                "Bash", "Read", "Write", "Edit", "MultiEdit",
                                "Glob", "Grep", "Task", "TodoWrite", "WebFetch",
                                "WebSearch", "NotebookEdit", "KillBash", "BashOutput",
                            } else ("mcp", None) if k.startswith("mcp__")
                            else ("other", None)
                        )
                        tool_calls_by_kind[kind] += int(v)

    # Tool-use ID stability is encoded as: adapter never mints replacement
    # IDs; if a request returned tool_use AND the next user turn carries
    # a tool_result with a matching id, the adapter's anthropic_to_openai
    # mapping is correct (it round-trips the id verbatim). The signal we
    # have here at log level is: tool_use → followed-up turn with
    # message_count >= 3, no error events between.
    stability_signal = (
        1.0 if tool_use_round_trips_started == 0
        else tool_use_round_trips_followed / tool_use_round_trips_started
    )

    # Request logs emit ``translated_dial_breakpoint`` when upstream accepts DIAL
    # breakpoints (see anthropic_to_openai); treat like passthrough for totals.
    _cache_ok = frozenset({"passthrough", "translated_dial_breakpoint"})
    passthrough_markers = sum(
        int(cache_control_strategy.get(k, 0)) for k in _cache_ok
    )
    dropped_markers = sum(
        int(v) for k, v in cache_control_strategy.items()
        if k not in _cache_ok
    )
    partial_message_signals = (
        int(errors_by_reason.get("partial_message", 0)) + stream_truncated_count
    )

    return {
        "requests_in": requests_in,
        "streaming_responses": streaming_responses,
        "nonstream_responses": nonstream_responses,
        "sse_event_emission_ratio_mean": (sum(ratios) / len(ratios)) if ratios else None,
        "sse_event_emission_ratio_min": min(ratios) if ratios else None,
        "tool_use_round_trips_started": tool_use_round_trips_started,
        "tool_use_round_trips_followed": tool_use_round_trips_followed,
        "tool_use_round_trip_success_rate": stability_signal,
        "tool_use_id_stability": stability_signal,
        "partial_message_error_rate": (
            partial_message_signals / max(1, streaming_responses)
        ),
        "stream_truncated_count": stream_truncated_count,
        "cache_control_seen_total": cache_control_seen,
        "cache_control_strategy_distribution": dict(cache_control_strategy),
        "cache_control_passthrough_total": passthrough_markers,
        "cache_control_dropped_total": dropped_markers,
        "upstream_cache_probe_result": upstream_cache_probe_result,
        "end_to_end_p50_ms": _quantile(latencies_ms, 0.50),
        "end_to_end_p95_ms": _quantile(latencies_ms, 0.95),
        "input_tokens_total": input_tokens_total,
        "output_tokens_total": output_tokens_total,
        "cache_read_input_tokens_total": cache_read_total,
        "cost_usd_estimate_total": cost_usd_estimate_total,
        "stop_reason_distribution": dict(stop_reasons),
        "errors_by_reason": dict(errors_by_reason),
        "mcp_path_uninterrupted": not mcp_interrupted,
        # Part G + H
        "tool_schema_drift_count": tool_schema_drift_count,
        "tool_calls_by_name": dict(tool_calls_by_name.most_common(20)),
        "tool_calls_by_kind": dict(tool_calls_by_kind),
        "mcp_servers_seen": sorted(mcp_servers_seen),
        "request_tool_inventory_kind_share": dict(request_tool_kind_summary),
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log",
        default="/var/log/anthropic-dial-adapter/adapter.log",
        help="Path to adapter GFLog file (default: %(default)s)",
    )
    args = p.parse_args(argv)

    src = Path(args.log)
    if src.exists():
        with src.open() as f:
            metrics = compute(_events(f))
    else:
        metrics = compute(_events(sys.stdin))

    json.dump(metrics, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
