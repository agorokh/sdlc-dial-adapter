"""Rolling-window selection over merged adapter GFLog streams."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s+")


def parse_adapter_timestamp(raw: str | None) -> datetime | None:
    """Parse ``@timestamp`` as emitted by :func:`app.emit` (``%Y-%m-%dT%H:%M:%S%z``)."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    if s.endswith("Z"):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _strip_log_line(raw: str) -> str:
    return _PREFIX_RE.sub("", raw.strip())


@dataclass(frozen=True)
class ParsedLine:
    source_index: int
    line_no: int
    raw: str
    parsed: dict[str, Any] | None
    parse_error: bool


def _iter_tail_lines(path: Path, max_bytes: int) -> Iterator[tuple[int, str]]:
    """Yield ``(line_no_within_tail, raw_line)`` for the last ``max_bytes`` of ``path``."""
    if not path.exists():
        return
    if max_bytes <= 0:
        return
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    with path.open(encoding="utf-8", errors="replace") as f:
        if start > 0:
            # seek(start) may land on a line boundary; an unconditional readline()
            # would drop a full first line. Only discard when start is mid-line.
            f.seek(start - 1)
            prev = f.read(1)
            if prev not in ("\n", "\r"):
                f.readline()
            else:
                f.seek(start)
        yield from enumerate(f, start=1)


def _iter_full_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        yield from enumerate(f, start=1)


def iter_parsed_lines(
    paths: list[Path],
    *,
    strict: bool = False,
    max_tail_bytes: int | None = None,
) -> Iterator[ParsedLine]:
    for source_index, path in enumerate(paths):
        if not path.exists():
            continue
        line_iter: Iterator[tuple[int, str]]
        if max_tail_bytes is not None:
            line_iter = _iter_tail_lines(path, max_tail_bytes)
        else:
            line_iter = _iter_full_lines(path)
        for line_no, raw in line_iter:
            stripped = _strip_log_line(raw)
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except (ValueError, TypeError):
                if strict:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from None
                yield ParsedLine(source_index, line_no, raw, None, True)
                continue
            if not isinstance(obj, dict):
                if strict:
                    raise ValueError(f"JSON root not object at {path}:{line_no}") from None
                yield ParsedLine(source_index, line_no, raw, None, True)
                continue
            yield ParsedLine(source_index, line_no, raw, obj, False)


def merged_events_sorted(
    paths: list[Path],
    *,
    strict: bool = False,
) -> list[tuple[datetime | None, int, int, int, dict[str, Any]]]:
    _default_tail = 32 * 1024 * 1024
    tail_raw = os.environ.get("ADAPTER_METRICS_TAIL_BYTES", str(_default_tail)).strip()
    try:
        max_tail = int(tail_raw)
    except ValueError:
        max_tail = _default_tail
    if max_tail <= 0:
        max_tail = None

    rows: list[tuple[datetime | None, int, int, int, dict[str, Any]]] = []
    for pl in iter_parsed_lines(paths, strict=strict, max_tail_bytes=max_tail):
        if pl.parse_error or pl.parsed is None:
            continue
        ev = dict(pl.parsed)
        ev["_source_index"] = pl.source_index
        ev["_line_no"] = pl.line_no
        ts_raw = ev.get("@timestamp")
        ts = parse_adapter_timestamp(ts_raw if isinstance(ts_raw, str) else None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        sort_key = int(ts.timestamp() * 1_000_000_000)
        rows.append((ts, sort_key, pl.source_index, pl.line_no, ev))
    rows.sort(key=lambda r: (r[1], r[2], r[3]))
    return rows


def events_in_window(
    merged: list[tuple[datetime | None, int, int, int, dict[str, Any]]],
    *,
    window_end: datetime,
    window_seconds: float,
) -> list[dict[str, Any]]:
    if window_end.tzinfo is None:
        window_end = window_end.replace(tzinfo=timezone.utc)
    start = window_end - timedelta(seconds=window_seconds)
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for ts, _sk, src, ln, ev in merged:
        if ts is None:
            continue
        if ts < start or ts >= window_end:
            continue
        key = (src, ln)
        if key in seen:
            continue
        seen.add(key)
        clean = {k: v for k, v in ev.items() if not str(k).startswith("_")}
        out.append(clean)
    return out


# AWS Bedrock cross-region inference id prefixes — these get stripped so
# ``us.anthropic.claude-haiku-...`` and ``global.anthropic.claude-sonnet-...``
# both collapse to family="anthropic" instead of "us" / "global".
_BEDROCK_REGION_PREFIXES = frozenset({"global", "us", "eu", "apac", "ap", "sa", "ca"})


def _family_from_model(model: Any) -> str:
    """Derive a coarse family label from a model deployment id.

    Handles:
      - Anthropic short form Claude Code emits when wrapper sets
        ``ANTHROPIC_DEFAULT_*_MODEL``: ``claude-sonnet-4-6`` → ``anthropic``
      - Namespaced DIAL ids: ``qwen.qwen3-coder-...`` → ``qwen``
      - Bedrock cross-region inference ids: ``us.anthropic.*``,
        ``global.anthropic.*``, ``eu.anthropic.*``, ``apac.anthropic.*`` all
        collapse to family of the SECOND segment (e.g. ``anthropic``) so the
        comparison treats every regional variant as the same provider.
    """
    if not isinstance(model, str) or not model:
        return "unknown"
    if model.startswith("claude-"):
        return "anthropic"
    if "." not in model:
        return "unknown"
    parts = model.split(".")
    head = parts[0]
    if head in _BEDROCK_REGION_PREFIXES:
        # Region-prefixed id — family is the SECOND segment, never the region.
        # Malformed ids like "global." or "us." → "unknown" (no fallback to head).
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return "unknown"
    return head


def annotate_and_partition(
    window_events: list[dict[str, Any]],
) -> dict[tuple[str, str], list[tuple[str, dict[str, Any]]]]:
    # Per-request_id snapshot of (client_name, target_model_family). Used to
    # back-fill response_out / error events that don't carry tags themselves.
    # If the request_in event has an explicit ``target_model_family`` we honor
    # it; otherwise we derive a family from the upstream ``model`` field the
    # adapter always logs (e.g. ``qwen.qwen3-coder-...`` → ``qwen``).
    rid_tags: dict[str, tuple[str, str]] = {}
    for ev in window_events:
        if ev.get("event") != "request_in":
            continue
        rid = ev.get("request_id")
        if rid is None:
            continue
        cn = ev.get("client_name")
        tf = ev.get("target_model_family")
        if not tf:
            tf = _family_from_model(ev.get("model"))
        rid_tags[str(rid)] = (
            str(cn) if cn is not None else "unknown",
            str(tf) if tf else "unknown",
        )

    # Preserve ``events_in_window`` / merge order. Re-sorting ties only by
    # ``(@timestamp, request_id)`` scrambles same-second sequences and breaks
    # ``compute()`` heuristics that depend on event order (tool-use follow-ups).
    ordered: list[dict[str, Any]] = list(window_events)
    filled: list[dict[str, Any]] = []
    for ev in ordered:
        rid = ev.get("request_id")
        cn = ev.get("client_name")
        tf = ev.get("target_model_family")
        # Step 1: inherit from request_in via request_id when this event
        # lacks a tag (response_out / error rarely carry tags themselves).
        if (cn is None or not tf) and rid is not None:
            fb = rid_tags.get(str(rid))
            if fb:
                cn = cn if cn is not None else fb[0]
                tf = tf if tf else fb[1]
        # Step 2: still no family? Try to derive from a `model` field on
        # this event (request_in usually has one even if rid_tags lookup
        # missed for some reason).
        if not tf:
            tf = _family_from_model(ev.get("model"))
        row = dict(ev)
        row["client_name"] = str(cn) if cn is not None else "unknown"
        row["target_model_family"] = str(tf) if tf else "unknown"
        filled.append(row)

    parts: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for ev in filled:
        e = ev.get("event")
        if not isinstance(e, str):
            continue
        cn = str(ev.get("client_name") or "unknown")
        fam = str(ev.get("target_model_family") or "unknown")
        key = (cn, fam)
        parts.setdefault(key, []).append((e, ev))
    return parts
