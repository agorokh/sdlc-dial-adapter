"""Operator-curated Bedrock list-price estimates for the adapter's structured log."""

from __future__ import annotations

import json
import math
import re

# Cross-region inference profiles on Bedrock (see Influx model tags).
_REGIONAL_PREFIX_RE = re.compile(
    r"^(?:global|us|eu|apac|ap|sa|ca)\.",
    re.IGNORECASE,
)
# Anthropic versioned inference profile ids, e.g. ...-20251001-v1:0
_VERSION_DATE_SUFFIX_RE = re.compile(r"-\d{8}-v\d+:\d+$", re.IGNORECASE)


def normalize_model_id_for_pricing(model_id: str) -> str:
    """Strip Bedrock cross-region prefixes and Anthropic dated profile suffixes."""
    s = (model_id or "").strip()
    while True:
        m = _REGIONAL_PREFIX_RE.match(s)
        if not m:
            break
        s = s[m.end() :]
    s = _VERSION_DATE_SUFFIX_RE.sub("", s)
    return s


def _numeric_rate(val: object) -> bool:
    """True for finite JSON numbers; false for bool and non-finite floats."""
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return False
    return math.isfinite(float(val))


def _coerce_table_rows(parsed: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mid, row in parsed.items():
        if not isinstance(mid, str) or not mid.strip():
            continue
        if not isinstance(row, dict):
            continue
        pin = row.get("in")
        pout = row.get("out")
        if not _numeric_rate(pin) or not _numeric_rate(pout):
            continue
        entry: dict[str, float] = {"in": float(pin), "out": float(pout)}
        cr = row.get("cache_read")
        if _numeric_rate(cr):
            entry["cache_read"] = float(cr)
        cw = row.get("cache_write")
        if _numeric_rate(cw):
            entry["cache_write"] = float(cw)
        # Normalize keys the same way as runtime model ids so regional/profile
        # variants in the operator JSON match cross-region traffic tags.
        norm_key = normalize_model_id_for_pricing(mid)
        if not norm_key:
            continue
        out[norm_key] = entry
    return out


def _loads_json_root(raw: str) -> tuple[object | None, str | None]:
    """``json.loads`` with a single error surface for callers to branch on."""
    try:
        return json.loads(raw), None
    except (ValueError, TypeError) as e:
        return None, str(e)


def parse_price_table_json(raw: str) -> dict[str, dict[str, float]]:
    """Parse ``ANTHROPIC_DIAL_PRICE_TABLE_JSON`` into model_id -> rate row.

    Each row must include numeric ``in`` and ``out`` (USD per 1M tokens).
    Optional ``cache_read`` and ``cache_write`` (USD per 1M tokens).
    Malformed rows are skipped; syntactically invalid JSON returns {}.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    parsed, err = _loads_json_root(raw)
    if err is not None or not isinstance(parsed, dict):
        return {}
    return _coerce_table_rows(parsed)


def load_price_table_env(raw: str | None) -> tuple[dict[str, dict[str, float]], str | None]:
    """Load operator JSON from env. Returns ``({}, error)`` on hard parse failure."""
    raw = (raw or "").strip()
    if not raw:
        return {}, None
    parsed, err = _loads_json_root(raw)
    if err is not None:
        return {}, err
    if not isinstance(parsed, dict):
        return {}, "root_not_object"
    return _coerce_table_rows(parsed), None


def estimate_cost_usd(
    *,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
    table: dict[str, dict[str, float]],
) -> float | None:
    """Return estimated USD for one response, or None if model is not priced."""
    key = normalize_model_id_for_pricing(model_id)
    row = table.get(key)
    if not row:
        return None
    inp = max(0, int(input_tokens or 0))
    out_t = max(0, int(output_tokens or 0))
    cr_tok = max(0, int(cache_read_input_tokens or 0))
    cw_tok = max(0, int(cache_creation_input_tokens or 0))
    cost = (inp / 1_000_000.0) * row["in"] + (out_t / 1_000_000.0) * row["out"]
    if "cache_read" in row and cr_tok:
        cost += (cr_tok / 1_000_000.0) * row["cache_read"]
    if "cache_write" in row and cw_tok:
        cost += (cw_tok / 1_000_000.0) * row["cache_write"]
    return cost
