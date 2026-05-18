"""Shared InfluxDB line-protocol escaping for ``ccppm`` sidecars (Issue #79)."""


def influx_escape_measurement(s: str) -> str:
    return s.replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,")


def influx_escape_tag(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace(" ", "\\ ")
        .replace(",", "\\,")
        .replace("=", "\\=")
        .replace("\n", "\\n")
    )


def influx_escape_string_field(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
