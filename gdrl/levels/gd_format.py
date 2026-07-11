"""Encode/decode the Geometry Dash level-string wire format.

Outer layer (what the game stores): URL-safe base64 of zlib/gzip-compressed
bytes. Official levels have the 13-byte gzip header stripped, restored here by
prepending the known constant.

Inner layer: "<header>;<obj>;<obj>;..." where each object is a flat list of
comma-separated key,value pairs (key 1 = object id, 2 = x units, 3 = y units,
6 = rotation, ...). See gd.docs for the full key list.

References:
  https://github.com/gd-programming/gd.docs (levelstring_encoding_decoding,
  inner-level-string, level-object)
"""

from __future__ import annotations

import base64
import zlib

# Gzip magic + header that official levels have stripped from the front.
OFFICIAL_GZIP_PREFIX = "H4sIAAAAAAAAA"


def decode_level_string(data: str, official: bool = False) -> str:
    """Base64+zlib decode a stored level string to the inner plaintext string."""
    if official:
        data = OFFICIAL_GZIP_PREFIX + data
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    # 15 | 32 window bits => autodetect zlib vs gzip.
    return zlib.decompress(raw, 15 | 32).decode("utf-8", errors="replace")


def encode_level_string(inner: str) -> str:
    """Compress + URL-safe-base64 an inner string (round-trip of decode)."""
    compressed = zlib.compress(inner.encode("utf-8"), 9)
    return base64.urlsafe_b64encode(compressed).decode("ascii")


def parse_inner(inner: str) -> tuple[dict[str, str], list[dict[int, str]]]:
    """Split an inner string into (header dict, list of object property dicts)."""
    segments = inner.split(";")
    header = _parse_header(segments[0]) if segments else {}
    objects = []
    for seg in segments[1:]:
        if not seg:
            continue
        obj = _parse_kv_int(seg)
        if 1 in obj:  # must have an object id
            objects.append(obj)
    return header, objects


def build_inner(header: str, objects: list[dict[int, float]]) -> str:
    """Inverse of parse_inner for a list of {key: value} object dicts."""
    parts = [header]
    for obj in objects:
        parts.append(",".join(f"{k},{_fmt(v)}" for k, v in obj.items()))
    return ";".join(parts) + ";"


# -- helpers -----------------------------------------------------------------

def _parse_header(seg: str) -> dict[str, str]:
    # Header keys are string-prefixed (kA2, kS38, ...); keep as raw str->str.
    tokens = seg.split(",")
    return {tokens[i]: tokens[i + 1] for i in range(0, len(tokens) - 1, 2)}


def _parse_kv_int(seg: str) -> dict[int, str]:
    tokens = seg.split(",")
    out: dict[int, str] = {}
    for i in range(0, len(tokens) - 1, 2):
        try:
            out[int(tokens[i])] = tokens[i + 1]
        except ValueError:
            continue  # non-integer key: skip (defensive)
    return out


def _fmt(v: float) -> str:
    # Integers stay integers (matches GD's own formatting for grid-aligned objs).
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
