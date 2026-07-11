"""Binary wire protocol, the Python mirror of gd-mod/src/protocol.hpp.

Struct formats are little-endian (`<`) with no padding, matching the mod's
#pragma pack(1) layout. Keep this in lockstep with protocol.hpp and PROTOCOL.md.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PROTOCOL_VERSION = 0
DEFAULT_PORT = 51717
UNITS_PER_BLOCK = 30.0

# Message type tags.
MSG_HELLO = 0x00
MSG_STATE = 0x01
MSG_ACTION = 0x02
MSG_GEOMETRY = 0x03

# State flag bits.
FLAG_DEAD = 1 << 0
FLAG_COMPLETE = 1 << 1
FLAG_UPSIDE = 1 << 2

# Action flag bits.
ACT_HOLD = 1 << 0
ACT_REQUEST_RESET = 1 << 1
ACT_REQUEST_GEOM = 1 << 2

# StatePayload: u8 type, f32 x, f32 y, f32 vy, u8 grounded, u8 gamemode,
# u8 flags, f32 length, u32 frame, f32 percent, f32 speed_mult, u8 reserved.
STATE_FMT = "<B fff BBB f I f f B"
STATE_SIZE = struct.calcsize(STATE_FMT)
assert STATE_SIZE == 33, STATE_SIZE

# GeometryRecord: u8 kind, f32 x, f32 y.
GEOM_REC_FMT = "<B ff"
GEOM_REC_SIZE = struct.calcsize(GEOM_REC_FMT)
assert GEOM_REC_SIZE == 9, GEOM_REC_SIZE


@dataclass
class State:
    x: float
    y: float
    vy: float
    grounded: bool
    gamemode: int
    flags: int
    length: float
    frame: int
    percent: float
    speed_mult: float

    @property
    def dead(self) -> bool:
        return bool(self.flags & FLAG_DEAD)

    @property
    def complete(self) -> bool:
        return bool(self.flags & FLAG_COMPLETE)

    @classmethod
    def unpack(cls, payload: bytes) -> "State":
        (typ, x, y, vy, grounded, gamemode, flags, length,
         frame, percent, speed_mult, _reserved) = struct.unpack(STATE_FMT, payload)
        if typ != MSG_STATE:
            raise ValueError(f"expected STATE (0x01), got {typ:#x}")
        return cls(x, y, vy, bool(grounded), gamemode, flags,
                   length, frame, percent, speed_mult)

    def pack(self) -> bytes:
        """Only used by the mock server / tests."""
        return struct.pack(STATE_FMT, MSG_STATE, self.x, self.y, self.vy,
                           int(self.grounded), self.gamemode, self.flags,
                           self.length, self.frame, self.percent, self.speed_mult, 0)


def pack_action(hold: bool, request_reset: bool = False, request_geom: bool = False) -> bytes:
    flags = 0
    if hold:
        flags |= ACT_HOLD
    if request_reset:
        flags |= ACT_REQUEST_RESET
    if request_geom:
        flags |= ACT_REQUEST_GEOM
    return struct.pack("<BB", MSG_ACTION, flags)


def unpack_action(payload: bytes) -> int:
    typ, flags = struct.unpack("<BB", payload[:2])
    if typ != MSG_ACTION:
        raise ValueError(f"expected ACTION (0x02), got {typ:#x}")
    return flags


def pack_geometry(records: list[tuple[int, float, float]]) -> bytes:
    """records: list of (kind, x, y). Returns a full GEOMETRY message payload."""
    out = bytes([MSG_GEOMETRY]) + struct.pack("<I", len(records))
    for kind, x, y in records:
        out += struct.pack(GEOM_REC_FMT, kind, x, y)
    return out


def unpack_geometry(payload: bytes) -> list[tuple[int, float, float]]:
    if payload[0] != MSG_GEOMETRY:
        raise ValueError(f"expected GEOMETRY (0x03), got {payload[0]:#x}")
    (count,) = struct.unpack("<I", payload[1:5])
    records = []
    off = 5
    for _ in range(count):
        kind, x, y = struct.unpack(GEOM_REC_FMT, payload[off:off + GEOM_REC_SIZE])
        records.append((kind, x, y))
        off += GEOM_REC_SIZE
    return records
