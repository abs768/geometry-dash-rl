"""Object-ID classification for the Geometry Dash level format.

These tables are deliberately conservative and DOCUMENTED AS PARTIAL. GD has
thousands of object IDs; here we map the common ones the cube sim understands
(solid blocks, spikes) plus gamemode/speed portals, and the importer REPORTS
everything it does not recognise rather than guessing. Extend the sets below to
widen coverage — the importer's coverage report tells you which IDs to add.

Coordinates in the level string are in units; UNITS_PER_BLOCK (30) converts to
the block grid the sim uses. An object's stored (x, y) is its CENTER, so the
bottom-left corner the sim wants is value/30 - 0.5.
"""

from __future__ import annotations

UNITS_PER_BLOCK = 30.0

# --- solid, collidable blocks (cube lands on / dies against the side of) ----
# 1 is the classic solid square; 2-6/40/83 etc. are early block variants. This
# is a small, high-confidence subset, not the full block catalogue.
SOLID_IDS: set[int] = {1, 2, 3, 4, 5, 6, 7, 40, 83, 90, 91, 92, 93, 94, 95, 96,
                       97, 98, 99, 100, 101, 102, 468, 469, 470, 471}

# --- hazards (instant death) ------------------------------------------------
# 8 normal spike, 39 short spike, 103 spike, 392 small spike, 205/206/207 saws.
HAZARD_IDS: set[int] = {8, 9, 39, 103, 145, 205, 206, 207, 216, 217, 218,
                        223, 348, 392, 458, 667}

# --- gamemode portals (id -> gamemode name) ---------------------------------
# Used for the coverage report: the cube sim can only simulate cube stretches.
PORTAL_GAMEMODE: dict[int, str] = {
    12: "cube", 13: "ship", 47: "ball", 111: "ufo", 660: "wave",
    745: "robot", 1331: "spider",
}

# --- speed portals (id -> multiplier) ---------------------------------------
PORTAL_SPEED: dict[int, float] = {
    200: 0.5, 201: 1.0, 202: 2.0, 203: 3.0, 1334: 4.0,
}

BLOCK = "block"
SPIKE = "spike"


def classify(object_id: int) -> str:
    """Return one of: 'block', 'spike', 'portal_gamemode', 'portal_speed', 'unknown'."""
    if object_id in HAZARD_IDS:
        return SPIKE
    if object_id in SOLID_IDS:
        return BLOCK
    if object_id in PORTAL_GAMEMODE:
        return "portal_gamemode"
    if object_id in PORTAL_SPEED:
        return "portal_speed"
    return "unknown"


def units_to_block(value_units: float) -> float:
    """Convert a stored center coordinate (units) to a bottom-left block coord."""
    return value_units / UNITS_PER_BLOCK - 0.5
