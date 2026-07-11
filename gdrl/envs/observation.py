"""Observation construction shared by the sim env and the real-game env.

Both backends MUST produce byte-identical observation vectors from the same
(x, y, vy, grounded, level), otherwise a policy trained in the sim would see a
different input distribution in the real game and sim-to-real transfer would
silently degrade. Keeping the one implementation here is what guarantees that.
"""

from __future__ import annotations

import numpy as np

from gdrl.sim.level import BLOCK, Level

GRID_COLS = 20
GRID_ROWS = 10
OBS_Y_SCALE = 10.0
OBS_VY_SCALE = 30.0

OBS_LEN = 4 + GRID_COLS * GRID_ROWS * 2


def lookahead_grid(x: float, level: Level) -> np.ndarray:
    """Occupancy grid of the GRID_COLS columns starting at the player's column."""
    grid = np.zeros((GRID_COLS, GRID_ROWS, 2), dtype=np.float32)
    col0 = int(np.floor(x))
    for obj in level.objects_near(col0, col0 + GRID_COLS):
        c = int(np.floor(obj.x)) - col0
        r = int(np.floor(obj.y))
        if 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS:
            grid[c, r, 0 if obj.type == BLOCK else 1] = 1.0
    return grid


def build_observation(x: float, y: float, vy: float, grounded: bool, level: Level) -> np.ndarray:
    head = np.array(
        [y / OBS_Y_SCALE, vy / OBS_VY_SCALE, float(grounded), x - np.floor(x)],
        dtype=np.float32,
    )
    return np.concatenate([head, lookahead_grid(x, level).ravel()])


def ascii_strip(x: float, y: float, level: Level) -> str:
    """Debug view of the look-ahead window; '@' is the player."""
    grid = lookahead_grid(x, level)
    player_row = int(np.floor(y))
    rows = []
    for r in range(GRID_ROWS - 1, -1, -1):
        line = []
        for c in range(GRID_COLS):
            ch = "."
            if grid[c, r, 0]:
                ch = "#"
            if grid[c, r, 1]:
                ch = "^"
            if c == 0 and r == min(player_row, GRID_ROWS - 1):
                ch = "@"
            line.append(ch)
        rows.append("".join(line))
    rows.append("=" * GRID_COLS)
    return "\n".join(rows)
