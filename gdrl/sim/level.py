"""Level representation and JSON loading.

Coordinates are in block units. x grows rightward, y grows upward, y=0 is the
ground surface. An object's (x, y) is its bottom-left corner.

JSON format:
    {
      "name": "spikes_easy",
      "length": 60.0,
      "ceiling": 10.0,                       // optional, default 10
      "start_mode": "cube",                  // optional, default cube
      "objects": [
        {"type": "spike", "x": 14.0, "y": 0.0},
        {"type": "block", "x": 22.0, "y": 0.0}
      ],
      "portals": [                            // optional
        {"kind": "gamemode", "x": 30.0, "value": "ship"},
        {"kind": "gravity",  "x": 45.0, "value": -1},
        {"kind": "speed",    "x": 50.0, "value": 2.0}
      ]
    }
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

BLOCK = "block"
SPIKE = "spike"
# 45-degree floor ramps the cube rides along instead of hitting as a wall.
SLOPE_UP = "slope_up"      # surface rises left->right: (x,y) -> (x+1, y+1)
SLOPE_DOWN = "slope_down"  # surface falls left->right: (x, y+1) -> (x+1, y)
SLOPES = (SLOPE_UP, SLOPE_DOWN)

# Gamemode ids, matching gd-mod/src/protocol.hpp and the bridge protocol.
GAMEMODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider"]
MODE_ID = {name: i for i, name in enumerate(GAMEMODES)}

DEFAULT_CEILING = 10.0

# Spike death hitbox, relative to its 1x1 cell. Real GD spike hitboxes are
# much smaller than the visual triangle; these ratios approximate that.
SPIKE_HITBOX_W = 0.40
SPIKE_HITBOX_H = 0.60


@dataclass(frozen=True)
class LevelObject:
    type: str  # BLOCK or SPIKE
    x: float
    y: float

    def aabb(self) -> tuple[float, float, float, float]:
        """Collision box as (x_min, y_min, x_max, y_max)."""
        if self.type == SPIKE:
            cx = self.x + 0.5
            return (
                cx - SPIKE_HITBOX_W / 2,
                self.y,
                cx + SPIKE_HITBOX_W / 2,
                self.y + SPIKE_HITBOX_H,
            )
        return (self.x, self.y, self.x + 1.0, self.y + 1.0)


@dataclass(frozen=True)
class Portal:
    kind: str    # "gamemode" | "gravity" | "speed"
    x: float
    value: object  # gamemode name (str), gravity dir (+1/-1), or speed mult (float)


class Level:
    def __init__(self, name: str, length: float, objects: list[LevelObject],
                 portals: list[Portal] | None = None, ceiling: float = DEFAULT_CEILING,
                 start_mode: str = "cube"):
        self.name = name
        self.length = float(length)
        self.ceiling = float(ceiling)
        self.start_mode = start_mode
        # Sorted by x so the engine can query a window with bisect.
        self.objects = sorted(objects, key=lambda o: o.x)
        self._xs = [o.x for o in self.objects]
        self.portals = sorted(portals or [], key=lambda p: p.x)

    @classmethod
    def from_file(cls, path: str | Path) -> "Level":
        data = json.loads(Path(path).read_text())
        objects = [
            LevelObject(type=o["type"], x=float(o["x"]), y=float(o["y"]))
            for o in data["objects"]
        ]
        portals = [
            Portal(kind=p["kind"], x=float(p["x"]), value=p["value"])
            for p in data.get("portals", [])
        ]
        return cls(data.get("name", Path(path).stem), data["length"], objects,
                   portals=portals, ceiling=data.get("ceiling", DEFAULT_CEILING),
                   start_mode=data.get("start_mode", "cube"))

    def objects_near(self, x_min: float, x_max: float) -> list[LevelObject]:
        """Objects whose cell could intersect [x_min, x_max]."""
        lo = bisect.bisect_left(self._xs, x_min - 1.0)
        hi = bisect.bisect_right(self._xs, x_max)
        return self.objects[lo:hi]

    def portals_crossed(self, x_prev: float, x_now: float) -> list[Portal]:
        """Portals whose x lies in (x_prev, x_now]."""
        return [p for p in self.portals if x_prev < p.x <= x_now]
