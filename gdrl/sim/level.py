"""Level representation and JSON loading.

Coordinates are in block units. x grows rightward, y grows upward, y=0 is the
ground surface. An object's (x, y) is its bottom-left corner.

JSON format:
    {
      "name": "spikes_easy",
      "length": 60.0,
      "objects": [
        {"type": "spike", "x": 14.0, "y": 0.0},
        {"type": "block", "x": 22.0, "y": 0.0}
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


class Level:
    def __init__(self, name: str, length: float, objects: list[LevelObject]):
        self.name = name
        self.length = float(length)
        # Sorted by x so the engine can query a window with bisect.
        self.objects = sorted(objects, key=lambda o: o.x)
        self._xs = [o.x for o in self.objects]

    @classmethod
    def from_file(cls, path: str | Path) -> "Level":
        data = json.loads(Path(path).read_text())
        objects = [
            LevelObject(type=o["type"], x=float(o["x"]), y=float(o["y"]))
            for o in data["objects"]
        ]
        return cls(data.get("name", Path(path).stem), data["length"], objects)

    def objects_near(self, x_min: float, x_max: float) -> list[LevelObject]:
        """Objects whose cell could intersect [x_min, x_max]."""
        lo = bisect.bisect_left(self._xs, x_min - 1.0)
        hi = bisect.bisect_right(self._xs, x_max)
        return self.objects[lo:hi]
