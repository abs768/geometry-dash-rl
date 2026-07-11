"""Turn a Geometry Dash level string / .gmd into a sim Level, honestly.

The importer never guesses: objects whose IDs are not in the known tables are
counted as "unknown" and left out, and gamemode portals are tallied so the
report can state how much of the level the cube-only sim can actually play.
"""

from __future__ import annotations

import plistlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gdrl.levels import gd_format, objects
from gdrl.sim.level import Level, LevelObject, Portal


@dataclass
class ImportReport:
    name: str
    length_blocks: float
    n_blocks: int
    n_spikes: int
    n_portals: int
    gamemode_portals: list[tuple[float, str]]  # (x_block, gamemode) in order
    unknown_ids: Counter = field(default_factory=Counter)
    total_objects: int = 0

    @property
    def recognized(self) -> int:
        return self.n_blocks + self.n_spikes + self.n_portals

    @property
    def coverage(self) -> float:
        return self.recognized / self.total_objects if self.total_objects else 1.0

    @property
    def cube_only(self) -> bool:
        return all(mode == "cube" for _, mode in self.gamemode_portals)

    def summary(self) -> str:
        lines = [
            f"level: {self.name}",
            f"  length:      {self.length_blocks:.1f} blocks",
            f"  blocks:      {self.n_blocks}",
            f"  spikes:      {self.n_spikes}",
            f"  portals:     {self.n_portals}",
            f"  objects:     {self.total_objects} total, "
            f"{self.coverage:.1%} recognized",
        ]
        if self.gamemode_portals:
            modes = ", ".join(f"{m}@{x:.0f}" for x, m in self.gamemode_portals)
            lines.append(f"  gamemodes:   {modes}")
            if not self.cube_only:
                lines.append("  WARNING: non-cube sections — the cube-only sim "
                             "cannot play those stretches yet")
        if self.unknown_ids:
            top = ", ".join(f"id{i}×{n}" for i, n in self.unknown_ids.most_common(8))
            lines.append(f"  unrecognized: {sum(self.unknown_ids.values())} objects "
                         f"({len(self.unknown_ids)} distinct): {top}")
        return "\n".join(lines)


def _objects_to_level(name: str, props: list[dict[int, str]]) -> tuple[Level, ImportReport]:
    level_objects: list[LevelObject] = []
    level_portals: list[Portal] = []
    gamemode_portals: list[tuple[float, str]] = []
    unknown: Counter = Counter()
    n_blocks = n_spikes = n_portals = 0
    max_x = 0.0

    for obj in props:
        oid = int(obj[1])
        x = objects.units_to_block(float(obj.get(2, 0.0)))
        y = objects.units_to_block(float(obj.get(3, 0.0)))
        max_x = max(max_x, x)
        kind = objects.classify(oid)
        if kind == objects.BLOCK:
            level_objects.append(LevelObject("block", x, max(y, 0.0)))
            n_blocks += 1
        elif kind == objects.SPIKE:
            level_objects.append(LevelObject("spike", x, max(y, 0.0)))
            n_spikes += 1
        elif kind == "portal_gamemode":
            mode = objects.PORTAL_GAMEMODE[oid]
            level_portals.append(Portal("gamemode", x, mode))
            gamemode_portals.append((x, mode))
            n_portals += 1
        elif kind == "portal_speed":
            level_portals.append(Portal("speed", x, objects.PORTAL_SPEED[oid]))
            n_portals += 1
        elif kind == "portal_gravity":
            level_portals.append(Portal("gravity", x, objects.PORTAL_GRAVITY[oid]))
            n_portals += 1
        else:
            unknown[oid] += 1

    # Level length: a little past the last object, in blocks.
    length = max_x + 15.0
    gamemode_portals.sort()
    level = Level(name, length, level_objects, portals=level_portals)
    report = ImportReport(
        name=name, length_blocks=length, n_blocks=n_blocks, n_spikes=n_spikes,
        n_portals=n_portals, gamemode_portals=gamemode_portals,
        unknown_ids=unknown, total_objects=len(props),
    )
    return level, report


def import_level_string(level_string: str, name: str = "imported",
                        official: bool = False) -> tuple[Level, ImportReport]:
    """Import from a stored (compressed) level string."""
    inner = gd_format.decode_level_string(level_string, official=official)
    return import_inner_string(inner, name=name)


def import_inner_string(inner: str, name: str = "imported") -> tuple[Level, ImportReport]:
    """Import from an already-decompressed inner level string."""
    _header, props = gd_format.parse_inner(inner)
    return _objects_to_level(name, props)


def import_gmd(path: str | Path) -> tuple[Level, ImportReport]:
    """Import a .gmd file (a plist wrapper with the level string under 'k4')."""
    path = Path(path)
    data = plistlib.loads(path.read_bytes())
    name = data.get("k2", path.stem)
    level_string = data["k4"]
    # .gmd level strings are stored compressed like in-game.
    return import_level_string(level_string, name=name)
