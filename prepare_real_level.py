"""Convert a captured real-game geometry dump into a sim-ready level.

The mod reports each object's CENTER in block units (game units / 30). Our sim
uses bottom-left corners and an implicit floor at y=0, while GD's ground line
sits at ~3.0 blocks (the cube spawns with its center at y=3.5, so its bottom is
at 3.0). So we shift everything down by the ground line and left/down by half a
block to go from center to bottom-left:

    sim_x = center_x - 0.5
    sim_y = max(0, center_y - GROUND_LINE - 0.5)   # GROUND_LINE + 0.5 = 3.5

Ground-level obstacles land at sim_y=0 (on the sim floor), the first staircase
step at y=1, and so on. Ground spikes whose object anchor sits slightly low are
clamped up onto the floor.

    python prepare_real_level.py --in levels/stereo_madness_real.json \
        --out levels/stereo_madness_cube.json [--max-x 360]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPAWN_CENTER_Y = 3.5  # observed cube spawn center; ground line = 3.5 - 0.5 = 3.0


def convert(dump: dict, max_x: float | None) -> dict:
    objs = []
    for o in dump["objects"]:
        # Objects at/behind the spawn (x<2) are start-line markers with no live
        # hitbox in-game; keeping them would kill the cube at frame 0 in the sim.
        if o["x"] < 2.0:
            continue
        sx = round(o["x"] - 0.5, 3)
        sy = round(max(0.0, o["y"] - SPAWN_CENTER_Y), 3)
        if sx < -0.5:
            continue
        if max_x is not None and sx > max_x:
            continue
        objs.append({"type": o["type"], "x": sx, "y": sy})
    length = min(dump["length"], max_x) if max_x else dump["length"]
    return {"name": dump.get("name", "real"), "length": round(length, 2),
            "ceiling": 12.0, "start_mode": "cube", "objects": objs, "portals": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="levels/stereo_madness_real.json")
    ap.add_argument("--out", default="levels/stereo_madness_cube.json")
    ap.add_argument("--max-x", type=float, default=None,
                    help="truncate past this x (cube-section only)")
    args = ap.parse_args()

    dump = json.loads(Path(args.inp).read_text())
    out = convert(dump, args.max_x)
    Path(args.out).write_text(json.dumps(out))
    nb = sum(1 for o in out["objects"] if o["type"] == "block")
    ns = sum(1 for o in out["objects"] if o["type"] == "spike")
    print(f"wrote {args.out}: {len(out['objects'])} objects "
          f"({nb} blocks, {ns} spikes), length {out['length']}")


if __name__ == "__main__":
    main()
