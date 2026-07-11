"""Import a Geometry Dash level into the sim's JSON level format.

    # from a .gmd exported by a GD tool or our own mod:
    python import_level.py --gmd path/to/level.gmd --out levels/stereo_madness.json

    # from a raw compressed level string:
    python import_level.py --string "<base64...>" --name my_level

    # from an official level string (13-byte gzip header stripped):
    python import_level.py --string "<...>" --official --name stereo_madness

    # from an already-decompressed inner string (header;obj;obj;...):
    python import_level.py --inner "1,1,2,45,3,15;..." --name demo

Prints a coverage report so you can see how much of the level the cube-only sim
actually understands before training on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdrl.levels import importer


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gmd", help="path to a .gmd file")
    src.add_argument("--string", help="stored (compressed) level string")
    src.add_argument("--inner", help="decompressed inner level string")
    parser.add_argument("--official", action="store_true",
                        help="official level string (restores stripped gzip header)")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None, help="output JSON path (default levels/<name>.json)")
    args = parser.parse_args()

    if args.gmd:
        level, report = importer.import_gmd(args.gmd)
    elif args.string:
        level, report = importer.import_level_string(
            args.string, name=args.name or "imported", official=args.official)
    else:
        level, report = importer.import_inner_string(args.inner, name=args.name or "imported")

    print(report.summary())

    name = args.name or level.name
    out = Path(args.out) if args.out else Path("levels") / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "length": level.length,
        "ceiling": level.ceiling,
        "start_mode": level.start_mode,
        "objects": [{"type": o.type, "x": o.x, "y": o.y} for o in level.objects],
        "portals": [{"kind": p.kind, "x": p.x, "value": p.value} for p in level.portals],
    }
    out.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out}  ({len(level.objects)} objects)")
    if not report.cube_only:
        print("note: contains non-cube sections the sim cannot fully play yet")


if __name__ == "__main__":
    main()
