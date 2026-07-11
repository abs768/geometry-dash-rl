"""Tests for the Geometry Dash level importer.

Since real official level strings are embedded in the game (not fetchable here),
these tests exercise the parser against the genuine wire format by encoding
known objects into it and reading them back — real base64/zlib, real object
IDs, real center-coordinate (÷30 − 0.5) conversion.
"""

import math
import plistlib

from gdrl.levels import gd_format, importer, objects
from gdrl.sim.level import BLOCK, Level, SPIKE


def test_encode_decode_roundtrip():
    inner = "1,0;1,1,2,45,3,15;1,8,2,75,3,15;"
    encoded = gd_format.encode_level_string(inner)
    assert gd_format.decode_level_string(encoded) == inner


def test_center_coordinate_conversion():
    # A block on the ground at grid column 1 has center (45, 15) units.
    assert objects.units_to_block(45.0) == 1.0   # x: 45/30 - 0.5 = 1.0
    assert objects.units_to_block(15.0) == 0.0   # y: 15/30 - 0.5 = 0.0


def test_parse_inner_extracts_objects():
    inner = "1,0,kA2,0;1,1,2,45,3,15;1,8,2,75,3,15,6,90;"
    header, objs = gd_format.parse_inner(inner)
    assert len(objs) == 2
    assert objs[0][1] == "1" and objs[0][2] == "45"
    assert objs[1][1] == "8" and objs[1][6] == "90"  # rotation preserved


def test_import_classifies_blocks_and_spikes():
    # block id 1 at col 2 ground, spike id 8 at col 4 ground.
    inner = "1,0;1,1,2,75,3,15;1,8,2,135,3,15;"
    level, report = importer.import_inner_string(inner, name="demo")
    assert report.n_blocks == 1
    assert report.n_spikes == 1
    assert report.coverage == 1.0
    kinds = sorted(o.type for o in level.objects)
    assert kinds == [BLOCK, SPIKE]
    block = next(o for o in level.objects if o.type == BLOCK)
    assert math.isclose(block.x, 2.0) and math.isclose(block.y, 0.0)


def test_unknown_ids_are_reported_not_guessed():
    # id 9999 is not in any table.
    inner = "1,0;1,1,2,45,3,15;1,9999,2,75,3,15;"
    level, report = importer.import_inner_string(inner)
    assert report.n_blocks == 1
    assert report.total_objects == 2
    assert report.coverage == 0.5
    assert report.unknown_ids[9999] == 1
    # unknown object is NOT added to the level (no phantom collisions)
    assert len(level.objects) == 1


def test_gamemode_portal_detection_flags_non_cube():
    # ship portal (id 13) mid-level.
    inner = "1,0;1,1,2,45,3,15;1,13,2,300,3,45;"
    _level, report = importer.import_inner_string(inner)
    assert report.n_portals == 1
    assert not report.cube_only
    assert report.gamemode_portals[0][1] == "ship"
    assert "non-cube" in report.summary()


def test_import_gmd_file(tmp_path):
    # A .gmd is a plist with the compressed level string under key 'k4'.
    inner = "1,0;1,1,2,45,3,15;1,8,2,105,3,15;"
    gmd = {"k2": "gmd_level", "k4": gd_format.encode_level_string(inner)}
    path = tmp_path / "level.gmd"
    path.write_bytes(plistlib.dumps(gmd))

    level, report = importer.import_gmd(path)
    assert report.name == "gmd_level"
    assert report.n_blocks == 1 and report.n_spikes == 1
    assert len(level.objects) == 2


def test_roundtrip_real_sim_level_through_gd_format():
    # Encode one of our sim levels into the GD wire format and import it back;
    # the reconstructed level must have the same block/spike layout.
    original = Level.from_file("levels/blocks_and_spikes.json")
    obj_dicts = []
    for o in original.objects:
        oid = 1 if o.type == BLOCK else 8
        obj_dicts.append({1: oid, 2: (o.x + 0.5) * 30.0, 3: (o.y + 0.5) * 30.0})
    inner = gd_format.build_inner("1,0", obj_dicts)
    encoded = gd_format.encode_level_string(inner)

    imported, report = importer.import_level_string(encoded, name="roundtrip")
    assert report.total_objects == len(original.objects)
    assert report.coverage == 1.0
    assert len(imported.objects) == len(original.objects)
    for a, b in zip(sorted(original.objects, key=lambda o: (o.x, o.y, o.type)),
                    sorted(imported.objects, key=lambda o: (o.x, o.y, o.type))):
        assert a.type == b.type
        assert math.isclose(a.x, b.x, abs_tol=1e-6)
        assert math.isclose(a.y, b.y, abs_tol=1e-6)
