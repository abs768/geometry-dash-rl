"""Capture Stereo Madness real geometry over the bridge and save + analyze it."""
import json
import sys
from collections import Counter
sys.path.insert(0, "/Users/bhavanishankar/Downloads/github/geometry-dash-rl")
from gdrl.envs.bridge import RealGameBridge

b = RealGameBridge(timeout=180.0)
print("connecting ...", flush=True)
b.connect()
print("HELLO ok. Requesting geometry — make sure you're in Stereo Madness ...", flush=True)
b.send_action(hold=False, request_reset=True, request_geom=True)
s = b.recv_state()
geo = b.last_geometry or []
b.close()

blocks = [(x, y) for k, x, y in geo if k == 0]
spikes = [(x, y) for k, x, y in geo if k == 1]
print(f"captured {len(geo)} objects: {len(blocks)} blocks, {len(spikes)} spikes", flush=True)
print(f"level length: {s.length:.1f} blocks", flush=True)
print(f"spawn state: x={s.x:.2f} y={s.y:.2f}", flush=True)

# Distribution diagnostics: are block y-values clean grid rows, or noisy?
by = Counter(round(y) for _, y in blocks)
sy = Counter(round(y) for _, y in spikes)
print("block y-rows (round):", dict(sorted(by.items())[:12]), flush=True)
print("spike y-rows (round):", dict(sorted(sy.items())[:12]), flush=True)
xs = [x for x, _ in blocks + spikes]
print(f"x-range: {min(xs):.1f} .. {max(xs):.1f}", flush=True)

out = {
    "name": "stereo_madness_real",
    "length": round(s.length, 2),
    "objects": [{"type": "block" if k == 0 else "spike", "x": round(x, 3), "y": round(y, 3)}
                for k, x, y in geo],
    "portals": [],
}
path = "/Users/bhavanishankar/Downloads/github/geometry-dash-rl/levels/stereo_madness_real.json"
with open(path, "w") as f:
    json.dump(out, f)
print(f"saved -> {path}", flush=True)
