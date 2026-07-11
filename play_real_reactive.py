"""Reactive controller that plays a real GD level live over the bridge.

Not a trained agent — a geometry-aware heuristic: read the cube's live position
from the game, look at upcoming hazards (from the mod's geometry dump), and jump
to clear them. A baseline that shows the bridge driving real gameplay.

    python play_real_reactive.py --level stereo_madness_real --attempts 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdrl.envs.bridge import RealGameBridge

REPO = Path(__file__).resolve().parent


def load_obstacles(level_name: str):
    data = json.loads((REPO / "levels" / f"{level_name}.json").read_text())
    spikes, steps = [], []
    for o in data["objects"]:
        x, y = o["x"], o["y"]
        if o["type"] == "spike" and x >= 2.0 and 2.5 <= y <= 7.0:
            spikes.append(x)
        elif o["type"] == "block" and x >= 2.0 and 4.3 <= y <= 7.5:
            steps.append(x)  # a raised block edge to hop onto
    return sorted(spikes), sorted(steps), data.get("length", 0)


def nearest_ahead(sorted_xs, x, lo, hi):
    for ox in sorted_xs:
        d = ox - x
        if d < lo:
            continue
        if d > hi:
            return False
        return True  # a hazard sits in the [lo, hi] jump window
    return False


def decide(state, spikes, steps, cfg) -> bool:
    if not state.grounded:
        return False
    # Jump slightly earlier for spikes (clear them at apex), later for steps.
    if nearest_ahead(spikes, state.x, cfg["spike_lo"], cfg["spike_hi"]):
        return True
    if nearest_ahead(steps, state.x, cfg["step_lo"], cfg["step_hi"]):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="stereo_madness_real")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=8000)
    ap.add_argument("--spike-lo", type=float, default=1.7)
    ap.add_argument("--spike-hi", type=float, default=2.7)
    ap.add_argument("--step-lo", type=float, default=1.0)
    ap.add_argument("--step-hi", type=float, default=2.2)
    args = ap.parse_args()
    cfg = {"spike_lo": args.spike_lo, "spike_hi": args.spike_hi,
           "step_lo": args.step_lo, "step_hi": args.step_hi}

    spikes, steps, length = load_obstacles(args.level)
    print(f"loaded {len(spikes)} spike-hazards, {len(steps)} step-blocks; length {length}", flush=True)

    b = RealGameBridge(timeout=30.0)
    print("connecting ... (be in the level)", flush=True)
    b.connect()
    print("HELLO ok", flush=True)

    best = 0.0
    for attempt in range(1, args.attempts + 1):
        b.send_action(hold=False, request_reset=True)
        s = b.recv_state()
        t0 = time.perf_counter()
        held_frames = 0
        for _ in range(args.max_frames):
            jump = decide(s, spikes, steps, cfg)
            # Hold the jump for a couple frames so it registers, then release.
            if jump:
                held_frames = 2
            hold = held_frames > 0
            if held_frames > 0:
                held_frames -= 1
            b.send_action(hold=hold)
            s = b.recv_state()
            if s.dead or s.complete:
                break
        dt = time.perf_counter() - t0
        best = max(best, s.percent)
        outcome = "WON!" if s.complete else "died"
        print(f"attempt {attempt}: {outcome} at {s.percent*100:5.1f}%  x={s.x:6.1f}  "
              f"({dt:.1f}s, {s.frame} frames, {s.frame/max(dt,1e-6):.0f} fps)", flush=True)
        if s.complete:
            break
    print(f"best: {best*100:.1f}%", flush=True)
    b.close()


if __name__ == "__main__":
    main()
