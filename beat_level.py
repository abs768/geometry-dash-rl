"""Beat a deterministic GD level by checkpoint-based segment search on the game.

Exploits two facts: the level is deterministic, and GD practice mode lets us
drop checkpoints. From the last checkpoint we run a geometry-guided policy with
randomized inputs; when it advances we drop a new checkpoint and lock progress
in; on death we respawn at the last checkpoint and retry. Each attempt replays
only a short segment and the frontier only moves forward. Nothing transfers
from the sim — the search runs on the real game's own physics.

Cube mode: height-aware jump timing. Flight modes (ship/ufo/wave): follow a
randomized target altitude with noise. --resume continues from the game's
current last checkpoint instead of restarting from the level start.

    python beat_level.py [--stride 6] [--attempts 20000] [--resume]
"""
from __future__ import annotations

import argparse
import bisect
import random
import time

from gdrl.envs.bridge import RealGameBridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=6.0)
    ap.add_argument("--horizon", type=int, default=700)
    ap.add_argument("--attempts", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    b = RealGameBridge(timeout=180.0)
    print("connecting ...", flush=True)
    b.connect()
    if args.resume:
        print("resuming from last checkpoint", flush=True)
        b.send_action(hold=False, load_checkpoint=True, request_geom=True, practice_on=True)
    else:
        print("practice mode + reset from start", flush=True)
        b.send_action(hold=False, request_reset=True, request_geom=True, practice_on=True)
    s = b.recv_state()
    geom = b.last_geometry or []
    objs = sorted((cx, cy) for k, cx, cy in geom if k in (0, 1))
    xs = [o[0] for o in objs]
    print(f"level: {len(objs)} collidables, length {s.length:.0f}, start x={s.x:.0f} "
          f"({s.percent*100:.1f}%), mode={s.gamemode}", flush=True)

    def nearest_ahead(px, py):
        i = bisect.bisect_right(xs, px)
        while i < len(xs) and xs[i] - px <= 12.0:
            cx, cy = objs[i]
            if abs(cy - py) < 1.7:
                return cx - px
            i += 1
        return None

    checkpoint_x = s.x
    has_cp = args.resume
    best_x = s.x
    t0 = time.perf_counter()

    for attempt in range(1, args.attempts + 1):
        jd = rng.uniform(1.6, 3.1)          # cube: jump-trigger distance
        target = rng.uniform(5.5, 10.5)     # flight: target altitude
        flip = rng.uniform(0.06, 0.20)      # flight: input-noise rate
        jitter = rng.random() < 0.3
        for _ in range(args.horizon):
            if s.gamemode == 0:  # cube
                d = nearest_ahead(s.x, s.y)
                act = bool(s.grounded and d is not None and jd - 0.5 <= d <= jd + 0.5)
                if jitter and s.grounded and rng.random() < 0.04:
                    act = True
            else:                # ship / ufo / wave: hold toward a target altitude
                act = s.y < target
                if rng.random() < flip:
                    act = not act

            past = s.x > checkpoint_x + args.stride
            if s.gamemode == 0:
                near = nearest_ahead(s.x, s.y)
                safe = s.grounded and (near is None or near > 1.5)
            else:
                safe = 5.0 < s.y < 11.0
            if past and safe:
                b.send_action(hold=act, place_checkpoint=True)
                s = b.recv_state()
                checkpoint_x, has_cp = s.x, True
                print(f"  checkpoint x={s.x:6.1f}  {s.percent*100:5.1f}%  mode={s.gamemode}  "
                      f"(attempt {attempt}, {time.perf_counter()-t0:.0f}s)", flush=True)
                continue
            b.send_action(hold=act)
            s = b.recv_state()
            best_x = max(best_x, s.x)
            if s.complete or s.dead:
                break

        if s.complete:
            print(f"\n*** LEVEL COMPLETE in {attempt} attempts, "
                  f"{time.perf_counter()-t0:.0f}s ***", flush=True)
            break
        if has_cp:
            b.send_action(hold=False, load_checkpoint=True)
        else:
            b.send_action(hold=False, request_reset=True, practice_on=True)
        s = b.recv_state()
        if attempt % 50 == 0:
            print(f"  attempt {attempt}: frontier {checkpoint_x:.0f} "
                  f"({checkpoint_x/s.length*100:.1f}%), best {best_x:.0f}", flush=True)

    b.close()


if __name__ == "__main__":
    main()
