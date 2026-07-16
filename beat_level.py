"""Beat a deterministic GD level by checkpoint-based segment search on the game.

Exploits two facts: the level is deterministic, and GD practice mode lets us
drop checkpoints. From the last checkpoint we run a height-aware, geometry-
guided policy with randomized jump timing; when it advances we drop a new
checkpoint and lock that progress in; on death we respawn at the last checkpoint
and retry with different timing. Each attempt replays only a short segment and
the frontier only moves forward. Nothing transfers from the sim — the search
runs against the real game's own physics, every gamemode included.

    python beat_level.py [--stride 6] [--attempts 8000]
"""
from __future__ import annotations

import argparse
import bisect
import random
import time

from gdrl.envs.bridge import RealGameBridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=6.0, help="blocks of new progress per checkpoint")
    ap.add_argument("--horizon", type=int, default=700, help="max frames per attempt")
    ap.add_argument("--attempts", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    b = RealGameBridge(timeout=180.0)
    print("connecting ... (be in the level)", flush=True)
    b.connect()
    print("HELLO ok — enabling practice mode + reset", flush=True)
    b.send_action(hold=False, request_reset=True, request_geom=True, practice_on=True)
    s = b.recv_state()
    geom = b.last_geometry or []
    # Collidable objects sorted by x, so we can scan a window ahead and only
    # react to ones near the cube's current height (works on platforms too).
    objs = sorted((cx, cy) for k, cx, cy in geom if k in (0, 1))
    xs = [o[0] for o in objs]
    print(f"level: {len(geom)} objs, {len(objs)} collidables, length {s.length:.0f}", flush=True)

    def nearest_ahead(px, py):
        """Distance to the nearest collidable ahead at the cube's height, or None."""
        i = bisect.bisect_right(xs, px)
        while i < len(xs) and xs[i] - px <= 12.0:
            cx, cy = objs[i]
            if abs(cy - py) < 1.7:  # at the cube's level -> it would hit this
                return cx - px
            i += 1
        return None

    checkpoint_x = s.x
    has_cp = False
    best_x = s.x
    t0 = time.perf_counter()

    for attempt in range(1, args.attempts + 1):
        jump_dist = rng.uniform(1.6, 3.1)   # search jump timing across attempts
        w = 0.5
        extra_jitter = rng.random() < 0.3   # sometimes add noisy extra jumps
        for _ in range(args.horizon):
            d = nearest_ahead(s.x, s.y)
            jump = bool(s.grounded and d is not None and jump_dist - w <= d <= jump_dist + w)
            if extra_jitter and s.grounded and rng.random() < 0.04:
                jump = True
            # Lock progress in with a checkpoint once safely past the frontier.
            near = nearest_ahead(s.x, s.y)
            if (s.grounded and s.x > checkpoint_x + args.stride
                    and (near is None or near > 1.5)):
                b.send_action(hold=jump, place_checkpoint=True)
                s = b.recv_state()
                checkpoint_x, has_cp = s.x, True
                print(f"  checkpoint x={s.x:6.1f}  {s.percent*100:5.1f}%  "
                      f"(attempt {attempt}, {time.perf_counter()-t0:.0f}s)", flush=True)
                continue
            b.send_action(hold=jump)
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
