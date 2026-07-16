"""Beat a deterministic GD level by checkpoint-based segment search on the game.

Exploits two facts: the level is deterministic, and GD practice mode lets us
drop checkpoints. From the last checkpoint we run a geometry-guided policy with
randomized jump timing; when it advances safely we drop a new checkpoint and
lock that progress in; on death we respawn at the last checkpoint and retry
with different timing. So each attempt only replays a short segment, and the
frontier only moves forward. Nothing has to transfer from the sim — the search
runs against the real game's own physics, every gamemode included.

    python beat_level.py [--stride 12] [--attempts 4000]
"""
from __future__ import annotations

import argparse
import random
import time

from gdrl.envs import protocol as proto
from gdrl.envs.bridge import RealGameBridge


def nearest_ahead(xs, x, lo, hi):
    for ox in xs:
        d = ox - x
        if d < lo:
            continue
        return d <= hi
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=12.0, help="blocks of new progress per checkpoint")
    ap.add_argument("--safe", type=float, default=3.0, help="no obstacle within this to checkpoint")
    ap.add_argument("--horizon", type=int, default=600, help="max frames per attempt")
    ap.add_argument("--attempts", type=int, default=5000)
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
    # Obstacles the ground cube must clear: spikes, plus ground-level blocks.
    obstacles = sorted(cx for k, cx, cy in geom if k == 1 or (k == 0 and 3.0 <= cy <= 6.0))
    print(f"level: {len(geom)} objs, {len(obstacles)} ground obstacles, length {s.length:.0f}", flush=True)

    checkpoint_x = s.x
    has_cp = False
    best_x = s.x
    t0 = time.perf_counter()

    for attempt in range(1, args.attempts + 1):
        jump_dist = 2.2 + rng.uniform(-0.7, 0.7)  # perturb timing each attempt
        lo, hi = jump_dist - 0.45, jump_dist + 0.45
        for _ in range(args.horizon):
            jump = bool(s.grounded and nearest_ahead(obstacles, s.x, lo, hi))
            # Lock in progress with a checkpoint when safely past the frontier.
            if (s.grounded and s.x > checkpoint_x + args.stride
                    and not nearest_ahead(obstacles, s.x, -0.5, args.safe)):
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
        # Respawn: at the last checkpoint if we have one, else from the start.
        if has_cp:
            b.send_action(hold=False, load_checkpoint=True)
        else:
            b.send_action(hold=False, request_reset=True, practice_on=True)
        s = b.recv_state()
        if attempt % 25 == 0:
            print(f"  attempt {attempt}: frontier {checkpoint_x:.0f}, best {best_x:.0f} "
                  f"({best_x/s.length*100:.1f}%)", flush=True)

    b.close()


if __name__ == "__main__":
    main()
