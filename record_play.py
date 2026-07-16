"""Record a human playthrough over the bridge, then replay it on the game.

Record mode: the mod stops injecting input and instead reports YOUR live jump
input each frame (via handleButton). We log your inputs + game state until you
clear the level, then save. Because the level is deterministic, replaying that
exact input sequence reproduces your clear every time — including the ship
section, because you flew it. The same log also trains a behavior-cloning
policy (see train_bc.py).

    python record_play.py --out recordings/stereo_madness.json   # play until you clear it
    python record_play.py --replay recordings/stereo_madness.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdrl.envs import protocol as proto
from gdrl.envs.bridge import RealGameBridge


def record(b, out_path):
    # Enter passive record mode: the game runs natively (no frame-lock) and
    # streams state; we only receive. One command starts it.
    b.send_action(hold=False, request_reset=True, request_geom=True, record=True)
    s = b.recv_state()
    geom = b.last_geometry or []
    length = s.length
    print("RECORDING at native speed — play the level. I save the run that clears it.", flush=True)

    inputs, states = [], []
    prev_frame = s.frame
    best = 0.0
    attempt = 1
    while True:
        s = b.recv_state()   # passive: just receive the stream, game runs itself
        if s.frame < prev_frame:  # frame counter reset -> a new attempt started
            print(f"  attempt {attempt}: reached {best*100:.1f}% — new attempt", flush=True)
            attempt += 1
            inputs, states, best = [], [], 0.0
        prev_frame = s.frame
        best = max(best, s.percent)
        inputs.append(int(s.input_held))
        states.append([round(s.x, 3), round(s.y, 3), round(s.vy, 3),
                       int(s.grounded), s.gamemode, int(bool(s.flags & proto.FLAG_UPSIDE))])
        if s.complete:
            data = {"length": length, "inputs": inputs, "states": states,
                    "geometry": [[k, round(x, 3), round(y, 3)] for k, x, y in geom]}
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(data))
            print(f"\n*** CLEARED! saved {len(inputs)}-frame run to {out_path} ***", flush=True)
            return


def replay(b, in_path):
    data = json.loads(Path(in_path).read_text())
    inputs = data["inputs"]
    print(f"replaying {len(inputs)} recorded frames ...", flush=True)
    b.send_action(hold=False, request_reset=True)
    s = b.recv_state()
    for held in inputs:
        b.send_action(hold=bool(held))
        s = b.recv_state()
        if s.complete:
            print(f"*** REPLAY CLEARED THE LEVEL ({s.percent*100:.0f}%) ***", flush=True)
            return
        if s.dead:
            print(f"replay diverged: died at {s.percent*100:.1f}% (x={s.x:.0f})", flush=True)
            return
    print(f"replay ended at {s.percent*100:.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recordings/stereo_madness.json")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    b = RealGameBridge(timeout=600.0)  # long: waits while you play
    print("connecting ... (be in the level)", flush=True)
    b.connect()
    print("HELLO ok", flush=True)
    try:
        if args.replay:
            replay(b, args.replay)
        else:
            record(b, args.out)
    finally:
        b.close()


if __name__ == "__main__":
    main()
