"""Replay a sim-trained policy on the real game, closed-loop over the bridge.

The policy was trained in the sim on a level built by prepare_real_level.py, so
we feed it observations in that same converted frame: object centers and the
live player position are shifted by (-0.5 x, -GROUND y) to match the sim's
bottom-left / floor-at-0 convention. The policy reacts to the *real* game state
each frame, so it corrects for physics mismatch rather than replaying blindly.

    python play_real_agent.py --run runs/ga_stereo_cube --algo ga \
        --episodes 3 [--x-offset 0.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdrl.envs import protocol as proto
from gdrl.envs.bridge import RealGameBridge, geometry_to_level
from gdrl.envs.observation import build_observation
from gdrl.models import ActorCritic, MLP
from gdrl.sim.level import Level, LevelObject

GROUND = 3.5     # real ground line (cube spawn center); maps to sim y=0
X_SHIFT = 0.5    # center -> bottom-left
HIDDEN = {"dqn": (256, 256), "ppo": (256, 256), "ga": (64, 64)}
_KIND = {0: "block", 1: "spike", 2: "slope_up", 3: "slope_down"}


def build_sim_level(records, length: float) -> Level:
    objs = []
    for kind, cx, cy in records:
        if cx < 2.0:  # spawn-line markers, as in prepare_real_level
            continue
        objs.append(LevelObject(_KIND.get(kind, "block"),
                                round(cx - X_SHIFT, 3), round(max(0.0, cy - GROUND), 3)))
    return Level("real", length, objs, ceiling=12.0)


def load_policy(run: str, algo: str, obs_dim: int):
    model = ActorCritic(obs_dim, 2, HIDDEN[algo]) if algo == "ppo" else MLP(obs_dim, 2, HIDDEN[algo])
    ckpt = torch.load(Path(run) / "latest.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--algo", required=True, choices=["dqn", "ppo", "ga"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--x-offset", type=float, default=0.0, help="extra player x shift for alignment")
    ap.add_argument("--max-frames", type=int, default=8000)
    args = ap.parse_args()

    b = RealGameBridge(timeout=180.0)
    print("connecting ... (be in the level)", flush=True)
    b.connect()
    print("HELLO ok", flush=True)
    b.send_action(hold=False, request_reset=True, request_geom=True)
    s = b.recv_state()
    level = build_sim_level(b.last_geometry or [], s.length)
    nb = sum(o.type == "block" for o in level.objects)
    nsp = sum(o.type == "spike" for o in level.objects)
    nsl = sum(o.type.startswith("slope") for o in level.objects)
    print(f"sim level from real geometry: {len(level.objects)} objs "
          f"({nb} block, {nsp} spike, {nsl} slope), length {level.length:.0f}", flush=True)

    from gdrl.envs.observation import OBS_LEN
    model = load_policy(args.run, args.algo, OBS_LEN)

    def obs_of(st):
        gravity = -1 if (st.flags & proto.FLAG_UPSIDE) else 1
        px = (st.x - X_SHIFT) + args.x_offset
        py = st.y - GROUND
        return build_observation(px, py, st.vy, st.grounded, level,
                                 mode=st.gamemode, gravity=gravity)

    def act(o):
        x = torch.as_tensor(o).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0] if args.algo == "ppo" else model(x)
        return int(logits.argmax())

    best = 0.0
    for ep in range(1, args.episodes + 1):
        b.send_action(hold=False, request_reset=True)
        s = b.recv_state()
        t0 = time.perf_counter()
        for _ in range(args.max_frames):
            b.send_action(hold=bool(act(obs_of(s))))
            s = b.recv_state()
            if s.dead or s.complete:
                break
        best = max(best, s.percent)
        dt = time.perf_counter() - t0
        print(f"ep {ep}: {'WON' if s.complete else 'died'} at {s.percent*100:5.1f}% "
              f"x={s.x:6.1f}  ({s.frame} frames, {s.frame/max(dt,1e-6):.0f} fps)", flush=True)
        if s.complete:
            break
    print(f"best: {best*100:.1f}%", flush=True)
    b.close()


if __name__ == "__main__":
    main()
