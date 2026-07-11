"""Evaluate a sim-trained checkpoint on the real game via the bridge.

    # against a running Geometry Dash + gd-mod on the default port:
    python eval_real.py --run runs/ga_spikes_easy_s0 --algo ga --level spikes_easy

    # against the sim-backed mock (no game required), for CI / smoke testing:
    python eval_real.py --run runs/ga_spikes_easy_s0 --algo ga --level spikes_easy --mock

The policy is loaded and run exactly as in the sim; only the environment backend
changes. This is the artifact that makes results claimable "on the game".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gdrl.envs import protocol as proto
from gdrl.envs.bridge import GDRealEnv, RealGameBridge
from gdrl.models import ActorCritic, MLP

HIDDEN = {"dqn": (256, 256), "ppo": (256, 256), "ga": (64, 64)}


def load_policy(run: str, algo: str, obs_dim: int):
    model = ActorCritic(obs_dim, 2, HIDDEN[algo]) if algo == "ppo" else MLP(obs_dim, 2, HIDDEN[algo])
    ckpt = torch.load(Path(run) / "latest.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def act(model, algo, obs) -> int:
    x = torch.as_tensor(obs).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0] if algo == "ppo" else model(x)
    return int(logits.argmax())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo", "ga"])
    parser.add_argument("--level", required=True, help="level name (also drives the mock)")
    parser.add_argument("--mock", action="store_true", help="use the sim-backed mock mod")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=proto.DEFAULT_PORT)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    server = None
    if args.mock:
        from gdrl.envs.mock_mod import MockModServer
        server = MockModServer(args.level, port=0).start()
        port = server.port
        print(f"[eval_real] mock mod serving {args.level!r} on port {port}")
    else:
        port = args.port
        print(f"[eval_real] connecting to gd-mod at {args.host}:{port}")

    bridge = RealGameBridge(host=args.host, port=port, timeout=15.0)
    bridge.connect()
    env = GDRealEnv(bridge)
    obs_dim = env.observation_space.shape[0]
    model = load_policy(args.run, args.algo, obs_dim)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            total = 0.0
            while True:
                obs, reward, terminated, truncated, info = env.step(act(model, args.algo, obs))
                total += reward
                if terminated or truncated:
                    break
            outcome = "WON" if info["won"] else ("died" if info["dead"] else "timeout")
            print(f"[eval_real] episode {ep}: {outcome}  progress={info['progress']:.1%}  return={total:.2f}")
    finally:
        env.close()
        if server:
            server.stop()


if __name__ == "__main__":
    main()
