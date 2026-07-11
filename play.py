"""Roll out a trained checkpoint and report (optionally render) the result.

    python play.py --run runs/ppo_spikes_easy --level spikes_easy --algo ppo
    python play.py --run runs/ga_spikes_easy --level spikes_easy --algo ga --render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gdrl.envs import GDEnv
from gdrl.models import MLP, ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory containing latest.pt")
    parser.add_argument("--level", required=True)
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo", "ga"])
    parser.add_argument("--hidden", type=int, nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = GDEnv(args.level)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden = tuple(args.hidden) if args.hidden else ((64, 64) if args.algo == "ga" else (256, 256))

    ckpt = torch.load(Path(args.run) / "latest.pt", map_location="cpu", weights_only=True)
    if args.algo == "ppo":
        model: torch.nn.Module = ActorCritic(obs_dim, n_actions, hidden)
    else:
        model = MLP(obs_dim, n_actions, hidden)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for ep in range(args.episodes):
        obs, info = env.reset()
        total = 0.0
        with torch.no_grad():
            while True:
                x = torch.as_tensor(obs).unsqueeze(0)
                if args.algo == "ppo":
                    logits, _ = model(x)
                    action = int(logits.argmax())
                else:
                    action = int(model(x).argmax())
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if args.render:
                    print(f"\033[2J\033[H{env.render()}\n"
                          f"x={info['x']:.1f}  progress={info['progress']:.1%}")
                    time.sleep(1 / 60)
                if terminated or truncated:
                    break
        outcome = "WON" if info["won"] else ("died" if info["dead"] else "timeout")
        print(f"episode {ep}: {outcome}  progress={info['progress']:.1%}  return={total:.2f}")


if __name__ == "__main__":
    main()
