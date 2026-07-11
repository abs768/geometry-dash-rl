"""Greedy-evaluate trained runs and emit a summary table.

    python evaluate.py --runs runs/ --out runs/summary.csv

Run directories are expected to be named <algo>_<level>_s<seed> (the sweep
convention) and contain latest.pt. The env is deterministic, so one greedy
rollout per run is an exact measurement.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from gdrl.envs import GDEnv
from gdrl.models import MLP, ActorCritic

RUN_RE = re.compile(r"^(dqn|ppo|ga)_(.+)_s(\d+)$")
HIDDEN = {"dqn": (256, 256), "ppo": (256, 256), "ga": (64, 64)}


def greedy_rollout(env: GDEnv, model: torch.nn.Module, algo: str) -> dict:
    obs, info = env.reset()
    total = 0.0
    with torch.no_grad():
        while True:
            x = torch.as_tensor(obs).unsqueeze(0)
            logits = model(x)[0] if algo == "ppo" else model(x)
            obs, reward, terminated, truncated, info = env.step(int(logits.argmax()))
            total += reward
            if terminated or truncated:
                return {"return": total, "progress": info["progress"], "won": int(info["won"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="runs/summary.csv")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(Path(args.runs).iterdir()):
        m = RUN_RE.match(run_dir.name)
        ckpt_path = run_dir / "latest.pt"
        if not m or not ckpt_path.exists():
            continue
        algo, level, seed = m.group(1), m.group(2), int(m.group(3))

        env = GDEnv(level)
        obs_dim = env.observation_space.shape[0]
        model = (ActorCritic(obs_dim, 2, HIDDEN[algo]) if algo == "ppo"
                 else MLP(obs_dim, 2, HIDDEN[algo]))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()

        result = {"algo": algo, "level": level, "seed": seed, **greedy_rollout(env, model, algo)}
        rows.append(result)
        print(f"{algo:>4} {level:<18} seed {seed}: "
              f"{'WON ' if result['won'] else 'lost'} progress={result['progress']:.1%}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "level", "seed", "won", "progress", "return"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
