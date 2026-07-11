"""Train an agent from a YAML recipe.

    python train.py --config configs/ppo.yaml
    python train.py --config configs/dqn.yaml --override total_steps=5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gdrl.agents import TRAINERS
from gdrl.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--override", nargs="*", default=[],
                        help="key=value config overrides (values parsed as YAML)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    import yaml
    for item in args.override:
        key, _, value = item.partition("=")
        cfg[key] = yaml.safe_load(value)

    algo = cfg["algo"]
    if algo not in TRAINERS:
        raise SystemExit(f"unknown algo {algo!r}; choose from {sorted(TRAINERS)}")

    run_name = args.run_name or f"{algo}_{cfg['level']}"
    run_dir = Path("runs") / run_name
    print(f"[train] algo={algo} level={cfg['level']} -> {run_dir}")
    TRAINERS[algo](cfg, str(run_dir))


if __name__ == "__main__":
    main()
