"""Run the full comparison sweep: {dqn, ppo, ga} x {seeds} x {levels}.

    python sweep.py --seeds 0 1 2 --levels spikes_easy blocks_and_spikes

Each run writes to runs/<algo>_<level>_s<seed>/ (the naming convention
evaluate.py and plot.py expect). Runs are sequential and deterministic; the
sim is fast enough that the whole sweep finishes in minutes on one core.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from gdrl.agents import TRAINERS
from gdrl.utils import load_config

ALGOS = ["dqn", "ppo", "ga"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--levels", nargs="+", default=["spikes_easy", "blocks_and_spikes"])
    parser.add_argument("--algos", nargs="+", default=ALGOS)
    args = parser.parse_args()

    jobs = [(a, lv, s) for a in args.algos for lv in args.levels for s in args.seeds]
    print(f"[sweep] {len(jobs)} runs: {args.algos} x {args.levels} x seeds {args.seeds}")

    for i, (algo, level, seed) in enumerate(jobs, 1):
        cfg = load_config(f"configs/{algo}.yaml")
        cfg["level"] = level
        cfg["seed"] = seed
        run_name = f"{algo}_{level}_s{seed}"
        run_dir = Path("runs") / run_name
        t0 = time.perf_counter()
        print(f"\n[sweep {i}/{len(jobs)}] {run_name}")
        TRAINERS[algo](cfg, str(run_dir))
        print(f"[sweep {i}/{len(jobs)}] {run_name} done in {time.perf_counter()-t0:.1f}s")

    print("\n[sweep] complete")


if __name__ == "__main__":
    main()
