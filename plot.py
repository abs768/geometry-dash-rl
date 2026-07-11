"""Build the seeds x algorithms comparison figure from sweep metrics.

    python plot.py --runs runs --out results/comparison.png

Reads runs/<algo>_<level>_s<seed>/metrics.csv. Produces, per level, a
learning curve (progress vs environment steps, mean +/- std over seeds) and a
grouped bar of final greedy performance from runs/summary.csv if present.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_RE = re.compile(r"^(dqn|ppo|ga)_(.+)_s(\d+)$")
COLORS = {"dqn": "#e05252", "ppo": "#4c8fd1", "ga": "#5cb85c"}
LABELS = {"dqn": "DQN", "ppo": "PPO", "ga": "Genetic Algorithm"}


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, progress) from a metrics.csv; progress column varies by algo."""
    steps, prog = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if "step" not in row or row["step"] == "":
                continue
            steps.append(float(row["step"]))
            # DQN/PPO log 'progress'; GA logs 'best_progress'.
            prog.append(float(row.get("progress") or row.get("best_progress") or 0.0))
    return np.array(steps), np.array(prog)


def resample(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step-interpolate a run onto a common x grid (last value carried forward)."""
    if len(steps) == 0:
        return np.zeros_like(grid)
    idx = np.searchsorted(steps, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def collect(runs_dir: Path):
    # curves[level][algo] = list of (steps, progress) arrays, one per seed
    curves = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(runs_dir.iterdir()):
        m = RUN_RE.match(run_dir.name)
        metrics = run_dir / "metrics.csv"
        if not m or not metrics.exists():
            continue
        algo, level = m.group(1), m.group(2)
        curves[level][algo].append(read_curve(metrics))
    return curves


def read_summary(path: Path):
    if not path.exists():
        return None
    data = defaultdict(lambda: defaultdict(list))  # data[level][algo] = [progress,...]
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["level"]][row["algo"]].append(float(row["progress"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/comparison.png")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    curves = collect(runs_dir)
    summary = read_summary(runs_dir / "summary.csv")
    levels = sorted(curves)

    n = len(levels)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 9), squeeze=False)

    for col, level in enumerate(levels):
        # --- learning curves (top row) ---
        ax = axes[0][col]
        for algo in ["dqn", "ppo", "ga"]:
            runs = [(s, p) for s, p in curves[level].get(algo, []) if len(s)]
            if not runs:
                continue
            # Grid spans only this algorithm's real training budget (the
            # shortest seed's last step), so a shorter run is never drawn as a
            # flat line carried forward across another algorithm's longer axis.
            algo_max = min(s.max() for s, _ in runs)
            grid = np.linspace(0, algo_max, 300)
            stacked = np.vstack([resample(s, p, grid) for s, p in runs])
            mean, std = stacked.mean(0), stacked.std(0)
            ax.plot(grid, mean, color=COLORS[algo], label=LABELS[algo], linewidth=2)
            ax.fill_between(grid, mean - std, mean + std, color=COLORS[algo], alpha=0.18)
        ax.set_title(f"{level}: learning curves (mean ± std over seeds)")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("level progress")
        ax.set_ylim(0, 1.02)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        # --- final greedy performance (bottom row) ---
        ax2 = axes[1][col]
        algos = ["dqn", "ppo", "ga"]
        if summary and level in summary:
            means = [np.mean(summary[level].get(a, [0])) for a in algos]
            errs = [np.std(summary[level].get(a, [0])) for a in algos]
            bars = ax2.bar([LABELS[a] for a in algos], means,
                           yerr=errs, capsize=6, color=[COLORS[a] for a in algos])
            for b, mu in zip(bars, means):
                ax2.text(b.get_x() + b.get_width() / 2, mu + 0.02, f"{mu:.0%}",
                         ha="center", fontsize=10)
            ax2.set_title(f"{level}: final greedy progress")
            ax2.set_ylabel("progress (greedy rollout)")
            ax2.set_ylim(0, 1.1)
            ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
            ax2.grid(alpha=0.3, axis="y")
        else:
            ax2.text(0.5, 0.5, "run evaluate.py to populate summary.csv",
                     ha="center", va="center")
            ax2.axis("off")

    fig.suptitle("Geometry Dash RL: DQN vs PPO vs Genetic Algorithm", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
