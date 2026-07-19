"""Build the seeds x algorithms comparison figure from sweep metrics.

    python plot.py --runs runs --out results/comparison.png

Reads runs/<algo>_<level>_s<seed>/metrics.csv. Per level it draws:
  * a learning curve — faint per-seed traces, the mean, and a 95% bootstrap
    confidence band over seeds (so PPO's seed-dependent split is visible, not
    averaged away); and
  * a grouped bar of final greedy performance from runs/summary.csv, with 95%
    bootstrap CI error bars and the individual seeds overlaid as dots.
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
RNG = np.random.default_rng(0)


def bootstrap_ci(samples: np.ndarray, axis: int = 0, reps: int = 2000,
                 alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap CI over the sampling axis (resamples seeds)."""
    n = samples.shape[axis]
    if n <= 1:
        m = samples.mean(axis)
        return m, m
    idx = RNG.integers(0, n, size=(reps, n))
    boot = np.take(samples, idx, axis=axis).mean(axis=axis + 1)
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, progress) from a metrics.csv; progress column varies by algo."""
    steps, prog = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if "step" not in row or row["step"] == "":
                continue
            steps.append(float(row["step"]))
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
    data = defaultdict(lambda: defaultdict(list))
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

    # seed count (for the caption) = max seeds any cell has
    n_seeds = max((len(v) for lv in curves.values() for v in lv.values()), default=0)

    n = len(levels)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 9), squeeze=False)

    for col, level in enumerate(levels):
        # --- learning curves (top row) ---
        ax = axes[0][col]
        for algo in ["dqn", "ppo", "ga"]:
            runs = [(s, p) for s, p in curves[level].get(algo, []) if len(s)]
            if not runs:
                continue
            algo_max = min(s.max() for s, _ in runs)
            grid = np.linspace(0, algo_max, 300)
            stacked = np.vstack([resample(s, p, grid) for s, p in runs])
            mean = stacked.mean(0)
            lo, hi = bootstrap_ci(stacked, axis=0)
            # faint individual seeds — shows PPO's split instead of hiding it
            for row in stacked:
                ax.plot(grid, row, color=COLORS[algo], linewidth=0.7, alpha=0.22)
            ax.plot(grid, mean, color=COLORS[algo], label=LABELS[algo], linewidth=2.2)
            ax.fill_between(grid, lo, hi, color=COLORS[algo], alpha=0.16)
        ax.set_title(f"{level}: learning curves (mean, 95% bootstrap CI, n={n_seeds})")
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
            means, los, his = [], [], []
            for a in algos:
                vals = np.array(summary[level].get(a, [0.0]))
                means.append(vals.mean())
                lo, hi = bootstrap_ci(vals, axis=0)
                los.append(vals.mean() - lo)
                his.append(hi - vals.mean())
            x = np.arange(len(algos))
            bars = ax2.bar(x, means, yerr=[los, his], capsize=6,
                           color=[COLORS[a] for a in algos])
            # overlay individual seeds as dots
            for xi, a in zip(x, algos):
                vals = summary[level].get(a, [])
                jitter = (RNG.random(len(vals)) - 0.5) * 0.22
                ax2.scatter(xi + jitter, vals, color="black", s=18, zorder=3, alpha=0.75)
            for b, mu in zip(bars, means):
                ax2.text(b.get_x() + b.get_width() / 2, min(mu + 0.05, 1.05), f"{mu:.0%}",
                         ha="center", fontsize=10)
            ax2.set_xticks(x, [LABELS[a] for a in algos])
            ax2.set_title(f"{level}: final greedy progress (mean, 95% CI, seeds shown)")
            ax2.set_ylabel("progress (greedy rollout)")
            ax2.set_ylim(0, 1.12)
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
    print(f"wrote {out} (seeds={n_seeds})")


if __name__ == "__main__":
    main()
