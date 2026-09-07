"""Domain-randomization robustness experiment.

Question: a policy trained on one *point estimate* of the physics is brittle to
the sim-to-real gap — if the real game's jump/speed differ even slightly, its
memorized timing breaks. Does **domain randomization** (jittering the physics
each episode during training) buy robustness to that gap?

Method:
  1. Train N seeds of DQN on the calibrated ("nominal") physics.
  2. Train N seeds of DQN with domain randomization (physics jittered +/- r per
     episode).
  3. Evaluate every policy across a sweep of *held-out* physics perturbations
     (scaling jump strength, and separately horizontal speed) and record greedy
     level progress.

Output: results/robustness.csv (per-perturbation evaluations),
results/robustness_bands.csv (the derived >=90% tolerance-band widths and the
DR-vs-nominal ratio the writeup quotes) and results/robustness.png — the robustness
curves (mean +/- std over seeds). If DR works, its curve stays high across a
much wider band of perturbations than the nominal policy's.

    python dr_experiment.py --seeds 0 1 2 3 4 --steps 150000 --randomize 0.18

Pure sim, no game needed.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gdrl.agents import train_dqn
from gdrl.envs import GDEnv
from gdrl.models import MLP
from gdrl.sim import physics

NOMINAL_COLOR = "#e05252"   # point-estimate policy
DR_COLOR = "#4c8fd1"        # domain-randomized policy


def train_one(level: str, seed: int, steps: int, randomize: float, run_dir: Path) -> None:
    cfg = dict(
        algo="dqn", level=level, seed=seed, hidden=[256, 256], lr=1e-4,
        gamma=0.99, buffer_size=100_000, batch_size=64,
        total_steps=steps, warmup_steps=2_000, eps_start=1.0, eps_end=0.05,
        eps_decay_steps=int(steps * 0.4), target_update_steps=1_000, double=True,
        checkpoint_every=steps, log_every_episodes=50, randomize=randomize,
    )
    train_dqn(cfg, str(run_dir))


def greedy_progress(model: MLP, level: str, params: physics.PhysicsParams) -> float:
    """One greedy rollout on a sim pinned to `params`; returns level progress."""
    env = GDEnv(level, params=params)
    obs, info = env.reset()
    with torch.no_grad():
        while True:
            a = int(model(torch.as_tensor(obs).unsqueeze(0)).argmax())
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                return info["progress"]


def load_policy(run_dir: Path, obs_dim: int) -> MLP:
    model = MLP(obs_dim, 2, (256, 256))
    ckpt = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--level", default="spikes_easy")
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--randomize", type=float, default=0.18, help="train-time DR half-range")
    ap.add_argument("--grid", type=float, default=0.30, help="eval perturbation half-range")
    ap.add_argument("--points", type=int, default=25)
    ap.add_argument("--out-dir", default="runs/dr")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- train both conditions ------------------------------------------------
    conditions = {"nominal": 0.0, "dr": args.randomize}
    if not args.skip_train:
        t0 = time.perf_counter()
        n = len(args.seeds) * len(conditions)
        i = 0
        for cond, r in conditions.items():
            for seed in args.seeds:
                i += 1
                run_dir = out / f"{cond}_s{seed}"
                print(f"\n[dr {i}/{n}] training {cond} seed {seed} "
                      f"(randomize={r}) -> {run_dir}")
                train_one(args.level, seed, args.steps, r, run_dir)
        print(f"\n[dr] training done in {(time.perf_counter()-t0)/60:.1f} min")

    # --- evaluate across held-out perturbations -------------------------------
    obs_dim = GDEnv(args.level).observation_space.shape[0]
    factors = np.linspace(1 - args.grid, 1 + args.grid, args.points)
    axes_def = {"jump": "jump strength", "speed": "horizontal speed"}

    rows = []  # cond, seed, axis, factor, progress
    curves = {}  # curves[axis][cond] = array (n_seeds, n_points)
    for axis in axes_def:
        curves[axis] = {}
        for cond in conditions:
            per_seed = []
            for seed in args.seeds:
                model = load_policy(out / f"{cond}_s{seed}", obs_dim)
                progs = []
                for f in factors:
                    kw = {axis: float(f)}
                    p = physics.NOMINAL.scaled(**kw)
                    prog = greedy_progress(model, args.level, p)
                    progs.append(prog)
                    rows.append(dict(cond=cond, seed=seed, axis=axis,
                                     factor=float(f), progress=float(prog)))
                per_seed.append(progs)
            curves[axis][cond] = np.array(per_seed)

    csv_path = Path("results/robustness.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cond", "seed", "axis", "factor", "progress"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} evals)")

    # --- robustness figure ----------------------------------------------------
    style = {"nominal": ("Point estimate (no DR)", NOMINAL_COLOR),
             "dr": (f"Domain randomization (±{args.randomize:.0%})", DR_COLOR)}
    fig, axs = plt.subplots(1, len(axes_def), figsize=(7.0 * len(axes_def), 5.2),
                            squeeze=False)
    for col, (axis, axis_label) in enumerate(axes_def.items()):
        ax = axs[0][col]
        # shade the DR training band
        ax.axvspan((1 - args.randomize) * 100 - 100, (1 + args.randomize) * 100 - 100,
                   color=DR_COLOR, alpha=0.07,
                   label=f"DR training band (±{args.randomize:.0%})")
        for cond in conditions:
            data = curves[axis][cond]  # (seeds, points)
            mean, std = data.mean(0), data.std(0)
            label, color = style[cond]
            x = (factors - 1) * 100
            ax.plot(x, mean, color=color, label=label, linewidth=2.2)
            ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=color, alpha=0.18)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Robustness to {axis_label} error")
        ax.set_xlabel(f"{axis_label} vs training (%)")
        ax.set_ylabel("level progress (greedy)")
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower center", fontsize=9)

    fig.suptitle("Domain randomization vs the sim-to-real gap "
                 f"({args.level}, DQN, {len(args.seeds)} seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = Path("results/robustness.png")
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")

    # Headline number: width of the band where mean progress stays >= 0.9.
    # This is the figure the writeup quotes ("~1.7x wider tolerance"), so it is
    # written to its own CSV rather than only printed — otherwise the claim has
    # no stored artifact and cannot be checked without re-running the script.
    THRESHOLD = 0.9
    widths: dict = {}
    for axis in axes_def:
        widths[axis] = {}
        for cond in conditions:
            mean = curves[axis][cond].mean(0)
            ok = factors[mean >= THRESHOLD]
            width = (ok.max() - ok.min()) * 100 if len(ok) else 0.0
            widths[axis][cond] = width
            # One decimal, not zero: rounding 12.5 to "12" here is what put a
            # slightly wrong band width into the writeup.
            print(f"  {axis:<6} {cond:<8}: >=90% progress over a {width:.1f}%-wide band")

    bands_path = Path("results/robustness_bands.csv")
    with open(bands_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "axis", "threshold", "nominal_width_pct", "dr_width_pct", "improvement_x",
            "seeds", "train_randomize", "eval_grid", "eval_points", "level",
        ])
        w.writeheader()
        for axis in axes_def:
            nom = widths[axis]["nominal"]
            dr = widths[axis]["dr"]
            w.writerow({
                "axis": axis,
                "threshold": THRESHOLD,
                "nominal_width_pct": round(nom, 1),
                "dr_width_pct": round(dr, 1),
                # Undefined rather than infinite if the point-estimate policy
                # never holds the threshold anywhere.
                "improvement_x": round(dr / nom, 2) if nom else "",
                "seeds": len(args.seeds),
                "train_randomize": args.randomize,
                "eval_grid": args.grid,
                "eval_points": args.points,
                "level": args.level,
            })
    print(f"wrote {bands_path}")


if __name__ == "__main__":
    main()
