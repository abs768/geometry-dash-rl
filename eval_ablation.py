"""Greedy-evaluate the PPO entropy ablation runs and emit results/ppo_ablation.csv.

    python eval_ablation.py

`evaluate.py` only matches run dirs named `<algo>_<level>_s<seed>`, so the
`ablation_ppo_ent001_s*` runs it deliberately skips had no committed artifact —
the "PPO stalls at 21.6% on 4 of 5 seeds" claim lived only in prose. This writes
that number out so it can be checked.

The ablation trains PPO on `blocks_and_spikes` at the *default* entropy
coefficient (0.01); the tuned configuration in `summary.csv` uses 0.05.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from gdrl.envs import GDEnv
from gdrl.models import ActorCritic

RUN_RE = re.compile(r"^ablation_ppo_ent(\d+)_s(\d+)$")
LEVEL = "blocks_and_spikes"

# The run-name suffix is a label, not a parseable number: "ent001" is the run for
# ent_coef=0.01 (see the reproduce block in results/RESULTS.md, which invokes
# `--run-name ablation_ppo_ent001_s$s --override ... ent_coef=0.01`). The run dirs
# store only latest.pt and metrics.csv, so the coefficient cannot be recovered
# from the checkpoint — it has to come from this map. Unknown labels raise rather
# than guess, so a new ablation can't silently be written out mislabelled.
ENT_LABELS = {"001": 0.01}


def greedy_rollout(env: GDEnv, model: torch.nn.Module) -> dict:
    obs, _ = env.reset()
    total = 0.0
    with torch.no_grad():
        while True:
            logits = model(torch.as_tensor(obs).unsqueeze(0))[0]
            obs, reward, terminated, truncated, info = env.step(int(logits.argmax()))
            total += reward
            if terminated or truncated:
                return {"return": total, "progress": info["progress"], "won": int(info["won"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--level", default=LEVEL)
    ap.add_argument("--out", default="results/ppo_ablation.csv")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(Path(args.runs).iterdir()):
        m = RUN_RE.match(run_dir.name)
        ckpt = run_dir / "latest.pt"
        if not m or not ckpt.exists():
            continue
        ent, seed = m.group(1), int(m.group(2))
        if ent not in ENT_LABELS:
            raise SystemExit(
                f"unknown entropy label 'ent{ent}' in {run_dir.name}; add it to "
                "ENT_LABELS with the value used to train it rather than guessing"
            )
        ent_coef = ENT_LABELS[ent]

        env = GDEnv(args.level)
        model = ActorCritic(env.observation_space.shape[0], 2, (256, 256))
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
        model.eval()
        r = greedy_rollout(env, model)
        rows.append({"ent_coef": ent_coef, "level": args.level, "seed": seed,
                     "won": r["won"], "progress": r["progress"], "return": r["return"]})
        print(f"  ent={ent_coef} seed {seed}: won={r['won']} progress={r['progress']*100:.2f}%")

    if not rows:
        raise SystemExit("no ablation runs found")

    rows.sort(key=lambda x: (x["ent_coef"], x["seed"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ent_coef", "level", "seed", "won", "progress", "return"])
        w.writeheader()
        w.writerows(rows)
    solved = sum(r["won"] for r in rows)
    print(f"\nwrote {out} ({len(rows)} runs, {solved}/{len(rows)} solved)")


if __name__ == "__main__":
    main()
