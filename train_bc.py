"""Behavior cloning: train a policy to imitate a recorded human playthrough.

Turns the demonstration (recordings/*.json) into supervised (observation ->
jump) pairs — observations are the same look-ahead vectors the sim policies use,
rebuilt from the recorded per-frame state + the level geometry — and trains an
MLP to predict the human's input. Honest metrics: jumps are the minority class
(~23%), so we report precision/recall/F1 on jumps, not just accuracy.

    python train_bc.py --rec recordings/stereo_madness2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gdrl.envs.observation import OBS_LEN, build_observation
from gdrl.models import MLP
from gdrl.sim.level import BLOCK, Level, LevelObject, SLOPE_DOWN, SLOPE_UP, SPIKE
from gdrl.utils import set_seed

KIND = {0: BLOCK, 1: SPIKE, 2: SLOPE_UP, 3: SLOPE_DOWN}


def build_dataset(rec: dict):
    geom = [LevelObject(KIND.get(k, BLOCK), x, y) for k, x, y in rec["geometry"]]
    level = Level("demo", rec["length"], geom, ceiling=12.0)
    X, y = [], []
    for (px, py, vy, grounded, mode, upside), held in zip(rec["states"], rec["inputs"]):
        obs = build_observation(px, py, vy, bool(grounded), level,
                                mode=mode, gravity=-1 if upside else 1)
        X.append(obs)
        y.append(float(held))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="recordings/stereo_madness2.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/bc_stereo")
    args = ap.parse_args()
    set_seed(args.seed)

    rec = json.loads(Path(args.rec).read_text())
    X, y = build_dataset(rec)
    print(f"demonstration: {len(y)} frames, {int(y.sum())} jumps ({y.mean()*100:.0f}%)", flush=True)

    # Temporal-aware random split (85/15) to gauge generalization.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    cut = int(0.85 * len(y))
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])

    model = MLP(OBS_LEN, 1, (128, 128))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Weight the positive (jump) class to counter imbalance.
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr).squeeze(1)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                pred = (model(Xva).squeeze(1) > 0).float()
            tp = ((pred == 1) & (yva == 1)).sum().item()
            fp = ((pred == 1) & (yva == 0)).sum().item()
            fn = ((pred == 0) & (yva == 1)).sum().item()
            acc = (pred == yva).float().mean().item()
            prec = tp / max(tp + fp, 1)
            rec_ = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
            print(f"epoch {ep:3d}  loss {loss.item():.3f}  val acc {acc:.3f}  "
                  f"jump P/R/F1 {prec:.2f}/{rec_:.2f}/{f1:.2f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out / "latest.pt")
    metrics = {"frames": len(y), "jump_frac": float(y.mean()),
               "val_acc": acc, "jump_precision": prec, "jump_recall": rec_, "jump_f1": f1}
    (out / "bc_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved BC policy -> {out}/latest.pt", flush=True)
    print(f"final: val acc {acc:.1%}, jump F1 {f1:.2f}", flush=True)


if __name__ == "__main__":
    main()
