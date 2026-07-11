from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class RunLogger:
    """Prints progress and appends metrics to runs/<name>/metrics.csv."""

    def __init__(self, run_dir: str | Path, fields: list[str]):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fields = fields
        self._path = self.dir / "metrics.csv"
        with open(self._path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    def log(self, **values) -> None:
        row = [values.get(k, "") for k in self.fields]
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print("  ".join(f"{k}={values[k]:.3f}" if isinstance(values.get(k), float)
                        else f"{k}={values.get(k)}" for k in self.fields))

    def save_checkpoint(self, model: torch.nn.Module, name: str = "latest.pt", **extra) -> None:
        torch.save({"model": model.state_dict(), **extra}, self.dir / name)
