"""Measure headless-simulator throughput, in steps/sec and × real time.

    python bench_sim.py                      # both sweep levels
    python bench_sim.py --steps 2000000      # longer run

The README and case study both claim a speedup figure for the sim; this is the
script that produces it, so the number in the docs is measured rather than
recalled. Results are written to results/sim_speed.md.

`× real time` is steps_per_sec / 60, because one sim step advances the world by
`physics.DT = 1/60 s` — the same 60 fps the retail game is frame-locked to. So a
throughput of 60 steps/sec would be exactly 1× real time.

Measures the environment step loop only (a fixed action, no policy forward pass),
which is the quantity the "trains N× faster than the game" claim is about. Real
training is slower because it also runs the network and the optimizer.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

from gdrl.envs import GDEnv
from gdrl.sim import physics


def bench(level: str, steps: int, warmup: int = 10_000) -> dict:
    env = GDEnv(level)
    obs, _ = env.reset()

    # Warm up so import/JIT/alloc costs don't land inside the timed window.
    for _ in range(warmup):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            env.reset()

    env.reset()
    resets = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            env.reset()
            resets += 1
    elapsed = time.perf_counter() - t0

    sps = steps / elapsed
    return {
        "level": level,
        "steps": steps,
        "seconds": elapsed,
        "steps_per_sec": sps,
        "x_real_time": sps * physics.DT,  # DT = 1/60, so this is sps/60
        "episodes": resets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--levels", nargs="+", default=["spikes_easy", "blocks_and_spikes"])
    ap.add_argument("--out", default="results/sim_speed.md")
    args = ap.parse_args()

    rows = [bench(lv, args.steps) for lv in args.levels]

    L = ["# Simulator throughput", ""]
    L.append(f"- **Machine:** {platform.platform()} / {platform.machine()}")
    L.append(f"- **Python:** {platform.python_version()}")
    L.append(f"- **Steps timed per level:** {args.steps:,} (plus 10,000 warm-up)")
    L.append(f"- **Sim timestep:** DT = 1/{int(1 / physics.DT)} s, so x real time = steps/sec / "
             f"{int(1 / physics.DT)}")
    L.append("")
    L.append("Single process, one core, fixed action, no policy forward pass.")
    L.append("")
    L.append("| Level | Steps/sec | x real time | Episodes |")
    L.append("|---|---:|---:|---:|")
    for r in rows:
        L.append(f"| `{r['level']}` | {r['steps_per_sec']:,.0f} | {r['x_real_time']:,.0f}x | "
                 f"{r['episodes']:,} |")
    L.append("")
    lo = min(r["steps_per_sec"] for r in rows)
    hi = max(r["steps_per_sec"] for r in rows)
    L.append(f"Range across levels: **{lo:,.0f}-{hi:,.0f} steps/sec** "
             f"(**{lo * physics.DT:,.0f}x-{hi * physics.DT:,.0f}x** real time).")
    L.append("")
    L.append("Reproduce: `python bench_sim.py`")
    report = "\n".join(L)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
