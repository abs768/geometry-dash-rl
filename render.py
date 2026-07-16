"""Render a trained policy's rollout to a shareable GIF.

The simulator is headless, so this is how you *watch* an agent play: it runs a
greedy rollout, then draws each frame (cube, blocks, spikes, slopes, a
camera that follows the player, and a progress HUD) and writes an animated GIF.

    python render.py --run runs/ga_stereo_open --algo ga \
        --level stereo_madness_open --out results/stereo_open.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from gdrl.envs import GDEnv
from gdrl.models import ActorCritic, MLP
from gdrl.sim.level import BLOCK, SLOPE_DOWN, SLOPE_UP, SPIKE

HIDDEN = {"dqn": (256, 256), "ppo": (256, 256), "ga": (64, 64)}

# Colors (dark theme, Geometry-Dash-ish).
BG = "#12121c"
GROUND_C = "#2b2b3d"
BLOCK_C = "#3a6ea5"
SPIKE_C = "#e05252"
SLOPE_C = "#4c8fd1"
CUBE_C = "#5ce08a"
CUBE_DEAD_C = "#e0e05c"


def load_policy(run, algo, obs_dim):
    model = ActorCritic(obs_dim, 2, HIDDEN[algo]) if algo == "ppo" else MLP(obs_dim, 2, HIDDEN[algo])
    ckpt = torch.load(Path(run) / "latest.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def rollout(env, model, algo, max_steps):
    obs, _ = env.reset()
    frames = [(env.state.x, env.state.y, env.state.mode, False, False)]
    for _ in range(max_steps):
        x = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0] if algo == "ppo" else model(x)
        obs, _, term, trunc, info = env.step(int(logits.argmax()))
        s = env.state
        frames.append((s.x, s.y, s.mode, s.dead, s.won))
        if term or trunc:
            break
    return frames, info


def draw_static(ax, level):
    """Draw the level geometry once (blocks/spikes/slopes)."""
    for o in level.objects:
        if o.type == BLOCK:
            ax.add_patch(mpatches.Rectangle((o.x, o.y), 1, 1, color=BLOCK_C, zorder=2))
        elif o.type == SPIKE:
            ax.add_patch(mpatches.Polygon([(o.x, o.y), (o.x + 1, o.y), (o.x + 0.5, o.y + 1)],
                                          color=SPIKE_C, zorder=2))
        elif o.type == SLOPE_UP:
            ax.add_patch(mpatches.Polygon([(o.x, o.y), (o.x + 1, o.y), (o.x + 1, o.y + 1)],
                                          color=SLOPE_C, zorder=2))
        elif o.type == SLOPE_DOWN:
            ax.add_patch(mpatches.Polygon([(o.x, o.y), (o.x + 1, o.y), (o.x, o.y + 1)],
                                          color=SLOPE_C, zorder=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--algo", required=True, choices=["dqn", "ppo", "ga"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth sim frame")
    ap.add_argument("--window", type=float, default=24.0, help="camera width in blocks")
    ap.add_argument("--max-steps", type=int, default=6000)
    args = ap.parse_args()

    env = GDEnv(args.level)
    model = load_policy(args.run, args.algo, env.observation_space.shape[0])
    frames, info = rollout(env, model, args.algo, args.max_steps)
    outcome = "WON" if info["won"] else ("died" if info["dead"] else "timeout")
    print(f"rollout: {outcome} at {info['progress']:.1%} ({len(frames)} frames)")

    ceiling = env.level.ceiling
    max_obj_y = max((o.y for o in env.level.objects), default=3.0)
    view_top = min(ceiling + 1, max_obj_y + 4.0)
    view_h = view_top + 1.5  # ylim is (-1.5, view_top)
    fig, ax = plt.subplots(figsize=(11, 11 * view_h / args.window))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    draw_static(ax, env.level)
    ax.axhline(0, color=GROUND_C, lw=3, zorder=1)
    cube = mpatches.Rectangle((0, 0), 1, 1, color=CUBE_C, zorder=5)
    ax.add_patch(cube)
    hud = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white",
                  fontsize=12, va="top", family="monospace")
    ax.set_ylim(-1.5, view_top)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    shown = list(range(0, len(frames), args.stride))
    if shown[-1] != len(frames) - 1:
        shown.append(len(frames) - 1)

    out = Path(args.out) if args.out else Path("results") / f"{Path(args.run).name}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    title = f"{args.algo.upper()} on {args.level}"

    with writer.saving(fig, str(out), dpi=100):
        for i in shown:
            x, y, mode, dead, won = frames[i]
            cube.set_xy((x, y))
            cube.set_color(CUBE_DEAD_C if dead else CUBE_C)
            ax.set_xlim(x - 5, x - 5 + args.window)
            pct = min(x / env.level.length, 1.0)
            hud.set_text(f"{title}\nprogress {pct*100:4.1f}%"
                         + ("   COMPLETE!" if won else ("   DEAD" if dead else "")))
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out} ({len(shown)} rendered frames)")


if __name__ == "__main__":
    main()
