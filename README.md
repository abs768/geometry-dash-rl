# geometry-dash-rl

Reinforcement learning agents that learn to play **Geometry Dash**, built as a
proper ML project: three algorithm families trained under identical conditions
and compared on the same benchmark.

- **DQN** (with double-DQN option) — value-based baseline, reproduces prior work
  ([geometry-dash-ai](https://github.com/ThePickleGawd/geometry-dash-ai)).
- **PPO** — on-policy actor-critic with GAE, trained on vectorized environments.
- **Genetic Algorithm** — neuroevolution of policy weights; Geometry Dash levels
  are deterministic, which is exactly the regime where GA memorization shines.

## Why this beats prior work

Prior projects train **inside the real game over a TCP socket**, so learning is
capped at real-time speed. This project trains in a **headless simulator**
(thousands of steps/sec, embarrassingly parallel) and treats the official game —
reached through a [Geode](https://geode-sdk.org) mod, see [`gd-mod/`](gd-mod/) —
as the **evaluation target**, not the training environment.

Second lever: agents observe **structured state** (player kinematics + a
look-ahead occupancy grid of upcoming geometry) instead of raw pixels, which is
dramatically more sample-efficient. A pixel-based variant is planned as an
ablation.

## Results so far

Three algorithms × three seeds × two levels, identical observation/action/reward,
greedy evaluation. Full write-up in [`results/RESULTS.md`](results/RESULTS.md).

| Algorithm | spikes_easy | blocks_and_spikes | Seeds solved |
|-----------|:-----------:|:-----------------:|:------------:|
| DQN (double)      | 100% | 100% | 6/6 |
| Genetic Algorithm | 100% | 100% | 6/6 |
| PPO               | 100% | 48%  | 4/6 |

![comparison](results/comparison.png)

Headline finding: DQN and the GA solve every seed, but **PPO gets trapped in a
local optimum** at the first death-risky jump on the harder level on 2 of 3
seeds — and more compute (6M steps) does not rescue it. A controlled multi-algorithm
comparison surfaces this; the prior single-algorithm work cannot.

## Layout

```
gdrl/sim/      headless physics simulator (cube mode, v0.1)
gdrl/envs/     Gymnasium environment (sim backend + real-game bridge stub)
gdrl/models/   policy / Q networks
gdrl/agents/   dqn.py, ppo.py, ga.py trainers
configs/       YAML recipes, one per experiment
levels/        level files (JSON block format)
tests/         physics, env-API and determinism tests
gd-mod/        Geode mod skeleton + bridge protocol spec
results/       comparison figure + RESULTS.md write-up
train.py       entry point: python train.py --config configs/ppo.yaml
sweep.py       run {dqn,ppo,ga} × seeds × levels
evaluate.py    greedy-evaluate all runs -> runs/summary.csv
plot.py        build the seeds × algorithms comparison figure
play.py        roll out a checkpoint, ASCII render
```

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # physics + env + determinism tests
python train.py --config configs/ppo.yaml
python play.py --run runs/ppo_spikes --render
```

## Environment

| | |
|---|---|
| Action | `Discrete(2)` — release / hold (jump). Matches real input exactly. |
| Observation | player `[y, vy, grounded]` + look-ahead grid (20 cols × 10 rows × {solid, hazard}), flattened → 403 floats |
| Reward | `+Δx` per step (progress in blocks), `−10` on death, `+10` on completion |
| Episode end | death, level completion, or step cap |

Physics constants live in `gdrl/sim/physics.py`. They are community-derived
approximations of cube-mode physics (~2.1-block jump apex, 10.38 blocks/s at 1×
speed) and are the calibration surface for sim-to-real transfer: the Geode mod
will log real trajectories and the constants get fitted to match.

## Roadmap

- [x] Headless sim: cube mode, blocks/spikes, deterministic step
- [x] Gymnasium env + structured observations
- [x] DQN / PPO / GA trainers + YAML recipes
- [x] Seeds × algorithms comparison sweep + figure (`results/`)
- [x] Bridge: binary protocol, `GDRealEnv`, sim-backed mock, end-to-end tests (`gd-mod/`, `eval_real.py`)
- [ ] Geode mod: build against GD + verify live game-state reads (C++ written, `VERIFY` markers in `gd-mod/src/main.cpp`)
- [ ] Physics calibration against real-game trajectories
- [ ] Ship / ball / wave gamemodes, gamemode-conditioned policy
- [ ] Official level import (GD level string parser)
- [ ] Pixel-observation ablation, W&B tracking, seeds × algorithms comparison report
