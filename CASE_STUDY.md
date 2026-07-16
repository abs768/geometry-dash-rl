# Teaching an AI to play the real Geometry Dash

**A reinforcement-learning + sim-to-real project.** I trained agents to play
*Geometry Dash* — not a clone, the actual retail game — by building a fast
headless simulator, comparing three RL algorithm families, transferring policies
onto the live game through a custom C++ mod, and finally completing a full
official level via learning from demonstration.

![The agent flying the Stereo Madness ship section](results/agent_clears_stereo_madness.gif)

*An agent completing the ship section of Stereo Madness on the retail game.*

---

## The problem

Geometry Dash is a deterministic, frame-perfect rhythm platformer: one binary
input (jump/hold), instant death on contact, and a level is a fixed obstacle
course. Prior hobby projects train RL agents **inside the running game**, which
caps learning at real time (60 fps) — millions of frames take days.

**My thesis:** decouple training from the game. Train in a *simulator* running
thousands of times faster than real time, then transfer the policy back to the
real game. That reframes the project around the interesting hard part —
**sim-to-real transfer** — instead of raw compute.

## What I built

```
Fast headless sim  ──►  RL agents (DQN / PPO / GA)  ──►  Geode C++ mod  ──►  retail game
   ~4,000× real-time        controlled comparison         binary socket bridge
```

- **Headless physics simulator** (`gdrl/sim`) — cube + all six other gamemodes
  (ship, ball, UFO, wave, robot, spider), slopes, gravity/speed portals.
  Measured **~244,000 steps/sec (~4,000× real time)** on one CPU core.
- **Gymnasium environment** with a structured observation (player kinematics +
  a look-ahead occupancy grid of upcoming geometry) — far more sample-efficient
  than raw pixels.
- **Three RL algorithm families** (`gdrl/agents`): DQN (double), PPO (GAE,
  vectorized envs), and a genetic algorithm — compared under identical
  conditions.
- **A C++ Geode mod** (`gd-mod/`) that hooks the real game, streams live state
  and injects input over a **length-prefixed binary socket protocol**, with a
  Python bridge (`gdrl/envs/bridge.py`).
- **Test-first bridge design**: a protocol-faithful *mock* of the game backed by
  the sim let me verify the entire Python↔game path (35 tests) before ever
  touching the retail game.

## Results

### 1. A controlled DQN vs PPO vs GA comparison — with a real finding

Same observation, reward, and levels; three seeds each.

![Algorithm comparison](results/comparison.png)

| Algorithm | seeds solved | headline |
|-----------|:------------:|----------|
| DQN (double)      | 6/6 | robust |
| Genetic Algorithm | 6/6 | robust, most sample-efficient |
| PPO               | 4/6 | **trapped in a local optimum** |

The clean result: **on-policy PPO reliably gets stuck** at the first
death-risky jump (free progress up to it, a −10 penalty for the risk), where
value-based and evolutionary methods don't — and *more compute didn't rescue it*
(6M steps). A reminder that "best algorithm" is entirely problem-dependent.

### 2. Sim-to-real transfer

A policy trained **entirely in simulation** — never touching the game during
training — drove the retail game through the mod, reading live state and jumping
real spikes. It reached ~11% before a physics-gap death. A **checkpoint-based
search** on the real game (exploiting determinism + practice-mode checkpoints)
then cleared the **entire cube section, 35%**, autonomously.

### 3. Completing the full level — learning from demonstration

Autonomous search stalls at the ship section (continuous control). So I closed
the loop with **imitation**: I built a lag-free *record mode* into the mod
(it passively streams the human's input each frame without frame-locking the
game), recorded a single human clear, and the agent **reproduced it
deterministically** on the real game — verified `REPLAY CLEARED THE LEVEL
(100%)`. Full clip:
[`results/agent_clears_stereo_madness.mp4`](results/agent_clears_stereo_madness.mp4).

From that same demonstration I trained a **behavior-cloning policy**
(`train_bc.py`) that predicts the human's jump decisions with **90.7%
validation accuracy (jump F1 0.82)** — a genuinely *learned* imitation policy.

## Honest limitations

I'd rather state these than hide them — and in interviews they're the most
credible part:

- The **guaranteed full clear is deterministic replay**, not an autonomous RL
  agent solving the level. The behavior-cloning policy *learns* the mapping but,
  like all BC on a single frame-perfect level, drifts — so replay is used for
  the exact clear.
- The **non-cube physics are approximate** (calibration against real
  trajectories is future work); robot/spider are first-pass.
- Sim-to-real transfer dies at ~11% because the sim jump arc isn't yet
  calibrated to the real game frame-for-frame.

## What I learned

- **Sim-to-real is a systems problem as much as an ML one** — a fixed physics
  timestep and a memory-safe background thread mattered as much as the policy.
- **Controlled experiments beat leaderboard-chasing**: the PPO local-optimum
  finding came from seeds and ablations, not a bigger model.
- **Test the untestable surface with a mock**: verifying the bridge against a
  sim-backed mock caught real protocol bugs before the game was ever involved,
  and the on-hardware bugs that remained (a wrong game-loop hook, a shutdown
  crash) were exactly the ones a mock can't cover.

## Tech stack

Python, PyTorch, Gymnasium, NumPy · C++20, Geode SDK (Geometry Dash 2.2) ·
POSIX sockets, a custom binary protocol · matplotlib/ffmpeg for the renders.

**Code:** the full repo, tests, and all footage are on GitHub.
