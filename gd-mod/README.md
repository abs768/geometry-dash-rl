# gd-mod — Geode mod for real-game evaluation

Status: **spec + skeleton, not yet implemented.** The Python side trains in
`gdrl/sim`; this mod makes the official game a drop-in evaluation backend for
`gdrl/envs/bridge.py`.

## What it must do

1. **State export** — every frame, read player x/y, y-velocity, grounded flag,
   gamemode, and death/completion events from `PlayLayer`/`PlayerObject`.
2. **Action injection** — apply hold/release before the frame's input is
   processed (equivalent to `PlayerObject::pushButton` / `releaseButton`).
3. **Frame lock** — hook the update loop so the game *waits for the agent's
   action each frame*. This is what makes evaluation deterministic and lets a
   speedhack multiplier run faster than real time.
4. **Instant reset** — restart the attempt programmatically
   (`PlayLayer::resetLevel`), no menu navigation.
5. **Trajectory logging** — dump per-frame kinematics to JSON so
   `gdrl/sim/physics.py` constants can be calibrated against ground truth.

## Why frame lock instead of free-running

Prior art (geometry-dash-ai) streams state over TCP while the game free-runs:
the agent's action arrives a variable number of frames late, so training is
non-deterministic and capped at real-time. Blocking the game loop on the
agent's reply removes both problems at the cost of requiring fast agent
inference — which an MLP forward pass trivially satisfies.

## Build (once implemented)

```bash
brew install geode-sdk/geode/geode-cli   # macOS; Geode also supports Windows
geode config setup
geode sdk install
geode build
```

Wire protocol: see [PROTOCOL.md](PROTOCOL.md).
