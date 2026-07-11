# gd-mod — Geode mod for real-game evaluation

Status: **C++ written, not yet built against the game.** The mod source
(`src/`) is complete and the SDK-independent parts (`protocol.hpp`,
`socket_server.cpp`) compile cleanly with `clang++ -std=c++20`. What remains is
building against an installed Geode SDK + Geometry Dash and verifying the
game-state field reads (all marked `VERIFY` in `src/main.cpp`) against the
actual bindings.

The **entire Python side of the bridge is implemented and tested end-to-end**
without the game: `gdrl/envs/mock_mod.py` is a protocol-faithful mock backed by
the headless sim, and `tests/test_bridge.py` proves a sim-trained policy runs
through the real socket path (`RealGameBridge` → socket → mock) with
observations and rewards matching the in-process sim env to float32 precision.
So the only untested surface is the mod's reads of live game memory.

```
src/protocol.hpp       wire format (mirrors gdrl/envs/protocol.py)
src/socket_server.*    blocking length-prefixed TCP server (POSIX)
src/main.cpp           Geode hooks: state export, input inject, reset, geometry
mod.json               Geode metadata + port setting
CMakeLists.txt         setup_geode_mod build
```

Verify the Python bridge now (no game needed):

```bash
python eval_real.py --run runs/ga_spikes_easy_s0 --algo ga --level spikes_easy --mock
```

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
