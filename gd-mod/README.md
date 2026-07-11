# gd-mod — Geode mod for real-game evaluation

Status: **builds against real Geometry Dash 2.2081 (Geode 5.8.1).** The mod
compiles and packages cleanly — every game-state binding that was written blind
and marked `VERIFY` in `src/main.cpp` matched the real 2.2081 headers on the
first build. What remains is the runtime handshake: install the Geode loader
into the game, launch it, and connect the Python bridge.

### Working build recipe (macOS, Apple Silicon)

```bash
brew install cmake geode-sdk/geode/geode-cli
geode profile add --name main "/path/to/Steam/steamapps/common/Geometry Dash/Geometry Dash.app" mac
geode sdk install ~/geode-sdk         # installs SDK v5.x (supports GD 2.208x)
geode sdk install-binaries
export GEODE_SDK=~/geode-sdk
cd gd-mod && geode build              # -> build/gdrl.bridge.geode, installed to the game
```

Two gotchas hit on the first real build:

- **Version string**: `mod.json`'s `gd` version must be the **granular** build
  string the SDK targets (`2.2081`), not the marketing version (`2.208`) — the
  build errors out otherwise and prints the exact string to use.
- **Architecture**: GD 2.208 runs its **arm64** slice natively on Apple
  Silicon, but Geode's Mac default is x86_64, so the mod fails to load with
  `incompatible architecture (have 'x86_64', need 'arm64')`. The CMakeLists now
  defaults to the host arch; the loader and SDK libs are universal, so it links
  either way.

Then install the Geode loader with the official installer
(`geode-installer-v<ver>-mac.pkg` from the Geode releases; auto-detects the
Steam install), launch GD once, enter a level, and run:

```bash
python eval_real.py --run runs/<checkpoint> --algo <algo> --level <name>   # no --mock: real game
```

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
