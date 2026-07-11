"""Gymnasium environment over the headless simulator.

Observation (float32, shape (4 + GRID_COLS*GRID_ROWS*2,)) — built by
gdrl.envs.observation, shared with the real-game env so a sim-trained policy
transfers unchanged:
    [0] player y / OBS_Y_SCALE
    [1] player vy / OBS_VY_SCALE
    [2] grounded flag
    [3] fractional x within the current block (x - floor(x)) — without this
        the block-aligned grid aliases ~6 consecutive frames to the same
        observation and the exact jump frame is unobservable
    [4:] look-ahead occupancy grid, GRID_COLS columns starting at the
         player's current column, GRID_ROWS rows from the ground up,
         2 channels (solid, hazard), row-major, flattened.

Action: Discrete(2) — 0 release, 1 hold. Exactly the real game's input.

Reward: +Δx per step (blocks), −10 on death, +10 on completion. Progress
reward is dense and potential-shaped by construction (x only moves forward),
so it cannot be gamed by dithering.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gdrl.envs.observation import OBS_LEN, ascii_strip, build_observation
from gdrl.sim import GDSim, Level

DEATH_REWARD = -10.0
WIN_REWARD = 10.0

LEVELS_DIR = Path(__file__).resolve().parents[2] / "levels"


class GDEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, level: str | Path | Level, max_steps: int | None = None):
        if not isinstance(level, Level):
            path = Path(level)
            if not path.exists():
                path = LEVELS_DIR / f"{level}.json"
            level = Level.from_file(path)
        self.level = level
        self.sim = GDSim(level)
        # Generous cap: 3x the frames a straight run needs.
        from gdrl.sim import physics
        self.max_steps = max_steps or int(3 * level.length / (physics.SPEED_1X * physics.DT))

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_LEN,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.state = None

    # -- helpers -----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        s = self.state
        return build_observation(s.x, s.y, s.vy, s.grounded, self.level)

    def _info(self) -> dict:
        return {
            "x": self.state.x,
            "progress": self.sim.progress(self.state),
            "won": self.state.won,
            "dead": self.state.dead,
        }

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.sim.reset()
        return self._obs(), self._info()

    def step(self, action):
        prev_x = self.state.x
        self.state = self.sim.step(self.state, hold=bool(action))

        reward = self.state.x - prev_x
        terminated = False
        if self.state.dead:
            reward += DEATH_REWARD
            terminated = True
        elif self.state.won:
            reward += WIN_REWARD
            terminated = True
        truncated = not terminated and self.state.frame >= self.max_steps

        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        return ascii_strip(self.state.x, self.state.y, self.level)


def make_env(level: str, max_steps: int | None = None, **_):
    """Factory usable directly or inside gym vector envs."""
    def _thunk():
        return GDEnv(level, max_steps=max_steps)
    return _thunk
