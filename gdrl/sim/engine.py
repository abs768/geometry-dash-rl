"""Deterministic headless simulator for cube mode.

The whole point of this module is wall-clock speed: no rendering, no sockets,
plain float math, so PPO/GA can take thousands of steps per second per core.
Determinism is load-bearing (tests assert it, GA relies on it): step() must be
a pure function of (state, action, level).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from gdrl.sim import physics
from gdrl.sim.level import BLOCK, SPIKE, Level


@dataclass(frozen=True)
class SimState:
    x: float
    y: float  # bottom of the player box
    vy: float
    grounded: bool
    dead: bool
    won: bool
    frame: int


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class GDSim:
    """One level, one player, fixed-timestep cube physics."""

    def __init__(self, level: Level, spawn_x: float = 0.0, spawn_y: float = 0.0):
        self.level = level
        self.spawn_x = spawn_x
        self.spawn_y = spawn_y

    def reset(self) -> SimState:
        return SimState(
            x=self.spawn_x, y=self.spawn_y, vy=0.0,
            grounded=True, dead=False, won=False, frame=0,
        )

    def step(self, state: SimState, hold: bool) -> SimState:
        if state.dead or state.won:
            return state

        vy = state.vy
        grounded = state.grounded

        # Holding while grounded jumps; in real GD a held button re-jumps on
        # the same frame the cube lands, which this reproduces.
        if hold and grounded:
            vy = physics.JUMP_VELOCITY
            grounded = False

        # Integrate (semi-implicit Euler).
        if not grounded:
            vy = max(vy - physics.GRAVITY * physics.DT, physics.TERMINAL_VELOCITY)
        prev_bottom = state.y
        x = state.x + physics.SPEED_1X * physics.DT
        y = state.y + vy * physics.DT

        # Ground plane.
        if y <= 0.0:
            y, vy, grounded = 0.0, 0.0, True

        size = physics.PLAYER_SIZE
        player = (x, y, x + size, y + size)
        dead = False
        landed_on_block = False

        for obj in self.level.objects_near(x - 1.0, x + size + 1.0):
            box = obj.aabb()
            if not _overlaps(player, box):
                continue
            if obj.type == SPIKE:
                dead = True
                break
            if obj.type == BLOCK:
                block_top = box[3]
                falling = vy <= 0.0
                came_from_above = prev_bottom >= block_top - physics.LANDING_TOLERANCE
                if falling and came_from_above:
                    y, vy, grounded = block_top, 0.0, True
                    player = (x, y, x + size, y + size)
                    landed_on_block = True
                else:
                    # Side or bottom contact kills the cube.
                    dead = True
                    break

        # Walking off a block edge: airborne until gravity brings us down.
        if grounded and y > 0.0 and not landed_on_block:
            still_supported = any(
                obj.type == BLOCK
                and abs(obj.aabb()[3] - y) < 1e-9
                and obj.aabb()[0] < x + size
                and obj.aabb()[2] > x
                for obj in self.level.objects_near(x - 1.0, x + size + 1.0)
            )
            if not still_supported:
                grounded = False

        won = not dead and x >= self.level.length

        return SimState(
            x=x, y=y, vy=vy, grounded=grounded,
            dead=dead, won=won, frame=state.frame + 1,
        )

    def progress(self, state: SimState) -> float:
        """Fraction of the level completed, in [0, 1]."""
        return min(state.x / self.level.length, 1.0)
