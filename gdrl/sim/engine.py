"""Deterministic headless simulator, gamemode-aware.

Supports cube, ship, ball, ufo and wave with faithful-ish physics, plus robot
and spider as documented approximations (robot ~ cube, spider ~ ball). Gravity
can be flipped and speed changed by portals mid-level. The cube path with
normal gravity and 1x speed is numerically identical to the original v0.1 sim,
so existing cube results and tests are unaffected.

Physics constants live in gdrl.sim.physics and are approximate for the
non-cube modes (see the note there). step() is a pure function of
(state, action, level): determinism is load-bearing for GA and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from gdrl.sim import physics
from gdrl.sim.level import BLOCK, MODE_ID, SPIKE, Level

CUBE, SHIP, BALL, UFO, WAVE, ROBOT, SPIDER = range(7)


@dataclass(frozen=True)
class SimState:
    x: float
    y: float          # bottom of the player box
    vy: float
    grounded: bool
    dead: bool
    won: bool
    frame: int
    mode: int = CUBE
    gravity: int = 1  # +1 normal (pulls down), -1 flipped (pulls up)
    speed: float = 1.0
    held: bool = False  # hold applied on the previous step (for edge detection)
    boost: int = 0      # robot jump-boost frames remaining


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class GDSim:
    def __init__(self, level: Level, spawn_x: float = 0.0, spawn_y: float = 0.0):
        self.level = level
        self.spawn_x = spawn_x
        self.spawn_y = spawn_y

    def reset(self) -> SimState:
        return SimState(
            x=self.spawn_x, y=self.spawn_y, vy=0.0, grounded=True,
            dead=False, won=False, frame=0,
            mode=MODE_ID.get(self.level.start_mode, CUBE), gravity=1, speed=1.0,
        )

    # -- vertical dynamics per gamemode ------------------------------------

    def _vertical(self, s: SimState, hold: bool) -> tuple[float, bool, int, int]:
        """Return (vy, grounded, gravity, boost) after applying input this frame.

        grounded here is the pre-integration value (a jump clears it so gravity
        applies the same frame, matching the original cube behaviour). Spider's
        teleport is handled separately in step(); here spider only falls if it
        has run off an edge.
        """
        g = s.gravity
        vy, grounded, boost = s.vy, s.grounded, s.boost
        rising_edge = hold and not s.held

        mode = s.mode
        if mode == CUBE:
            if hold and grounded:
                vy = physics.JUMP_VELOCITY * g
                grounded = False
            if not grounded:
                vy -= physics.GRAVITY * physics.DT * g
        elif mode == ROBOT:
            # Variable-height jump: pop on press, extra thrust while held.
            if rising_edge and grounded:
                vy = physics.ROBOT_JUMP_VELOCITY * g
                grounded = False
                boost = physics.ROBOT_BOOST_FRAMES
            if not grounded:
                if hold and boost > 0:
                    vy += physics.ROBOT_BOOST_ACCEL * physics.DT * g
                    boost -= 1
                else:
                    boost = 0
                vy -= physics.GRAVITY * physics.DT * g
        elif mode == SHIP:
            accel = -g * physics.SHIP_GRAVITY + (g * physics.SHIP_THRUST if hold else 0.0)
            vy += accel * physics.DT
            vy = max(-physics.SHIP_MAX_VY, min(physics.SHIP_MAX_VY, vy))
            grounded = False
        elif mode == UFO:
            if rising_edge:
                vy = physics.UFO_IMPULSE * g
                grounded = False
            if not grounded:
                vy -= physics.UFO_GRAVITY * physics.DT * g
        elif mode == BALL:
            if rising_edge:
                g = -g  # tap flips gravity
                grounded = False
            if not grounded:
                vy -= physics.BALL_GRAVITY * physics.DT * g
        elif mode == SPIDER:
            if not grounded:  # ran off an edge: fall until it hits a surface
                vy -= physics.SPIDER_GRAVITY * physics.DT * g
        elif mode == WAVE:
            vy = (1.0 if hold else -1.0) * g * physics.SPEED_1X * s.speed
            grounded = False

        # Common terminal-velocity clamp (symmetric).
        term = abs(physics.TERMINAL_VELOCITY)
        vy = max(-term, min(term, vy))
        return vy, grounded, g, boost

    # -- step --------------------------------------------------------------

    def _snap_surface(self, x: float, cur_y: float, g: int, size: float) -> float | None:
        """The y (player bottom) of the nearest surface in gravity direction g,
        for the spider teleport. Returns None if none found (shouldn't happen:
        floor/ceiling always bound the space)."""
        ceil_top = self.level.ceiling - size
        blocks = [o for o in self.level.objects_near(x - 1.0, x + size + 1.0)
                  if o.type == BLOCK and o.aabb()[0] < x + size and o.aabb()[2] > x]
        if g > 0:  # teleport down: land on the highest surface at/below us
            candidates = [0.0] + [o.aabb()[3] for o in blocks if o.aabb()[3] <= cur_y + 1e-6]
            return max(candidates)
        else:      # teleport up: land under the lowest surface at/above us
            tops = [ceil_top] + [o.aabb()[1] - size for o in blocks
                                 if o.aabb()[1] - size >= cur_y - 1e-6]
            return min(tops)

    def _spider_teleport(self, state: SimState, hold: bool) -> SimState:
        """Spider tap: flip gravity and snap instantly to the opposite surface."""
        size = physics.PLAYER_SIZE
        g = -state.gravity
        x = state.x + physics.SPEED_1X * state.speed * physics.DT
        y = self._snap_surface(x, state.y, g, size)
        if y is None:
            y = state.y

        # Dying if we teleport into a spike at the destination.
        player = (x, y, x + size, y + size)
        dead = any(o.type == SPIKE and _overlaps(player, o.aabb())
                   for o in self.level.objects_near(x - 1.0, x + size + 1.0))

        mode, speed = state.mode, state.speed
        for portal in self.level.portals_crossed(state.x, x):
            if portal.kind == "gamemode":
                mode = MODE_ID.get(portal.value, mode) if isinstance(portal.value, str) else int(portal.value)
            elif portal.kind == "gravity":
                g = int(portal.value)
            elif portal.kind == "speed":
                speed = float(portal.value)

        won = not dead and x >= self.level.length
        return SimState(x=x, y=y, vy=0.0, grounded=True, dead=dead, won=won,
                        frame=state.frame + 1, mode=mode, gravity=g, speed=speed,
                        held=hold, boost=0)

    def step(self, state: SimState, hold: bool) -> SimState:
        if state.dead or state.won:
            return state

        # Spider teleports on the tap frame instead of integrating physics.
        if state.mode == SPIDER and hold and not state.held:
            return self._spider_teleport(state, hold)

        vy, grounded, g, boost = self._vertical(state, hold)
        size = physics.PLAYER_SIZE
        ceil_top = self.level.ceiling - size

        prev_bottom = state.y
        prev_top = state.y + size
        x = state.x + physics.SPEED_1X * state.speed * physics.DT
        y = state.y + vy * physics.DT

        # Floor / ceiling. The surface gravity pulls toward makes the player
        # grounded; the opposite surface just stops motion.
        if y <= 0.0:
            y = 0.0
            if g > 0:
                vy, grounded = 0.0, True
            else:
                vy = max(vy, 0.0)
        elif y >= ceil_top:
            y = ceil_top
            if g < 0:
                vy, grounded = 0.0, True
            else:
                vy = min(vy, 0.0)

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
                if g > 0:
                    top = box[3]
                    if vy <= 0.0 and prev_bottom >= top - physics.LANDING_TOLERANCE:
                        y, vy, grounded, landed_on_block = top, 0.0, True, True
                    else:
                        dead = True
                        break
                else:
                    bottom = box[1]
                    if vy >= 0.0 and prev_top <= bottom + physics.LANDING_TOLERANCE:
                        y, vy, grounded, landed_on_block = bottom - size, 0.0, True, True
                    else:
                        dead = True
                        break
                player = (x, y, x + size, y + size)

        # Walking off a block edge: airborne until a surface is under (over) us.
        if grounded and not landed_on_block:
            on_floor = (g > 0 and y <= 0.0) or (g < 0 and y >= ceil_top)
            if not on_floor:
                surface = y if g > 0 else y + size
                supported = any(
                    obj.type == BLOCK
                    and abs((obj.aabb()[3] if g > 0 else obj.aabb()[1]) - surface) < 1e-9
                    and obj.aabb()[0] < x + size and obj.aabb()[2] > x
                    for obj in self.level.objects_near(x - 1.0, x + size + 1.0)
                )
                if not supported:
                    grounded = False

        # Apply any portals crossed this frame (gamemode / gravity / speed).
        mode, speed = state.mode, state.speed
        for portal in self.level.portals_crossed(state.x, x):
            if portal.kind == "gamemode":
                mode = MODE_ID.get(portal.value, mode) if isinstance(portal.value, str) else int(portal.value)
                grounded = False
            elif portal.kind == "gravity":
                g = int(portal.value)
                grounded = False
            elif portal.kind == "speed":
                speed = float(portal.value)

        won = not dead and x >= self.level.length

        return SimState(
            x=x, y=y, vy=vy, grounded=grounded, dead=dead, won=won,
            frame=state.frame + 1, mode=mode, gravity=g, speed=speed, held=hold,
            boost=boost,
        )

    def progress(self, state: SimState) -> float:
        return min(state.x / self.level.length, 1.0)
