"""Physics tests for the non-cube gamemodes and portals.

These assert the qualitative behaviour that defines each mode (which way it
moves when you hold vs release, taps flipping/impulsing) and that an empty
corridor is completable, rather than frame-perfect GD fidelity — the constants
are approximate until calibrated against the real game.
"""

from gdrl.sim import GDSim, Level
from gdrl.sim.engine import BALL, SHIP, UFO, WAVE
from gdrl.sim.level import BLOCK, MODE_ID, SLOPE_DOWN, SLOPE_UP, LevelObject, Portal


def make(mode="cube", length=40.0, ceiling=10.0, objects=None, portals=None):
    lvl = Level("t", length, objects or [], portals=portals, ceiling=ceiling, start_mode=mode)
    return GDSim(lvl)


def run(sim, policy, max_frames=6000):
    s = sim.reset()
    for _ in range(max_frames):
        if s.dead or s.won:
            return s
        s = sim.step(s, policy(s))
    return s


def test_start_mode_is_applied():
    assert make("ship").reset().mode == SHIP
    assert make("wave").reset().mode == WAVE


def test_ship_holds_up_releases_down():
    sim = make("ship", ceiling=30.0)  # tall, so we test flight not the ceiling
    s = sim.reset()
    for _ in range(20):
        s = sim.step(s, True)   # thrust up
    assert s.y > 1.0 and s.vy > 0
    vy_thrust = s.vy
    s = sim.step(s, False)      # one release step: downward acceleration
    assert s.vy < vy_thrust


def test_wave_moves_at_45_degrees():
    sim = make("wave", ceiling=12.0)
    s = sim.step(sim.reset(), True)
    # |vy| should equal the horizontal speed (SPEED_1X) within a hair.
    from gdrl.sim import physics
    assert abs(s.vy - physics.SPEED_1X) < 1e-6
    # Climb clear of the floor, then release and confirm it dives.
    for _ in range(10):
        s = sim.step(s, True)
    s = sim.step(s, False)
    assert s.vy < 0  # release dives


def test_ball_tap_flips_gravity():
    sim = make("ball", ceiling=8.0)
    s = sim.reset()
    assert s.gravity == 1
    s = sim.step(s, True)   # rising edge -> flip
    assert s.gravity == -1
    s = sim.step(s, True)   # held, no new edge -> no flip
    assert s.gravity == -1
    s = sim.step(s, False)
    s = sim.step(s, True)   # new rising edge -> flip back
    assert s.gravity == 1


def test_ufo_tap_gives_upward_impulse():
    sim = make("ufo")
    s = sim.reset()
    s = sim.step(s, True)   # tap
    assert s.vy > 0


def test_empty_corridors_complete():
    # Ship/ufo hover, wave weaves, ball rolls — none should die in open space.
    policies = {
        "ship": lambda s: s.y < 4.0,
        "ufo": lambda s: s.y < 4.0,
        "wave": lambda s: s.y < 4.0,
        "ball": lambda s: False,
    }
    for mode, pol in policies.items():
        sim = make(mode, length=40.0, ceiling=8.0)
        s = run(sim, pol)
        assert s.won and not s.dead, f"{mode} corridor not completed (dead={s.dead})"


def _robot_apex(hold_frames, ceiling=15.0):
    sim = make("robot", length=100.0, ceiling=ceiling)
    s = sim.reset()
    apex = 0.0
    for f in range(120):
        s = sim.step(s, f < hold_frames)
        apex = max(apex, s.y)
        if s.grounded and f > hold_frames:
            break
    return apex


def test_robot_hold_controls_jump_height():
    # Longer hold -> higher jump (variable-height jump is the robot's mechanic).
    short, medium, long = _robot_apex(2), _robot_apex(6), _robot_apex(12)
    assert short < medium < long
    assert short > 0.0  # a tap still leaves the ground


def test_robot_boost_window_caps_height():
    # Holding past the boost window gains nothing more.
    assert _robot_apex(12) == _robot_apex(30)


def test_spider_teleports_between_surfaces():
    sim = make("spider", ceiling=8.0)
    s = sim.reset()
    assert s.y == 0.0 and s.gravity == 1
    s = sim.step(s, True)   # tap -> snap to ceiling, gravity up
    assert s.gravity == -1 and s.y > 6.0 and s.grounded
    s = sim.step(s, False)
    s = sim.step(s, True)   # tap -> snap back to floor
    assert s.gravity == 1 and s.y == 0.0


def test_spider_teleport_into_spike_dies():
    # A hazard on the ceiling: teleporting up lands in it.
    ceiling = 8.0
    spike = LevelObject("spike", 3.0, ceiling - 1.0)  # sits just under the ceiling
    sim = make("spider", length=40.0, ceiling=ceiling, objects=[spike])
    s = sim.reset()
    # advance to just before the spike's x, then tap up into it
    while s.x < 2.6:
        s = sim.step(s, False)
    s = sim.step(s, True)
    assert s.dead


def test_cube_rides_up_slope_onto_platform():
    # A 3-block 45-degree ramp lifting the cube from y0 to a y3 platform.
    objs = [LevelObject(SLOPE_UP, 10.0, 0.0), LevelObject(SLOPE_UP, 11.0, 1.0),
            LevelObject(SLOPE_UP, 12.0, 2.0)]
    objs += [LevelObject(BLOCK, float(x), 2.0) for x in range(13, 26)]  # platform top=3
    sim = GDSim(Level("slope", 25.0, objs))
    s = run(sim, lambda st: False)  # no jumping — pure ride
    assert s.won and not s.dead
    # It actually climbed (the platform sits at y=3).
    peak = 0.0
    s2 = sim.reset()
    for _ in range(400):
        s2 = sim.step(s2, False)
        peak = max(peak, s2.y)
        if s2.dead or s2.won:
            break
    assert peak >= 2.8


def test_cube_rides_down_slope():
    # Spawn on a platform, ride a down-ramp back to the floor.
    objs = [LevelObject(BLOCK, float(x), 2.0) for x in range(0, 8)]  # platform top=3
    objs += [LevelObject(SLOPE_DOWN, 8.0, 2.0), LevelObject(SLOPE_DOWN, 9.0, 1.0),
             LevelObject(SLOPE_DOWN, 10.0, 0.0)]
    sim = GDSim(Level("down", 20.0, objs), spawn_y=3.0)
    s = run(sim, lambda st: False)
    assert s.won and not s.dead


def test_gamemode_portal_switches_mode():
    sim = make("cube", length=60.0,
               portals=[Portal("gamemode", 15.0, "ship")])
    s = sim.reset()
    assert s.mode == MODE_ID["cube"]
    while s.x < 16.0 and not s.dead:
        s = sim.step(s, False)
    assert s.mode == SHIP


def test_speed_portal_changes_advance_rate():
    sim = make("ship", length=200.0, portals=[Portal("speed", 10.0, 2.0)])
    s = sim.reset()
    # measure dx before and after the portal
    while s.x < 5.0:
        s = sim.step(s, True)
    x0 = s.x
    s = sim.step(s, True)
    dx_before = s.x - x0
    while s.x < 12.0:
        s = sim.step(s, True)
    x1 = s.x
    s = sim.step(s, True)
    dx_after = s.x - x1
    assert dx_after > 1.9 * dx_before


def test_gravity_portal_flips_gravity():
    sim = make("ship", length=60.0, portals=[Portal("gravity", 10.0, -1)])
    s = sim.reset()
    while s.x < 11.0 and not s.dead:
        s = sim.step(s, False)
    assert s.gravity == -1
