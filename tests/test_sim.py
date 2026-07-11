import math

from gdrl.sim import GDSim, Level, LevelObject
from gdrl.sim import physics
from gdrl.sim.level import BLOCK, SPIKE


def run(sim, actions):
    state = sim.reset()
    for hold in actions:
        state = sim.step(state, hold)
    return state


def roll_until_done(sim, policy, max_frames=10_000):
    state = sim.reset()
    for _ in range(max_frames):
        if state.dead or state.won:
            return state
        state = sim.step(state, policy(state))
    return state


def test_flat_level_completes_without_jumping():
    sim = GDSim(Level("flat", 30.0, []))
    state = roll_until_done(sim, lambda s: False)
    assert state.won and not state.dead


def test_jump_apex_matches_constants():
    sim = GDSim(Level("flat", 1000.0, []))
    state = sim.reset()
    state = sim.step(state, True)  # takeoff
    apex = 0.0
    while not state.grounded:
        apex = max(apex, state.y)
        state = sim.step(state, False)
    expected = physics.JUMP_VELOCITY ** 2 / (2 * physics.GRAVITY)
    assert math.isclose(apex, expected, rel_tol=0.10)
    assert 1.8 < apex < 2.5  # sane cube jump height in blocks


def test_running_into_spike_dies():
    level = Level("spike", 30.0, [LevelObject(SPIKE, 10.0, 0.0)])
    state = roll_until_done(GDSim(level), lambda s: False)
    assert state.dead
    assert 9.0 < state.x < 11.5


def test_jumping_clears_single_spike():
    level = Level("spike", 30.0, [LevelObject(SPIKE, 10.0, 0.0)])
    sim = GDSim(level)
    # Jump when the spike is ~2.5 blocks ahead (a full jump covers ~4.4 blocks,
    # so jumping too early lands the cube on the spike).
    state = roll_until_done(sim, lambda s: s.grounded and 7.0 < s.x < 8.0)
    assert state.won and not state.dead


def test_running_into_block_side_dies():
    level = Level("wall", 30.0, [LevelObject(BLOCK, 10.0, 0.0)])
    state = roll_until_done(GDSim(level), lambda s: False)
    assert state.dead


def test_landing_on_block_survives():
    level = Level("step", 30.0, [LevelObject(BLOCK, 10.0, 0.0), LevelObject(BLOCK, 11.0, 0.0)])
    sim = GDSim(level)
    state = roll_until_done(sim, lambda s: s.grounded and 7.5 < s.x < 8.5)
    assert state.won and not state.dead


def test_step_is_deterministic():
    level = Level("spike", 30.0, [LevelObject(SPIKE, 10.0, 0.0)])
    sim = GDSim(level)
    actions = [i % 7 == 0 for i in range(300)]
    a = run(sim, actions)
    b = run(sim, actions)
    assert a == b
