"""Tests for the tunable physics bundle and domain randomization.

These guard the sim-to-real machinery: the default bundle must not change the
numerics (determinism / prior results depend on it), perturbations must move the
jump arc in the physically-correct direction, and env randomization must be
reproducible from the reset seed and off by default.
"""

import numpy as np

from gdrl.envs import GDEnv
from gdrl.sim import GDSim, Level
from gdrl.sim import physics


def _jump_apex(sim: GDSim) -> float:
    s = sim.reset()
    s = sim.step(s, True)
    apex = s.y
    while not s.grounded:
        apex = max(apex, s.y)
        s = sim.step(s, False)
    return apex


def test_default_params_are_calibrated_constants():
    p = physics.PhysicsParams()
    assert p.jump_velocity == physics.JUMP_VELOCITY
    assert p.gravity == physics.GRAVITY
    assert p.speed_1x == physics.SPEED_1X
    assert physics.NOMINAL == p


def test_default_bundle_reproduces_calibrated_jump():
    # The whole determinism story rides on this: default params == old sim.
    apex = _jump_apex(GDSim(Level("t", 1000.0, [])))
    assert abs(apex - 2.128) < 1e-2


def test_scaled_jump_moves_apex_monotonically():
    level = Level("t", 1000.0, [])
    weak = _jump_apex(GDSim(level, params=physics.NOMINAL.scaled(jump=0.85)))
    nominal = _jump_apex(GDSim(level, params=physics.NOMINAL))
    strong = _jump_apex(GDSim(level, params=physics.NOMINAL.scaled(jump=1.15)))
    assert weak < nominal < strong


def test_scaled_speed_changes_horizontal_travel():
    level = Level("t", 1000.0, [])
    fast = GDSim(level, params=physics.NOMINAL.scaled(speed=1.2))
    base = GDSim(level, params=physics.NOMINAL)
    sf, sb = fast.reset(), base.reset()
    for _ in range(30):
        sf, sb = fast.step(sf, False), base.step(sb, False)
    assert sf.x > sb.x * 1.15  # ~20% faster horizontally


def test_randomize_off_by_default_is_deterministic():
    def run():
        env = GDEnv("spikes_easy")  # randomize defaults to 0.0
        env.reset(seed=0)
        xs = []
        for _ in range(40):
            _, _, term, trunc, info = env.step(1)
            xs.append(info["x"])
            if term or trunc:
                break
        return xs
    assert run() == run()


def test_randomize_perturbs_physics_within_range():
    # With DR on, a fresh episode uses jittered physics: the same action stream
    # should generally NOT reproduce the un-randomized trajectory, and the
    # sampled jump velocity must stay inside the requested band.
    env = GDEnv("spikes_easy", randomize=0.2)
    env.reset(seed=1)
    assert env.sim.p.jump_velocity != physics.JUMP_VELOCITY  # was jittered
    for attr, base in [("jump_velocity", physics.JUMP_VELOCITY),
                       ("gravity", physics.GRAVITY),
                       ("speed_1x", physics.SPEED_1X)]:
        val = getattr(env.sim.p, attr)
        assert 0.8 * base <= val <= 1.2 * base


def test_randomize_is_reproducible_from_seed():
    def sampled():
        env = GDEnv("spikes_easy", randomize=0.2)
        env.reset(seed=7)
        return (env.sim.p.jump_velocity, env.sim.p.gravity, env.sim.p.speed_1x)
    assert sampled() == sampled()
