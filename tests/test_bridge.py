"""End-to-end verification of the real-game bridge path, without Geometry Dash.

The bridge (RealGameBridge -> socket -> MockModServer -> GDSim) must yield the
exact same observations and rewards as the in-process sim env for identical
actions. If this holds, the only bridge surface not covered by tests is the C++
mod's reads of the live game's memory.
"""

import numpy as np
import pytest

from gdrl.envs import GDEnv
from gdrl.envs import protocol as proto
from gdrl.envs.bridge import GDRealEnv, RealGameBridge
from gdrl.envs.mock_mod import MockModServer


# -- protocol serialization --------------------------------------------------

def test_state_roundtrip():
    s = proto.State(x=12.5, y=1.5, vy=-3.25, grounded=True, gamemode=0,
                    flags=proto.FLAG_DEAD, length=80.0, frame=42,
                    percent=0.15625, speed_mult=1.0)
    back = proto.State.unpack(s.pack())
    assert back.x == s.x and back.y == s.y and back.vy == s.vy
    assert back.frame == 42 and back.dead and not back.complete
    assert back.length == 80.0 and back.percent == pytest.approx(0.15625)


def test_action_roundtrip():
    payload = proto.pack_action(hold=True, request_reset=False, request_geom=True)
    flags = proto.unpack_action(payload)
    assert flags & proto.ACT_HOLD
    assert flags & proto.ACT_REQUEST_GEOM
    assert not (flags & proto.ACT_REQUEST_RESET)


def test_geometry_roundtrip():
    recs = [(0, 20.0, 0.0), (1, 15.0, 0.0), (1, 21.0, 1.0)]
    back = proto.unpack_geometry(proto.pack_geometry(recs))
    assert back == recs


# -- end-to-end parity -------------------------------------------------------

def _bridge_env(level: str):
    server = MockModServer(level, port=0).start()
    bridge = RealGameBridge(port=server.port, timeout=5.0)
    bridge.connect()
    return server, GDRealEnv(bridge)


@pytest.mark.parametrize("level", ["spikes_easy", "blocks_and_spikes"])
def test_bridge_matches_sim(level):
    # Reference: in-process sim env.
    sim_env = GDEnv(level)
    sim_obs, _ = sim_env.reset(seed=0)

    # Bridge env over a real socket to the sim-backed mock.
    server, real_env = _bridge_env(level)
    # The wire protocol transmits float32 (as the real mod does), so parity is
    # to float32 precision, not bit-exact float64. The sim dynamics themselves
    # stay identical (both sides run the same float64 GDSim); only the values
    # read back through the socket are float32-truncated.
    F32_ATOL = 1e-5
    try:
        real_obs, _ = real_env.reset(seed=0)
        np.testing.assert_allclose(real_obs, sim_obs, atol=F32_ATOL, err_msg="reset obs mismatch")

        rng = np.random.default_rng(123)
        for t in range(400):
            action = int(rng.integers(2))
            s_obs, s_rew, s_term, s_trunc, s_info = sim_env.step(action)
            r_obs, r_rew, r_term, r_trunc, r_info = real_env.step(action)

            np.testing.assert_allclose(r_obs, s_obs, atol=F32_ATOL, err_msg=f"obs mismatch at step {t}")
            assert r_rew == pytest.approx(s_rew, abs=1e-4), f"reward mismatch at step {t}"
            assert r_term == s_term, f"termination mismatch at step {t}"
            assert r_info["dead"] == s_info["dead"]
            assert r_info["won"] == s_info["won"]
            assert r_info["progress"] == pytest.approx(s_info["progress"], abs=1e-5)
            if s_term:
                break
    finally:
        real_env.close()
        server.stop()


def test_bridge_reset_recovers_geometry():
    server, real_env = _bridge_env("blocks_and_spikes")
    try:
        real_env.reset(seed=0)
        # geometry_to_level should reconstruct the same object count as the level.
        assert real_env.level is not None
        assert len(real_env.level.objects) == len(server.level.objects)
    finally:
        real_env.close()
        server.stop()
