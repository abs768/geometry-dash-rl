import numpy as np

from gdrl.envs import GDEnv
from gdrl.envs.gd_env import WIN_REWARD
from gdrl.envs.observation import GRID_COLS, GRID_ROWS


def test_observation_shape_and_spaces():
    env = GDEnv("spikes_easy")
    obs, info = env.reset(seed=0)
    assert obs.shape == (4 + GRID_COLS * GRID_ROWS * 2,)
    assert obs.dtype == np.float32
    assert env.action_space.n == 2
    assert info["progress"] == 0.0


def test_grid_sees_upcoming_spike():
    env = GDEnv("single_spike")  # spike at x=15
    obs, _ = env.reset(seed=0)
    grid = obs[4:].reshape(GRID_COLS, GRID_ROWS, 2)
    assert grid[15, 0, 1] == 1.0  # hazard channel, column 15, ground row
    assert grid[:, :, 0].sum() == 0.0  # no solids in this level


def test_env_is_deterministic():
    def trajectory():
        env = GDEnv("spikes_easy")
        obs, _ = env.reset(seed=0)
        out = [obs.copy()]
        rng = np.random.default_rng(42)
        for _ in range(200):
            obs, reward, terminated, truncated, _ = env.step(int(rng.integers(2)))
            out.append(obs.copy())
            if terminated or truncated:
                break
        return out

    a, b = trajectory(), trajectory()
    assert len(a) == len(b)
    for oa, ob in zip(a, b):
        np.testing.assert_array_equal(oa, ob)


def test_flat_level_win_reward():
    env = GDEnv("flat")
    env.reset(seed=0)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(0)
        total += reward
        if terminated or truncated:
            break
    assert info["won"]
    # total = level length traversed + win bonus
    assert total > WIN_REWARD


def test_death_gives_negative_reward():
    env = GDEnv("single_spike")
    env.reset(seed=0)
    while True:
        _, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert info["dead"]
    assert reward < -5.0
