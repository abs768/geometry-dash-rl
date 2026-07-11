"""DQN with target network, epsilon-greedy exploration and optional double-DQN.

This is the reproduce-prior-work baseline: same algorithm family as
geometry-dash-ai, but running against the headless sim instead of the live
game, so it is not capped at real-time speed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from gdrl.envs import GDEnv
from gdrl.models import MLP
from gdrl.utils import RunLogger, set_seed


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        idxs = self.rng.integers(0, len(self), size=batch_size)
        t = torch.as_tensor
        return (t(self.obs[idxs]), t(self.actions[idxs]), t(self.rewards[idxs]),
                t(self.next_obs[idxs]), t(self.dones[idxs]))


def train_dqn(cfg: dict, run_dir: str) -> None:
    set_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))

    env = GDEnv(cfg["level"])
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    hidden = tuple(cfg.get("hidden", [256, 256]))
    q_net = MLP(obs_dim, n_actions, hidden)
    target_net = MLP(obs_dim, n_actions, hidden)
    target_net.load_state_dict(q_net.state_dict())
    optim = torch.optim.Adam(q_net.parameters(), lr=cfg.get("lr", 1e-4))

    buffer = ReplayBuffer(cfg.get("buffer_size", 100_000), obs_dim, rng)

    total_steps = cfg.get("total_steps", 200_000)
    warmup = cfg.get("warmup_steps", 2_000)
    batch_size = cfg.get("batch_size", 64)
    gamma = cfg.get("gamma", 0.99)
    eps_start, eps_end = cfg.get("eps_start", 1.0), cfg.get("eps_end", 0.05)
    eps_decay_steps = cfg.get("eps_decay_steps", total_steps // 2)
    target_every = cfg.get("target_update_steps", 1_000)
    double = cfg.get("double", True)

    logger = RunLogger(run_dir, ["step", "episode", "ep_return", "progress", "epsilon", "won"])

    obs, _ = env.reset(seed=cfg.get("seed", 0))
    ep_return, episode, best_progress = 0.0, 0, 0.0

    for step in range(1, total_steps + 1):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        if step <= warmup or rng.random() < epsilon:
            action = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                action = int(q_net(torch.as_tensor(obs).unsqueeze(0)).argmax())

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward

        if terminated or truncated:
            episode += 1
            best_progress = max(best_progress, info["progress"])
            if episode % cfg.get("log_every_episodes", 20) == 0:
                logger.log(step=step, episode=episode, ep_return=ep_return,
                           progress=info["progress"], epsilon=epsilon, won=int(info["won"]))
            obs, _ = env.reset()
            ep_return = 0.0

        if step > warmup and len(buffer) >= batch_size:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(batch_size)
            with torch.no_grad():
                if double:
                    next_actions = q_net(b_next).argmax(dim=1)
                    next_q = target_net(b_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_q = target_net(b_next).max(dim=1).values
                target = b_rew + gamma * (1.0 - b_done) * next_q
            q = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optim.step()

        if step % target_every == 0:
            target_net.load_state_dict(q_net.state_dict())
        if step % cfg.get("checkpoint_every", 25_000) == 0:
            logger.save_checkpoint(q_net, step=step)

    logger.save_checkpoint(q_net, step=total_steps)
    print(f"[dqn] done: {episode} episodes, best progress {best_progress:.1%}")
