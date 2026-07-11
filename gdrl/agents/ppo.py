"""PPO with clipped objective and GAE, on vectorized sim environments.

The headless sim is cheap enough that num_envs=16 on CPU outruns any
real-time in-game trainer by orders of magnitude in frames/sec.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from gdrl.envs import make_env
from gdrl.models import ActorCritic
from gdrl.utils import RunLogger, set_seed


def train_ppo(cfg: dict, run_dir: str) -> None:
    seed = cfg.get("seed", 0)
    set_seed(seed)

    num_envs = cfg.get("num_envs", 16)
    envs = gym.vector.SyncVectorEnv([make_env(cfg["level"]) for _ in range(num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    model = ActorCritic(obs_dim, n_actions, tuple(cfg.get("hidden", [256, 256])))
    optim = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 3e-4))

    total_steps = cfg.get("total_steps", 1_000_000)
    rollout_len = cfg.get("rollout_len", 128)
    gamma = cfg.get("gamma", 0.99)
    gae_lambda = cfg.get("gae_lambda", 0.95)
    clip_eps = cfg.get("clip_eps", 0.2)
    epochs = cfg.get("epochs", 4)
    minibatches = cfg.get("minibatches", 4)
    ent_coef = cfg.get("ent_coef", 0.01)
    vf_coef = cfg.get("vf_coef", 0.5)

    logger = RunLogger(run_dir, ["step", "ep_return", "progress", "won_rate"])

    obs, _ = envs.reset(seed=seed)
    obs = torch.as_tensor(obs, dtype=torch.float32)
    dones = torch.zeros(num_envs)

    batch = rollout_len * num_envs
    recent_progress, recent_wins, recent_returns = [], [], []
    ep_returns = np.zeros(num_envs)
    global_step = 0

    while global_step < total_steps:
        obs_buf = torch.zeros((rollout_len, num_envs, obs_dim))
        act_buf = torch.zeros((rollout_len, num_envs), dtype=torch.long)
        logp_buf = torch.zeros((rollout_len, num_envs))
        rew_buf = torch.zeros((rollout_len, num_envs))
        done_buf = torch.zeros((rollout_len, num_envs))
        val_buf = torch.zeros((rollout_len, num_envs))

        for t in range(rollout_len):
            global_step += num_envs
            with torch.no_grad():
                logits, value = model(obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            obs_buf[t], act_buf[t], logp_buf[t] = obs, action, logp
            val_buf[t], done_buf[t] = value, dones

            next_obs, reward, terminated, truncated, infos = envs.step(action.numpy())
            done = np.logical_or(terminated, truncated)
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32)
            ep_returns += reward

            for i in range(num_envs):
                if done[i]:
                    # gymnasium <1.0 puts terminal info in infos["final_info"];
                    # >=1.0 (next-step autoreset) reports it in the per-key arrays.
                    if "final_info" in infos and infos["final_info"][i] is not None:
                        final = infos["final_info"][i]
                        recent_progress.append(final["progress"])
                        recent_wins.append(float(final["won"]))
                    elif "progress" in infos:
                        recent_progress.append(float(infos["progress"][i]))
                        recent_wins.append(float(infos["won"][i]))
                    recent_returns.append(ep_returns[i])
                    ep_returns[i] = 0.0

            obs = torch.as_tensor(next_obs, dtype=torch.float32)
            dones = torch.as_tensor(done, dtype=torch.float32)

        # GAE
        with torch.no_grad():
            _, next_value = model(obs)
        adv_buf = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(num_envs)
        for t in reversed(range(rollout_len)):
            next_nonterminal = 1.0 - (dones if t == rollout_len - 1 else done_buf[t + 1])
            next_val = next_value if t == rollout_len - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * next_nonterminal - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape(batch, obs_dim)
        b_act = act_buf.reshape(batch)
        b_logp = logp_buf.reshape(batch)
        b_adv = adv_buf.reshape(batch)
        b_ret = ret_buf.reshape(batch)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        idxs = np.arange(batch)
        mb_size = batch // minibatches
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch, mb_size):
                mb = idxs[start:start + mb_size]
                logits, value = model(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[mb])
                ratio = torch.exp(logp - b_logp[mb])
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (value - b_ret[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optim.step()

        if recent_returns:
            logger.log(
                step=global_step,
                ep_return=float(np.mean(recent_returns[-50:])),
                progress=float(np.mean(recent_progress[-50:])) if recent_progress else 0.0,
                won_rate=float(np.mean(recent_wins[-50:])) if recent_wins else 0.0,
            )
        if global_step % cfg.get("checkpoint_every", 100_000) < batch:
            logger.save_checkpoint(model, step=global_step)

    logger.save_checkpoint(model, step=global_step)
    envs.close()
    print(f"[ppo] done at {global_step} steps")
