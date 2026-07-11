"""Genetic algorithm / neuroevolution over MLP policy weights.

Geometry Dash levels are deterministic, so a single rollout is an exact
fitness measurement — the ideal regime for evolutionary search. Selection is
truncation + elitism; variation is Gaussian weight mutation (crossover is
deliberately omitted: for NN genomes it mostly destroys structure).
"""

from __future__ import annotations

import numpy as np
import torch

from gdrl.envs import GDEnv
from gdrl.models import MLP
from gdrl.utils import RunLogger, set_seed


def _flatten(model: torch.nn.Module) -> np.ndarray:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()]).numpy().copy()


def _unflatten(model: torch.nn.Module, flat: np.ndarray) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.as_tensor(flat[i:i + n], dtype=torch.float32).reshape(p.shape)
        i += n


def _rollout(env: GDEnv, model: MLP) -> tuple[float, float, bool]:
    """Deterministic greedy rollout. Returns (fitness, progress, won)."""
    obs, _ = env.reset()
    total = 0.0
    with torch.no_grad():
        while True:
            action = int(model(torch.as_tensor(obs).unsqueeze(0)).argmax())
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                return total, info["progress"], info["won"]


def train_ga(cfg: dict, run_dir: str) -> None:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    env = GDEnv(cfg["level"])
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden = tuple(cfg.get("hidden", [64, 64]))  # small genome evolves faster

    pop_size = cfg.get("pop_size", 64)
    generations = cfg.get("generations", 200)
    elite_frac = cfg.get("elite_frac", 0.125)
    mutation_std = cfg.get("mutation_std", 0.05)
    init_std = cfg.get("init_std", 0.5)

    model = MLP(obs_dim, n_actions, hidden)
    genome_len = len(_flatten(model))
    population = rng.normal(0.0, init_std, size=(pop_size, genome_len)).astype(np.float32)

    n_elite = max(1, int(pop_size * elite_frac))
    logger = RunLogger(run_dir, ["generation", "best_fitness", "mean_fitness", "best_progress", "won"])
    best_ever = (-np.inf, None)

    for gen in range(1, generations + 1):
        fitness = np.zeros(pop_size)
        progress = np.zeros(pop_size)
        won_any = False
        for i in range(pop_size):
            _unflatten(model, population[i])
            fitness[i], progress[i], won = _rollout(env, model)
            won_any = won_any or won

        order = np.argsort(fitness)[::-1]
        if fitness[order[0]] > best_ever[0]:
            best_ever = (fitness[order[0]], population[order[0]].copy())

        logger.log(generation=gen, best_fitness=float(fitness[order[0]]),
                   mean_fitness=float(fitness.mean()),
                   best_progress=float(progress[order[0]]), won=int(won_any))

        elites = population[order[:n_elite]]
        children = np.empty_like(population)
        children[:n_elite] = elites  # elitism: best genomes survive unmutated
        parent_idx = rng.integers(0, n_elite, size=pop_size - n_elite)
        noise = rng.normal(0.0, mutation_std, size=(pop_size - n_elite, genome_len)).astype(np.float32)
        children[n_elite:] = elites[parent_idx] + noise
        population = children

        if won_any and cfg.get("stop_on_win", True):
            print(f"[ga] level completed at generation {gen}")
            break

    _unflatten(model, best_ever[1])
    logger.save_checkpoint(model, fitness=float(best_ever[0]))
    print(f"[ga] done: best fitness {best_ever[0]:.2f}")
