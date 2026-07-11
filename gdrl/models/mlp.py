"""Networks for structured-state observations.

A CNN over the look-ahead grid is a planned ablation; for a 403-float
observation an MLP trains faster than anything and keeps GA genomes small.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _trunk(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Q-network for DQN and policy genome for the GA."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.body = _trunk(in_dim, hidden)
        self.head = nn.Linear(hidden[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for PPO."""

    def __init__(self, in_dim: int, n_actions: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.body = _trunk(in_dim, hidden)
        self.pi = nn.Linear(hidden[-1], n_actions)
        self.v = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)

    def dist(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits, _ = self(x)
        return torch.distributions.Categorical(logits=logits)
