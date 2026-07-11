from gdrl.agents.dqn import train_dqn
from gdrl.agents.ppo import train_ppo
from gdrl.agents.ga import train_ga

TRAINERS = {"dqn": train_dqn, "ppo": train_ppo, "ga": train_ga}

__all__ = ["train_dqn", "train_ppo", "train_ga", "TRAINERS"]
