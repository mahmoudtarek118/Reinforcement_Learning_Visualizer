"""
Environments package for the educational RL web tool.

This package contains environment implementations for reinforcement learning.
"""

from backend.envs.base_env import BaseEnv
from backend.envs.gridworld import GridWorld
from backend.envs.frozenlake import FrozenLake

__all__ = ["BaseEnv", "GridWorld", "FrozenLake"]
