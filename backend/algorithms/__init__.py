"""
Algorithms package for the educational RL web tool.

This package contains reinforcement learning algorithm implementations
and data containers.
"""

from backend.algorithms.containers import ValueTable, QTable, Policy
from backend.algorithms.policy_evaluation import PolicyEvaluation
from backend.algorithms.policy_iteration import PolicyIteration
from backend.algorithms.value_iteration import ValueIteration
from backend.algorithms.monte_carlo_prediction import MonteCarloPrediction
from backend.algorithms.td_learning import TD0, NStepTD
from backend.algorithms.sarsa import SARSA
from backend.algorithms.q_learning import QLearning

__all__ = [
    "ValueTable", "QTable", "Policy",
    "PolicyEvaluation", "PolicyIteration", "ValueIteration",
    "MonteCarloPrediction", "TD0", "NStepTD", "SARSA", "QLearning"
]
