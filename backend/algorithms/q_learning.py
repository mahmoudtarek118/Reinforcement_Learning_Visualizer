"""
Q-Learning Algorithm.

Implements Q-Learning, an off-policy TD control algorithm that learns the
optimal Q-values regardless of the policy being followed.

Q-LEARNING UPDATE:
    Q(S, A) = Q(S, A) + α * [R + γ * max_a Q(S', a) - Q(S, A)]
    
    Note: Uses max over next state actions, not the action actually taken.

OFF-POLICY:
    Q-learning learns the optimal policy Q* while following any policy
    (e.g., ε-greedy for exploration). The max makes it optimistic.

DIFFERENCE FROM SARSA:
    - SARSA: Uses Q(S', A') where A' is the action taken
    - Q-Learning: Uses max_a Q(S', a) regardless of action taken
"""

from typing import List, Dict, Any
import random


class QLearning:
    """
    Q-Learning algorithm.
    
    Off-policy TD control that learns optimal Q(s, a) while exploring.
    
    Attributes:
        num_states (int): Total number of states.
        num_actions (int): Total number of actions.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate for ε-greedy.
        q_values (List[List[float]]): Q-table Q(s, a).
    """
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        """
        Initialize Q-Learning.
        
        Args:
            num_states: Total number of states.
            num_actions: Total number of actions.
            alpha: Learning rate (default: 0.1). Must be in (0, 1].
            gamma: Discount factor (default: 0.99). Must be in (0, 1].
            epsilon: Exploration rate (default: 0.1). Must be in [0, 1].
        """
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if num_actions < 1:
            raise ValueError("num_actions must be at least 1")
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Zero-initialize Q-table
        self.q_values: List[List[float]] = [
            [0.0] * num_actions for _ in range(num_states)
        ]
        
        # Diagnostics
        self.td_errors: List[float] = []
        self.episode_rewards: List[float] = []
        self.step_count = 0
    
    def select_action(self, state: int) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state.
            
        Returns:
            int: Selected action.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        
        # ε-greedy: explore with probability ε
        if random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.num_actions - 1)
        else:
            # Greedy action (exploitation)
            return self._get_greedy_action(state)
    
    def _get_greedy_action(self, state: int) -> int:
        """Get action with highest Q-value for state."""
        best_action = 0
        best_value = self.q_values[state][0]
        for action in range(1, self.num_actions):
            if self.q_values[state][action] > best_value:
                best_value = self.q_values[state][action]
                best_action = action
        return best_action
    
    def _get_max_q(self, state: int) -> float:
        """Get maximum Q-value for state."""
        max_q = self.q_values[state][0]
        for action in range(1, self.num_actions):
            if self.q_values[state][action] > max_q:
                max_q = self.q_values[state][action]
        return max_q
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Perform a single Q-learning update.
        
        Uses max_a Q(S', a) for the TD target (off-policy).
        
        Args:
            state: Current state S.
            action: Action taken A.
            reward: Reward received R.
            next_state: Next state S'.
            done: Whether episode terminated.
            
        Returns:
            float: The TD error.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}")
        
        # Compute TD target using max (off-policy)
        if done:
            td_target = reward
        else:
            max_next_q = self._get_max_q(next_state)
            td_target = reward + self.gamma * max_next_q
        
        # Compute TD error
        td_error = td_target - self.q_values[state][action]
        
        # Update Q-value
        self.q_values[state][action] = self.q_values[state][action] + self.alpha * td_error
        
        # Record diagnostics
        self.td_errors.append(td_error)
        self.step_count += 1
        
        return td_error
    
    def get_q_values(self) -> List[List[float]]:
        """Get current Q-table."""
        return [row.copy() for row in self.q_values]
    
    def get_q_value(self, state: int, action: int) -> float:
        """Get Q-value for a specific state-action pair."""
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}")
        return self.q_values[state][action]
    
    def get_policy(self) -> List[List[float]]:
        """
        Extract current ε-greedy policy.
        
        Returns:
            List[List[float]]: Policy π(a|s) with ε-greedy probabilities.
        """
        policy = []
        
        for state in range(self.num_states):
            best_action = self._get_greedy_action(state)
            
            # Compute ε-greedy probabilities
            probs = [self.epsilon / self.num_actions] * self.num_actions
            probs[best_action] += 1.0 - self.epsilon
            
            policy.append(probs)
        
        return policy
    
    def get_greedy_policy(self) -> List[int]:
        """
        Extract greedy policy (deterministic).
        
        Returns:
            List[int]: Best action for each state.
        """
        return [self._get_greedy_action(state) for state in range(self.num_states)]
    
    def get_best_action(self, state: int) -> int:
        """Get greedy action for a state (no exploration)."""
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        return self._get_greedy_action(state)
    
    def get_td_errors(self) -> List[float]:
        """Get history of TD errors."""
        return self.td_errors.copy()
    
    def set_epsilon(self, epsilon: float) -> None:
        """Update exploration rate."""
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
    
    def reset(self) -> None:
        """Reset Q-values and history."""
        self.q_values = [
            [0.0] * self.num_actions for _ in range(self.num_states)
        ]
        self.td_errors = []
        self.episode_rewards = []
        self.step_count = 0
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "type": "QLearning",
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q_values": [row.copy() for row in self.q_values],
            "policy": self.get_greedy_policy(),
            "step_count": self.step_count,
            "td_errors": self.td_errors.copy(),
            "mean_td_error": sum(abs(e) for e in self.td_errors) / len(self.td_errors) if self.td_errors else 0.0
        }
