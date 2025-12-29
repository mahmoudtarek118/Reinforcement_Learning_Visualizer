"""
RL Data Containers for the educational RL web tool.

This module provides simple data storage containers for reinforcement learning.
These are NOT decision-making agents - they only store and retrieve values.

Containers:
    - ValueTable: Stores state values V(s), zero-initialized
    - QTable: Stores state-action values Q(s,a), zero-initialized
    - Policy: Stores action probability distributions per state

All containers support:
    - Zero initialization
    - Getter methods for retrieving stored values
    - Diagnostics output for debugging and visualization
"""

from typing import Dict, List, Any


class ValueTable:
    """
    Container for state value function V(s).
    
    Stores a single value for each state. All values are zero-initialized.
    This is a pure data container - no learning logic.
    
    Attributes:
        num_states (int): Total number of states.
        values (List[float]): Value for each state, indexed by state ID.
    """
    
    def __init__(self, num_states: int):
        """
        Initialize value table with zeros.
        
        Args:
            num_states: Total number of states in the environment.
        """
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        
        self.num_states = num_states
        # Zero-initialize all state values
        self.values: List[float] = [0.0] * num_states
    
    def get_value(self, state: int) -> float:
        """
        Get the value for a specific state.
        
        Args:
            state: State ID to get value for.
            
        Returns:
            float: The value V(s) for the given state.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        return self.values[state]
    
    def get_all_values(self) -> List[float]:
        """
        Get all state values.
        
        Returns:
            List[float]: Copy of all state values.
        """
        return self.values.copy()
    
    def set_value(self, state: int, value: float) -> None:
        """
        Set the value for a specific state.
        
        Args:
            state: State ID to set value for.
            value: The value to store.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        self.values[state] = value
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostics output for debugging and visualization.
        
        Returns:
            Dict containing:
                - type: "ValueTable"
                - num_states: number of states
                - values: list of all values
                - min_value: minimum value across all states
                - max_value: maximum value across all states
        """
        return {
            "type": "ValueTable",
            "num_states": self.num_states,
            "values": self.values.copy(),
            "min_value": min(self.values),
            "max_value": max(self.values)
        }


class QTable:
    """
    Container for action-value function Q(s, a).
    
    Stores a value for each state-action pair. All values are zero-initialized.
    This is a pure data container - no learning logic.
    
    Attributes:
        num_states (int): Total number of states.
        num_actions (int): Total number of actions.
        values (List[List[float]]): Q-values indexed by [state][action].
    """
    
    def __init__(self, num_states: int, num_actions: int):
        """
        Initialize Q-table with zeros.
        
        Args:
            num_states: Total number of states in the environment.
            num_actions: Total number of actions in the environment.
        """
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if num_actions < 1:
            raise ValueError("num_actions must be at least 1")
        
        self.num_states = num_states
        self.num_actions = num_actions
        # Zero-initialize all Q-values: values[state][action]
        self.values: List[List[float]] = [
            [0.0] * num_actions for _ in range(num_states)
        ]
    
    def get_value(self, state: int, action: int) -> float:
        """
        Get the Q-value for a specific state-action pair.
        
        Args:
            state: State ID.
            action: Action ID.
            
        Returns:
            float: The value Q(s, a) for the given state-action pair.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}. Must be 0 to {self.num_actions - 1}")
        return self.values[state][action]
    
    def get_values_for_state(self, state: int) -> List[float]:
        """
        Get all action values for a specific state.
        
        Args:
            state: State ID.
            
        Returns:
            List[float]: Copy of Q(s, a) for all actions a.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        return self.values[state].copy()
    
    def get_all_values(self) -> List[List[float]]:
        """
        Get all Q-values.
        
        Returns:
            List[List[float]]: Copy of all Q-values indexed by [state][action].
        """
        return [row.copy() for row in self.values]
    
    def set_value(self, state: int, action: int, value: float) -> None:
        """
        Set the Q-value for a specific state-action pair.
        
        Args:
            state: State ID.
            action: Action ID.
            value: The value to store.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}. Must be 0 to {self.num_actions - 1}")
        self.values[state][action] = value
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostics output for debugging and visualization.
        
        Returns:
            Dict containing:
                - type: "QTable"
                - num_states: number of states
                - num_actions: number of actions
                - values: 2D list of all Q-values
                - min_value: minimum Q-value across all state-action pairs
                - max_value: maximum Q-value across all state-action pairs
        """
        all_values = [v for row in self.values for v in row]
        return {
            "type": "QTable",
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "values": [row.copy() for row in self.values],
            "min_value": min(all_values),
            "max_value": max(all_values)
        }


class Policy:
    """
    Container for policy π(a|s).
    
    Stores action probability distribution for each state.
    Initialized with uniform distribution over all actions.
    This is a pure data container - no decision-making logic.
    
    Attributes:
        num_states (int): Total number of states.
        num_actions (int): Total number of actions.
        probabilities (List[List[float]]): Action probabilities indexed by [state][action].
    """
    
    def __init__(self, num_states: int, num_actions: int):
        """
        Initialize policy with uniform distribution.
        
        Args:
            num_states: Total number of states in the environment.
            num_actions: Total number of actions in the environment.
        """
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if num_actions < 1:
            raise ValueError("num_actions must be at least 1")
        
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Initialize with uniform distribution: each action has probability 1/num_actions
        uniform_prob = 1.0 / num_actions
        self.probabilities: List[List[float]] = [
            [uniform_prob] * num_actions for _ in range(num_states)
        ]
    
    def get_action_probabilities(self, state: int) -> List[float]:
        """
        Get action probability distribution for a specific state.
        
        Args:
            state: State ID.
            
        Returns:
            List[float]: Copy of π(a|s) for all actions a.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        return self.probabilities[state].copy()
    
    def get_probability(self, state: int, action: int) -> float:
        """
        Get probability of taking a specific action in a specific state.
        
        Args:
            state: State ID.
            action: Action ID.
            
        Returns:
            float: Probability π(a|s).
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}. Must be 0 to {self.num_actions - 1}")
        return self.probabilities[state][action]
    
    def get_all_probabilities(self) -> List[List[float]]:
        """
        Get all action probabilities.
        
        Returns:
            List[List[float]]: Copy of all probabilities indexed by [state][action].
        """
        return [row.copy() for row in self.probabilities]
    
    def set_action_probabilities(self, state: int, probabilities: List[float]) -> None:
        """
        Set action probability distribution for a specific state.
        
        Args:
            state: State ID.
            probabilities: List of probabilities for each action. Must sum to 1.0.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        if len(probabilities) != self.num_actions:
            raise ValueError(f"Expected {self.num_actions} probabilities, got {len(probabilities)}")
        
        # Validate probabilities sum to 1 (with small tolerance for floating point)
        prob_sum = sum(probabilities)
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")
        
        # Validate all probabilities are non-negative
        if any(p < 0 for p in probabilities):
            raise ValueError("Probabilities must be non-negative")
        
        self.probabilities[state] = probabilities.copy()
    
    def set_deterministic(self, state: int, action: int) -> None:
        """
        Set a deterministic policy for a specific state (one action has probability 1).
        
        Args:
            state: State ID.
            action: Action ID to set as deterministic choice.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}. Must be 0 to {self.num_states - 1}")
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}. Must be 0 to {self.num_actions - 1}")
        
        # Set all probabilities to 0, except the chosen action
        self.probabilities[state] = [0.0] * self.num_actions
        self.probabilities[state][action] = 1.0
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostics output for debugging and visualization.
        
        Returns:
            Dict containing:
                - type: "Policy"
                - num_states: number of states
                - num_actions: number of actions
                - probabilities: 2D list of all probabilities
                - deterministic_states: count of states with deterministic policy
        """
        # Count states where one action has probability 1.0
        deterministic_count = 0
        for state_probs in self.probabilities:
            if any(p >= 0.999 for p in state_probs):
                deterministic_count += 1
        
        return {
            "type": "Policy",
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "probabilities": [row.copy() for row in self.probabilities],
            "deterministic_states": deterministic_count
        }
