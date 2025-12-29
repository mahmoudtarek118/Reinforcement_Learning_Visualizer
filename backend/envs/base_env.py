"""
Base environment interface for the educational RL web tool.

This module defines the abstract base class that all environments must implement.
All environments in this tool follow a consistent interface for state management,
action execution, and episode control.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseEnv(ABC):
    """
    Abstract base class for all reinforcement learning environments.
    
    All environments must implement:
    - reset(): Initialize/reset the environment to starting state
    - step(action): Execute action and return (next_state, reward, done, info)
    - get_state(): Return current state
    - get_valid_actions(): Return list of valid actions from current state
    - get_state_space_size(): Return total number of states
    - get_action_space_size(): Return total number of actions
    """
    
    @abstractmethod
    def reset(self) -> int:
        """
        Reset the environment to its initial state.
        
        Returns:
            int: The initial state ID.
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Integer ID of the action to take.
            
        Returns:
            Tuple containing:
                - next_state (int): The new state ID after taking the action.
                - reward (float): The reward received for this transition.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information about the transition.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> int:
        """
        Get the current state of the environment.
        
        Returns:
            int: The current state ID.
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> list:
        """
        Get the list of valid actions from the current state.
        
        Returns:
            list: List of valid action IDs.
        """
        pass
    
    @abstractmethod
    def get_state_space_size(self) -> int:
        """
        Get the total number of states in the environment.
        
        Returns:
            int: Number of states.
        """
        pass
    
    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Get the total number of actions in the environment.
        
        Returns:
            int: Number of actions.
        """
        pass
