"""
Monte Carlo Prediction Algorithm.

Implements First-Visit Monte Carlo for estimating state-value function V^π(s).
Uses complete episodes to compute returns and update value estimates.

FIRST-VISIT MC:
    For each state s appearing in an episode:
    - Only consider the FIRST time s is visited in that episode
    - Compute the return G from that first visit to end of episode
    - Update the average return for state s

RETURN CALCULATION:
    G_t = R_{t+1} + γ * R_{t+2} + γ² * R_{t+3} + ... + γ^{T-t-1} * R_T
    
    Where T is the terminal time step.

VALUE UPDATE:
    V(s) = average of all returns observed from first visits to s
    
    Incremental update formula:
    V(s) = V(s) + (1/N(s)) * (G - V(s))
    
    Where N(s) is the count of first visits to state s.
"""

from typing import List, Dict, Any, Tuple


class MonteCarloPrediction:
    """
    First-Visit Monte Carlo Prediction algorithm.
    
    Estimates V^π(s) by averaging returns from complete episodes.
    
    This class does NOT contain environment logic. Episodes are provided
    as lists of (state, action, reward) tuples.
    
    Attributes:
        num_states (int): Total number of states.
        gamma (float): Discount factor (0 < γ ≤ 1).
        values (List[float]): Current value estimates V(s).
        visit_counts (List[int]): Number of first visits to each state.
        returns_history (List[Dict]): Diagnostics for returns.
        episode_history (List[Dict]): Stored episodes.
    """
    
    def __init__(
        self,
        num_states: int,
        gamma: float = 0.99
    ):
        """
        Initialize Monte Carlo Prediction.
        
        Args:
            num_states: Total number of states in the environment.
            gamma: Discount factor (default: 0.99). Must be in (0, 1].
        """
        # Validate inputs
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        
        self.num_states = num_states
        self.gamma = gamma
        
        # Zero-initialize value estimates
        self.values: List[float] = [0.0] * num_states
        
        # Track number of first visits to each state
        self.visit_counts: List[int] = [0] * num_states
        
        # Store all returns for each state (for diagnostics)
        self.state_returns: List[List[float]] = [[] for _ in range(num_states)]
        
        # Episode storage
        self.episode_history: List[List[Tuple[int, int, float]]] = []
        
        # Value update history for diagnostics
        self.update_history: List[Dict[str, Any]] = []
    
    def store_episode(
        self,
        episode: List[Tuple[int, int, float]]
    ) -> None:
        """
        Store an episode for later processing.
        
        Args:
            episode: List of (state, action, reward) tuples representing
                    one complete episode from start to termination.
        """
        self.episode_history.append(episode.copy())
    
    def process_episode(
        self,
        episode: List[Tuple[int, int, float]]
    ) -> Dict[str, Any]:
        """
        Process a single episode using First-Visit MC.
        
        Args:
            episode: List of (state, action, reward) tuples.
            
        Returns:
            Dict with diagnostics about the processing.
        """
        if len(episode) == 0:
            return {"states_updated": [], "returns": {}}
        
        # Track which states have been visited in this episode (first-visit)
        visited_states: set = set()
        
        # Track updates made in this episode
        states_updated: List[int] = []
        returns_computed: Dict[int, float] = {}
        value_changes: Dict[int, float] = {}
        
        # Process episode backwards to compute returns efficiently
        # First, compute returns for each time step
        T = len(episode)
        returns: List[float] = [0.0] * T
        
        # Last step return is just the reward (no future)
        returns[T - 1] = episode[T - 1][2]
        
        # Work backwards to compute returns
        for t in range(T - 2, -1, -1):
            reward = episode[t][2]
            returns[t] = reward + self.gamma * returns[t + 1]
        
        # Now process forward to apply first-visit rule
        for t in range(T):
            state = episode[t][0]
            
            # First-visit check: only update if not yet visited in this episode
            if state not in visited_states:
                visited_states.add(state)
                
                # Get the return from this time step
                G = returns[t]
                
                # Store return for diagnostics
                self.state_returns[state].append(G)
                returns_computed[state] = G
                
                # Increment visit count
                self.visit_counts[state] += 1
                N = self.visit_counts[state]
                
                # Incremental mean update: V(s) = V(s) + (1/N) * (G - V(s))
                old_value = self.values[state]
                self.values[state] = old_value + (1.0 / N) * (G - old_value)
                
                value_changes[state] = self.values[state] - old_value
                states_updated.append(state)
        
        # Create diagnostics for this episode
        diagnostics = {
            "episode_length": T,
            "states_updated": states_updated,
            "returns": returns_computed,
            "value_changes": value_changes
        }
        
        self.update_history.append(diagnostics)
        
        return diagnostics
    
    def process_all_stored_episodes(self) -> int:
        """
        Process all stored episodes.
        
        Returns:
            int: Number of episodes processed.
        """
        count = 0
        for episode in self.episode_history:
            self.process_episode(episode)
            count += 1
        return count
    
    def get_values(self) -> List[float]:
        """
        Get current value estimates.
        
        Returns:
            List[float]: Copy of current V(s) for all states.
        """
        return self.values.copy()
    
    def get_value(self, state: int) -> float:
        """
        Get value estimate for a specific state.
        
        Args:
            state: State ID.
            
        Returns:
            float: Current V(s) for the given state.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        return self.values[state]
    
    def get_visit_counts(self) -> List[int]:
        """
        Get number of first visits to each state.
        
        Returns:
            List[int]: Visit count for each state.
        """
        return self.visit_counts.copy()
    
    def get_returns_for_state(self, state: int) -> List[float]:
        """
        Get all returns observed for a specific state.
        
        Args:
            state: State ID.
            
        Returns:
            List[float]: All returns from first visits to this state.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        return self.state_returns[state].copy()
    
    def get_episode_count(self) -> int:
        """
        Get number of stored episodes.
        
        Returns:
            int: Number of episodes stored.
        """
        return len(self.episode_history)
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """
        Get history of value updates.
        
        Returns:
            List[Dict]: Update diagnostics for each processed episode.
        """
        return self.update_history.copy()
    
    def reset(self) -> None:
        """
        Reset all values and history.
        """
        self.values = [0.0] * self.num_states
        self.visit_counts = [0] * self.num_states
        self.state_returns = [[] for _ in range(self.num_states)]
        self.episode_history = []
        self.update_history = []
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics output.
        
        Returns:
            Dict containing:
                - type: "MonteCarloPrediction"
                - num_states: number of states
                - gamma: discount factor
                - values: current value estimates
                - visit_counts: first-visit counts per state
                - episodes_stored: number of stored episodes
                - episodes_processed: number of processed episodes
                - all_returns: returns for each state
        """
        return {
            "type": "MonteCarloPrediction",
            "num_states": self.num_states,
            "gamma": self.gamma,
            "values": self.values.copy(),
            "visit_counts": self.visit_counts.copy(),
            "episodes_stored": len(self.episode_history),
            "episodes_processed": len(self.update_history),
            "all_returns": [returns.copy() for returns in self.state_returns]
        }
