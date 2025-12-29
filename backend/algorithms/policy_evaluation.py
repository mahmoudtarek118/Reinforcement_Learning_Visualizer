"""
Policy Evaluation Algorithm.

Implements iterative policy evaluation using Bellman expectation updates.
Given a policy π, computes the state-value function V^π(s) for all states.

BELLMAN EXPECTATION EQUATION:
    V(s) = Σ_a π(a|s) * Σ_s' P(s'|s,a) * [R(s,a,s') + γ * V(s')]

For deterministic environments (like GridWorld), P(s'|s,a) = 1 for the
resulting state, so the equation simplifies to:
    V(s) = Σ_a π(a|s) * [R(s,a) + γ * V(s')]

CONVERGENCE:
    The algorithm iterates until max|V_new(s) - V_old(s)| < θ for all states.
"""

from typing import List, Dict, Any, Callable, Tuple


class PolicyEvaluation:
    """
    Iterative Policy Evaluation algorithm.
    
    Evaluates a given stochastic policy by computing V^π(s) for all states
    using iterative Bellman expectation updates.
    
    This class does NOT contain environment logic. It requires a transition
    function to be provided that returns (next_state, reward, done) for any
    (state, action) pair.
    
    Attributes:
        num_states (int): Total number of states.
        num_actions (int): Total number of actions.
        gamma (float): Discount factor (0 < γ ≤ 1).
        theta (float): Convergence threshold.
        values (List[float]): Current value estimates V(s).
        iteration_history (List[Dict]): Diagnostics for each iteration.
    """
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float = 0.99,
        theta: float = 1e-6
    ):
        """
        Initialize Policy Evaluation.
        
        Args:
            num_states: Total number of states in the environment.
            num_actions: Total number of actions in the environment.
            gamma: Discount factor (default: 0.99). Must be in (0, 1].
            theta: Convergence threshold (default: 1e-6).
        """
        # Validate inputs
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if num_actions < 1:
            raise ValueError("num_actions must be at least 1")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        if theta <= 0:
            raise ValueError("theta must be positive")
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta
        
        # Zero-initialize value estimates
        self.values: List[float] = [0.0] * num_states
        
        # Diagnostics storage
        self.iteration_history: List[Dict[str, Any]] = []
    
    def evaluate(
        self,
        policy: List[List[float]],
        transition_func: Callable[[int, int], Tuple[int, float, bool]],
        max_iterations: int = 1000
    ) -> List[float]:
        """
        Run iterative policy evaluation until convergence.
        
        Args:
            policy: Policy π(a|s) as a 2D list [state][action] of probabilities.
            transition_func: Function(state, action) -> (next_state, reward, done).
                            This function encapsulates environment dynamics.
            max_iterations: Maximum number of iterations (default: 1000).
            
        Returns:
            List[float]: Converged value function V^π(s).
        """
        # Validate policy dimensions
        if len(policy) != self.num_states:
            raise ValueError(f"Policy must have {self.num_states} states")
        for state_idx in range(self.num_states):
            if len(policy[state_idx]) != self.num_actions:
                raise ValueError(f"Policy state {state_idx} must have {self.num_actions} actions")
        
        # Store policy snapshot for diagnostics
        policy_snapshot = [row.copy() for row in policy]
        
        # Clear previous history
        self.iteration_history = []
        
        # Reset values to zero
        self.values = [0.0] * self.num_states
        
        # Iterative policy evaluation loop
        for iteration in range(max_iterations):
            # Track maximum change in value for convergence check
            max_delta = 0.0
            
            # Store old values for delta calculation
            old_values = self.values.copy()
            
            # Update value for each state
            for state in range(self.num_states):
                # Compute new value using Bellman expectation equation
                # V(s) = Σ_a π(a|s) * [R(s,a) + γ * V(s')]
                new_value = 0.0
                
                # Sum over all actions weighted by policy probability
                for action in range(self.num_actions):
                    # Get policy probability for this action
                    action_prob = policy[state][action]
                    
                    # Skip if action has zero probability
                    if action_prob == 0.0:
                        continue
                    
                    # Get transition result from environment dynamics
                    next_state, reward, done = transition_func(state, action)
                    
                    # Compute expected value for this action
                    # If terminal, future value is 0
                    if done:
                        action_value = reward
                    else:
                        action_value = reward + self.gamma * old_values[next_state]
                    
                    # Weight by policy probability and add to total
                    new_value += action_prob * action_value
                
                # Update value for this state
                self.values[state] = new_value
                
                # Track maximum change
                delta = abs(new_value - old_values[state])
                if delta > max_delta:
                    max_delta = delta
            
            # Record iteration diagnostics
            iteration_diagnostics = {
                "iteration": iteration,
                "values": self.values.copy(),
                "max_delta": max_delta
            }
            self.iteration_history.append(iteration_diagnostics)
            
            # Check for convergence
            if max_delta < self.theta:
                break
        
        return self.values.copy()
    
    def get_values(self) -> List[float]:
        """
        Get current value estimates.
        
        Returns:
            List[float]: Copy of current V(s) for all states.
        """
        return self.values.copy()
    
    def get_delta_history(self) -> List[float]:
        """
        Get history of max delta values across iterations.
        
        Returns:
            List[float]: Max ΔV for each iteration.
        """
        return [entry["max_delta"] for entry in self.iteration_history]
    
    def get_iteration_count(self) -> int:
        """
        Get number of iterations performed.
        
        Returns:
            int: Number of iterations until convergence or max_iterations.
        """
        return len(self.iteration_history)
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics output.
        
        Returns:
            Dict containing:
                - type: "PolicyEvaluation"
                - num_states: number of states
                - num_actions: number of actions
                - gamma: discount factor used
                - theta: convergence threshold
                - num_iterations: iterations performed
                - final_values: converged value function
                - delta_history: max ΔV per iteration
                - converged: whether algorithm converged
        """
        num_iterations = len(self.iteration_history)
        
        # Check if converged (last delta was below theta)
        converged = False
        if num_iterations > 0:
            final_delta = self.iteration_history[-1]["max_delta"]
            converged = final_delta < self.theta
        
        return {
            "type": "PolicyEvaluation",
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "theta": self.theta,
            "num_iterations": num_iterations,
            "final_values": self.values.copy(),
            "delta_history": self.get_delta_history(),
            "converged": converged
        }
