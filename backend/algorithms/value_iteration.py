"""
Value Iteration Algorithm.

Implements the Value Iteration algorithm which directly computes the optimal
value function V* using Bellman optimality updates.

BELLMAN OPTIMALITY EQUATION:
    V(s) = max_a [R(s,a) + γ * V(s')]

Unlike Policy Iteration, Value Iteration combines evaluation and improvement
into a single step by always taking the max over actions.

POLICY EXTRACTION:
    After V* converges, extract the greedy policy:
    π*(s) = argmax_a [R(s,a) + γ * V*(s')]

CONVERGENCE:
    The algorithm iterates until max|V_new(s) - V_old(s)| < θ for all states.
"""

from typing import List, Dict, Any, Callable, Tuple


class ValueIteration:
    """
    Value Iteration algorithm.
    
    Computes the optimal value function V* and extracts the optimal policy π*
    using iterative Bellman optimality updates.
    
    This class does NOT contain environment logic. It requires a transition
    function to be provided that returns (next_state, reward, done) for any
    (state, action) pair.
    
    Attributes:
        num_states (int): Total number of states.
        num_actions (int): Total number of actions.
        gamma (float): Discount factor (0 < γ ≤ 1).
        theta (float): Convergence threshold.
        values (List[float]): Current value estimates V(s).
        policy (List[List[float]]): Extracted policy π(a|s).
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
        Initialize Value Iteration.
        
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
        
        # Initialize uniform policy (will be updated after convergence)
        uniform_prob = 1.0 / num_actions
        self.policy: List[List[float]] = [
            [uniform_prob] * num_actions for _ in range(num_states)
        ]
        
        # Diagnostics storage
        self.iteration_history: List[Dict[str, Any]] = []
    
    def run(
        self,
        transition_func: Callable[[int, int], Tuple[int, float, bool]],
        max_iterations: int = 1000
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Run Value Iteration until convergence.
        
        Args:
            transition_func: Function(state, action) -> (next_state, reward, done).
            max_iterations: Maximum number of iterations (default: 1000).
            
        Returns:
            Tuple containing:
                - policy: Optimal policy π*(a|s) as 2D probability list.
                - values: Optimal value function V*(s).
        """
        # Clear history and reset values
        self.iteration_history = []
        self.values = [0.0] * self.num_states
        
        # Value Iteration loop
        for iteration in range(max_iterations):
            max_delta = 0.0
            old_values = self.values.copy()
            
            # Update value for each state using Bellman optimality
            for state in range(self.num_states):
                # Compute Q(s, a) for all actions and take the max
                best_value = None
                
                for action in range(self.num_actions):
                    # Get transition result
                    next_state, reward, done = transition_func(state, action)
                    
                    # Compute Q(s, a) = R(s,a) + γ * V(s')
                    if done:
                        q_value = reward
                    else:
                        q_value = reward + self.gamma * old_values[next_state]
                    
                    # Track best value
                    if best_value is None or q_value > best_value:
                        best_value = q_value
                
                # V(s) = max_a Q(s, a)
                self.values[state] = best_value
                
                # Track maximum change
                delta = abs(best_value - old_values[state])
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
        
        # Extract greedy policy from converged values
        self._extract_policy(transition_func)
        
        return [row.copy() for row in self.policy], self.values.copy()
    
    def _extract_policy(
        self,
        transition_func: Callable[[int, int], Tuple[int, float, bool]]
    ) -> None:
        """
        Extract greedy policy from value function.
        
        For each state, selects the action that maximizes Q(s, a).
        
        Args:
            transition_func: Function(state, action) -> (next_state, reward, done).
        """
        for state in range(self.num_states):
            # Compute Q(s, a) for all actions
            q_values: List[float] = []
            
            for action in range(self.num_actions):
                next_state, reward, done = transition_func(state, action)
                
                if done:
                    q_value = reward
                else:
                    q_value = reward + self.gamma * self.values[next_state]
                
                q_values.append(q_value)
            
            # Find the best action
            best_action = 0
            best_value = q_values[0]
            for action in range(1, self.num_actions):
                if q_values[action] > best_value:
                    best_value = q_values[action]
                    best_action = action
            
            # Set deterministic policy for this state
            self.policy[state] = [0.0] * self.num_actions
            self.policy[state][best_action] = 1.0
    
    def get_policy(self) -> List[List[float]]:
        """
        Get current policy.
        
        Returns:
            List[List[float]]: Copy of current policy π(a|s).
        """
        return [row.copy() for row in self.policy]
    
    def get_values(self) -> List[float]:
        """
        Get current value estimates.
        
        Returns:
            List[float]: Copy of current V(s) for all states.
        """
        return self.values.copy()
    
    def get_best_action(self, state: int) -> int:
        """
        Get the best action for a state according to current policy.
        
        Args:
            state: State ID.
            
        Returns:
            int: Action with highest probability.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        
        best_action = 0
        best_prob = self.policy[state][0]
        for action in range(1, self.num_actions):
            if self.policy[state][action] > best_prob:
                best_prob = self.policy[state][action]
                best_action = action
        return best_action
    
    def get_delta_history(self) -> List[float]:
        """
        Get history of max delta values across iterations.
        
        Returns:
            List[float]: Max ΔV for each iteration.
        """
        return [entry["max_delta"] for entry in self.iteration_history]
    
    def get_value_convergence(self) -> List[List[float]]:
        """
        Get value function at each iteration.
        
        Returns:
            List: Value function snapshots from each iteration.
        """
        return [entry["values"] for entry in self.iteration_history]
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics output.
        
        Returns:
            Dict containing:
                - type: "ValueIteration"
                - num_states: number of states
                - num_actions: number of actions
                - gamma: discount factor
                - theta: convergence threshold
                - num_iterations: iterations performed
                - final_policy: optimal policy
                - final_values: optimal value function
                - delta_history: max ΔV per iteration
                - converged: whether algorithm converged
        """
        num_iterations = len(self.iteration_history)
        
        # Check if converged
        converged = False
        if num_iterations > 0:
            final_delta = self.iteration_history[-1]["max_delta"]
            converged = final_delta < self.theta
        
        return {
            "type": "ValueIteration",
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "theta": self.theta,
            "num_iterations": num_iterations,
            "final_policy": [row.copy() for row in self.policy],
            "final_values": self.values.copy(),
            "delta_history": self.get_delta_history(),
            "converged": converged
        }
