"""
Policy Iteration Algorithm.

Implements the Policy Iteration algorithm which alternates between:
1. Policy Evaluation: Compute V^π(s) for current policy
2. Policy Improvement: Update policy greedily based on V^π

ALGORITHM:
    1. Initialize policy π arbitrarily (e.g., uniform random)
    2. Repeat:
        a. Policy Evaluation: Compute V^π using iterative Bellman updates
        b. Policy Improvement: For each state, set π(s) = argmax_a Q(s,a)
           where Q(s,a) = R(s,a) + γ * V^π(s')
    3. Until policy is stable (no changes)

CONVERGENCE:
    Policy Iteration is guaranteed to converge to the optimal policy π*
    in a finite number of iterations for finite MDPs.
"""

from typing import List, Dict, Any, Callable, Tuple


class PolicyIteration:
    """
    Policy Iteration algorithm.
    
    Finds the optimal policy by alternating between policy evaluation
    and greedy policy improvement.
    
    This class does NOT contain environment logic. It requires a transition
    function to be provided that returns (next_state, reward, done) for any
    (state, action) pair.
    
    Attributes:
        num_states (int): Total number of states.
        num_actions (int): Total number of actions.
        gamma (float): Discount factor (0 < γ ≤ 1).
        theta (float): Convergence threshold for policy evaluation.
        policy (List[List[float]]): Current policy π(a|s).
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
        Initialize Policy Iteration.
        
        Args:
            num_states: Total number of states in the environment.
            num_actions: Total number of actions in the environment.
            gamma: Discount factor (default: 0.99). Must be in (0, 1].
            theta: Convergence threshold for evaluation (default: 1e-6).
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
        
        # Initialize uniform random policy
        uniform_prob = 1.0 / num_actions
        self.policy: List[List[float]] = [
            [uniform_prob] * num_actions for _ in range(num_states)
        ]
        
        # Zero-initialize value estimates
        self.values: List[float] = [0.0] * num_states
        
        # Diagnostics storage
        self.iteration_history: List[Dict[str, Any]] = []
    
    def _evaluate_policy(
        self,
        transition_func: Callable[[int, int], Tuple[int, float, bool]],
        max_eval_iterations: int = 1000
    ) -> List[float]:
        """
        Policy Evaluation step: Compute V^π for current policy.
        
        Uses iterative Bellman expectation updates until convergence.
        
        Args:
            transition_func: Function(state, action) -> (next_state, reward, done).
            max_eval_iterations: Maximum iterations for evaluation.
            
        Returns:
            List[float]: Converged value function V^π(s).
        """
        # Reset values to zero for fresh evaluation
        self.values = [0.0] * self.num_states
        
        # Track evaluation iterations
        eval_iterations = 0
        
        # Iterative evaluation loop
        for eval_iter in range(max_eval_iterations):
            eval_iterations += 1
            max_delta = 0.0
            old_values = self.values.copy()
            
            # Update value for each state
            for state in range(self.num_states):
                new_value = 0.0
                
                # Sum over all actions weighted by policy probability
                for action in range(self.num_actions):
                    action_prob = self.policy[state][action]
                    
                    if action_prob == 0.0:
                        continue
                    
                    # Get transition result
                    next_state, reward, done = transition_func(state, action)
                    
                    # Compute expected value
                    if done:
                        action_value = reward
                    else:
                        action_value = reward + self.gamma * old_values[next_state]
                    
                    new_value += action_prob * action_value
                
                self.values[state] = new_value
                
                delta = abs(new_value - old_values[state])
                if delta > max_delta:
                    max_delta = delta
            
            # Check convergence
            if max_delta < self.theta:
                break
        
        return self.values.copy()
    
    def _improve_policy(
        self,
        transition_func: Callable[[int, int], Tuple[int, float, bool]]
    ) -> bool:
        """
        Policy Improvement step: Greedily update policy based on V^π.
        
        For each state, selects the action that maximizes Q(s,a).
        
        Args:
            transition_func: Function(state, action) -> (next_state, reward, done).
            
        Returns:
            bool: True if policy changed, False if policy is stable.
        """
        policy_changed = False
        
        # Improve policy for each state
        for state in range(self.num_states):
            # Store old best action (the one with probability 1.0, if deterministic)
            old_action = -1
            for action in range(self.num_actions):
                if self.policy[state][action] >= 0.999:
                    old_action = action
                    break
            
            # Compute Q(s, a) for all actions
            q_values: List[float] = []
            for action in range(self.num_actions):
                next_state, reward, done = transition_func(state, action)
                
                if done:
                    q_value = reward
                else:
                    q_value = reward + self.gamma * self.values[next_state]
                
                q_values.append(q_value)
            
            # Find the best action (greedy selection)
            best_value = q_values[0]
            best_action = 0
            for action in range(1, self.num_actions):
                if q_values[action] > best_value:
                    best_value = q_values[action]
                    best_action = action
            
            # Update policy to be deterministic on best action
            new_policy_state = [0.0] * self.num_actions
            new_policy_state[best_action] = 1.0
            self.policy[state] = new_policy_state
            
            # Check if policy changed for this state
            if best_action != old_action:
                policy_changed = True
        
        return policy_changed
    
    def run(
        self,
        transition_func: Callable[[int, int], Tuple[int, float, bool]],
        max_iterations: int = 100,
        max_eval_iterations: int = 1000
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Run Policy Iteration until convergence.
        
        Args:
            transition_func: Function(state, action) -> (next_state, reward, done).
            max_iterations: Maximum policy improvement iterations (default: 100).
            max_eval_iterations: Maximum iterations per evaluation (default: 1000).
            
        Returns:
            Tuple containing:
                - policy: Optimal policy π*(a|s) as 2D probability list.
                - values: Optimal value function V*(s).
        """
        # Clear history
        self.iteration_history = []
        
        # Reset to uniform policy
        uniform_prob = 1.0 / self.num_actions
        self.policy = [
            [uniform_prob] * self.num_actions for _ in range(self.num_states)
        ]
        
        # Policy iteration loop
        for iteration in range(max_iterations):
            # Step 1: Policy Evaluation
            values_before = self.values.copy()
            self._evaluate_policy(transition_func, max_eval_iterations)
            
            # Record policy before improvement
            policy_before = [row.copy() for row in self.policy]
            
            # Step 2: Policy Improvement
            policy_changed = self._improve_policy(transition_func)
            
            # Record iteration diagnostics
            iteration_diagnostics = {
                "iteration": iteration,
                "policy": [row.copy() for row in self.policy],
                "values": self.values.copy(),
                "policy_changed": policy_changed,
                "value_change": max(
                    abs(self.values[s] - values_before[s])
                    for s in range(self.num_states)
                ) if iteration > 0 else 0.0
            }
            self.iteration_history.append(iteration_diagnostics)
            
            # Check for policy stability
            if not policy_changed:
                break
        
        return [row.copy() for row in self.policy], self.values.copy()
    
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
    
    def get_policy_evolution(self) -> List[List[List[float]]]:
        """
        Get policy at each iteration.
        
        Returns:
            List: Policy snapshots from each iteration.
        """
        return [entry["policy"] for entry in self.iteration_history]
    
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
                - type: "PolicyIteration"
                - num_states: number of states
                - num_actions: number of actions
                - gamma: discount factor
                - theta: convergence threshold
                - num_iterations: iterations performed
                - final_policy: optimal policy
                - final_values: optimal value function
                - policy_evolution: policy at each iteration
                - value_convergence: values at each iteration
                - converged: whether policy is stable
        """
        num_iterations = len(self.iteration_history)
        
        # Check if converged (policy unchanged in last iteration)
        converged = False
        if num_iterations > 0:
            converged = not self.iteration_history[-1]["policy_changed"]
        
        return {
            "type": "PolicyIteration",
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "theta": self.theta,
            "num_iterations": num_iterations,
            "final_policy": [row.copy() for row in self.policy],
            "final_values": self.values.copy(),
            "policy_evolution": self.get_policy_evolution(),
            "value_convergence": self.get_value_convergence(),
            "converged": converged
        }
