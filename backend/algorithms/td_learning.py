"""
Temporal Difference Learning Algorithms.

Implements TD(0) and n-step TD for value prediction.
These methods bootstrap from current value estimates, unlike Monte Carlo
which waits for complete episode returns.

TD(0) UPDATE:
    V(S_t) = V(S_t) + α * [R_{t+1} + γ * V(S_{t+1}) - V(S_t)]
    
    TD Error: δ_t = R_{t+1} + γ * V(S_{t+1}) - V(S_t)

N-STEP TD UPDATE:
    G_t:t+n = R_{t+1} + γ*R_{t+2} + ... + γ^{n-1}*R_{t+n} + γ^n * V(S_{t+n})
    V(S_t) = V(S_t) + α * [G_t:t+n - V(S_t)]

ADVANTAGES:
    - TD(0): Updates after each step, more efficient than MC
    - n-step TD: Bridges TD(0) and MC, allows tuning bias-variance tradeoff
"""

from typing import List, Dict, Any, Tuple


class TD0:
    """
    TD(0) Prediction algorithm.
    
    Updates value estimates after each step using bootstrapped estimates.
    
    Attributes:
        num_states (int): Total number of states.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        values (List[float]): Current value estimates V(s).
        td_errors (List[float]): History of TD errors.
        step_count (int): Total steps processed.
    """
    
    def __init__(
        self,
        num_states: int,
        alpha: float = 0.1,
        gamma: float = 0.99
    ):
        """
        Initialize TD(0).
        
        Args:
            num_states: Total number of states.
            alpha: Learning rate (default: 0.1). Must be in (0, 1].
            gamma: Discount factor (default: 0.99). Must be in (0, 1].
        """
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        
        # Zero-initialize values
        self.values: List[float] = [0.0] * num_states
        
        # Diagnostics
        self.td_errors: List[float] = []
        self.value_history: List[List[float]] = []
        self.step_count = 0
    
    def update(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Perform a single TD(0) update.
        
        Args:
            state: Current state S_t.
            reward: Reward R_{t+1} received.
            next_state: Next state S_{t+1}.
            done: Whether episode terminated.
            
        Returns:
            float: The TD error δ_t.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        if next_state < 0 or next_state >= self.num_states:
            raise ValueError(f"Invalid next_state: {next_state}")
        
        # Compute TD target
        # If terminal, V(S_{t+1}) = 0
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.values[next_state]
        
        # Compute TD error: δ = R + γV(S') - V(S)
        td_error = td_target - self.values[state]
        
        # Update value: V(S) = V(S) + α * δ
        self.values[state] = self.values[state] + self.alpha * td_error
        
        # Record diagnostics
        self.td_errors.append(td_error)
        self.step_count += 1
        
        return td_error
    
    def process_episode(
        self,
        episode: List[Tuple[int, int, float, int, bool]]
    ) -> List[float]:
        """
        Process an entire episode with TD(0) updates.
        
        Args:
            episode: List of (state, action, reward, next_state, done) tuples.
            
        Returns:
            List[float]: TD errors for each step.
        """
        episode_td_errors = []
        
        for state, action, reward, next_state, done in episode:
            td_error = self.update(state, reward, next_state, done)
            episode_td_errors.append(td_error)
        
        # Record value snapshot after episode
        self.value_history.append(self.values.copy())
        
        return episode_td_errors
    
    def get_values(self) -> List[float]:
        """Get current value estimates."""
        return self.values.copy()
    
    def get_td_errors(self) -> List[float]:
        """Get history of all TD errors."""
        return self.td_errors.copy()
    
    def get_learning_curve(self) -> List[float]:
        """
        Get learning curve (average absolute TD error per episode).
        
        Returns:
            List[float]: Mean |δ| for each episode in value_history.
        """
        # This returns cumulative average TD error at each value snapshot
        curve = []
        for i, _ in enumerate(self.value_history):
            if i == 0:
                start = 0
            else:
                # Approximate: divide errors evenly
                start = i * (len(self.td_errors) // len(self.value_history))
            end = (i + 1) * (len(self.td_errors) // len(self.value_history))
            if start < end:
                segment = self.td_errors[start:end]
                avg = sum(abs(e) for e in segment) / len(segment)
                curve.append(avg)
        return curve
    
    def reset(self) -> None:
        """Reset values and history."""
        self.values = [0.0] * self.num_states
        self.td_errors = []
        self.value_history = []
        self.step_count = 0
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "type": "TD0",
            "num_states": self.num_states,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "values": self.values.copy(),
            "step_count": self.step_count,
            "td_errors": self.td_errors.copy(),
            "episodes_processed": len(self.value_history),
            "mean_td_error": sum(abs(e) for e in self.td_errors) / len(self.td_errors) if self.td_errors else 0.0
        }


class NStepTD:
    """
    N-Step TD Prediction algorithm.
    
    Uses a sliding window of n steps to compute returns before updating.
    Bridges TD(0) (n=1) and Monte Carlo (n=∞).
    
    Attributes:
        num_states (int): Total number of states.
        n (int): Number of steps for return calculation.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        values (List[float]): Current value estimates V(s).
    """
    
    def __init__(
        self,
        num_states: int,
        n: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.99
    ):
        """
        Initialize n-step TD.
        
        Args:
            num_states: Total number of states.
            n: Number of steps (default: 3). Must be >= 1.
            alpha: Learning rate (default: 0.1). Must be in (0, 1].
            gamma: Discount factor (default: 0.99). Must be in (0, 1].
        """
        if num_states < 1:
            raise ValueError("num_states must be at least 1")
        if n < 1:
            raise ValueError("n must be at least 1")
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        
        self.num_states = num_states
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        
        # Zero-initialize values
        self.values: List[float] = [0.0] * num_states
        
        # Sliding window buffer
        self.buffer: List[Tuple[int, float]] = []  # (state, reward)
        self.buffer_terminal: bool = False
        self.buffer_final_state: int = 0
        
        # Diagnostics
        self.td_errors: List[float] = []
        self.value_history: List[List[float]] = []
        self.step_count = 0
    
    def _compute_n_step_return(
        self,
        rewards: List[float],
        final_state: int,
        is_terminal: bool
    ) -> float:
        """
        Compute n-step return G_t:t+n.
        
        Args:
            rewards: List of rewards [R_{t+1}, R_{t+2}, ..., R_{t+n}].
            final_state: State S_{t+n}.
            is_terminal: Whether S_{t+n} is terminal.
            
        Returns:
            float: The n-step return.
        """
        # G = R_{t+1} + γ*R_{t+2} + ... + γ^{n-1}*R_{t+n}
        G = 0.0
        for i, reward in enumerate(rewards):
            G += (self.gamma ** i) * reward
        
        # Add bootstrapped value if not terminal
        if not is_terminal:
            G += (self.gamma ** len(rewards)) * self.values[final_state]
        
        return G
    
    def update(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Add a step to buffer and perform update if buffer is full.
        
        Args:
            state: Current state S_t.
            reward: Reward R_{t+1}.
            next_state: Next state S_{t+1}.
            done: Whether episode terminated.
            
        Returns:
            float: TD error if update performed, 0.0 otherwise.
        """
        if state < 0 or state >= self.num_states:
            raise ValueError(f"Invalid state: {state}")
        
        # Add to buffer
        self.buffer.append((state, reward))
        self.buffer_final_state = next_state
        self.buffer_terminal = done
        
        td_error = 0.0
        
        # If buffer has n steps, perform update for oldest state
        if len(self.buffer) >= self.n:
            # Extract state to update and rewards
            update_state = self.buffer[0][0]
            rewards = [step[1] for step in self.buffer[:self.n]]
            
            # Compute n-step return
            G = self._compute_n_step_return(
                rewards, 
                self.buffer_final_state, 
                self.buffer_terminal
            )
            
            # TD error
            td_error = G - self.values[update_state]
            
            # Update value
            self.values[update_state] = self.values[update_state] + self.alpha * td_error
            
            # Record diagnostics
            self.td_errors.append(td_error)
            self.step_count += 1
            
            # Remove oldest step from buffer (sliding window)
            self.buffer.pop(0)
        
        # If terminal, flush remaining buffer
        if done:
            while len(self.buffer) > 0:
                update_state = self.buffer[0][0]
                rewards = [step[1] for step in self.buffer]
                
                G = self._compute_n_step_return(
                    rewards, 
                    self.buffer_final_state, 
                    True  # Terminal
                )
                
                td_error = G - self.values[update_state]
                self.values[update_state] = self.values[update_state] + self.alpha * td_error
                
                self.td_errors.append(td_error)
                self.step_count += 1
                
                self.buffer.pop(0)
            
            # Clear buffer state
            self.buffer = []
            self.buffer_terminal = False
        
        return td_error
    
    def process_episode(
        self,
        episode: List[Tuple[int, int, float, int, bool]]
    ) -> List[float]:
        """
        Process an entire episode with n-step TD updates.
        
        Args:
            episode: List of (state, action, reward, next_state, done) tuples.
            
        Returns:
            List[float]: TD errors for each update.
        """
        # Reset buffer for new episode
        self.buffer = []
        self.buffer_terminal = False
        
        episode_td_errors = []
        
        for state, action, reward, next_state, done in episode:
            td_error = self.update(state, reward, next_state, done)
            if td_error != 0.0 or done:
                episode_td_errors.append(td_error)
        
        # Record value snapshot
        self.value_history.append(self.values.copy())
        
        return episode_td_errors
    
    def get_values(self) -> List[float]:
        """Get current value estimates."""
        return self.values.copy()
    
    def get_td_errors(self) -> List[float]:
        """Get history of all TD errors."""
        return self.td_errors.copy()
    
    def get_learning_curve(self) -> List[float]:
        """Get mean absolute TD error per episode."""
        curve = []
        errors_per_episode = len(self.td_errors) // max(1, len(self.value_history))
        for i in range(len(self.value_history)):
            start = i * errors_per_episode
            end = min((i + 1) * errors_per_episode, len(self.td_errors))
            if start < end:
                segment = self.td_errors[start:end]
                avg = sum(abs(e) for e in segment) / len(segment)
                curve.append(avg)
        return curve
    
    def reset(self) -> None:
        """Reset values and history."""
        self.values = [0.0] * self.num_states
        self.buffer = []
        self.buffer_terminal = False
        self.buffer_final_state = 0
        self.td_errors = []
        self.value_history = []
        self.step_count = 0
    
    def to_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "type": "NStepTD",
            "num_states": self.num_states,
            "n": self.n,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "values": self.values.copy(),
            "step_count": self.step_count,
            "td_errors": self.td_errors.copy(),
            "episodes_processed": len(self.value_history),
            "mean_td_error": sum(abs(e) for e in self.td_errors) / len(self.td_errors) if self.td_errors else 0.0
        }
