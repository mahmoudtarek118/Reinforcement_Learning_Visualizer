"""
FrozenLake Environment Implementation.

A stochastic grid-based environment with slippery ice for reinforcement learning education.

ENVIRONMENT LAYOUT (4x4 default):
    +---+---+---+---+
    | S | F | F | F |   S = Start (state 0)
    +---+---+---+---+   F = Frozen (safe)
    | F | H | F | H |   H = Hole (terminal, failure)
    +---+---+---+---+   G = Goal (terminal, success)
    | F | F | F | H |
    +---+---+---+---+
    | H | F | F | G |
    +---+---+---+---+

STATE MAPPING:
    States are integer IDs: state_id = row * cols + col
    
    For 4x4 grid:
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15
    
    Holes: [5, 7, 11, 12]
    Goal: 15

ACTIONS:
    0 = LEFT
    1 = DOWN
    2 = RIGHT
    3 = UP

SLIPPERY TRANSITIONS (Stochastic):
    When the ice is slippery (is_slippery=True), the agent may slip:
    - 70% probability: Move in intended direction
    - 15% probability: Slip perpendicular (clockwise)
    - 15% probability: Slip perpendicular (counter-clockwise)
    
    Example: If action is DOWN (1):
        - 70% chance: Move DOWN
        - 15% chance: Slip LEFT (perpendicular)
        - 15% chance: Slip RIGHT (perpendicular)

    This stochasticity is FULLY EXPOSED via get_transition_prob() for DP algorithms.
    No hidden randomness - DP algorithms can compute expected values exactly.

REWARDS:
    +1.0 for reaching the Goal
    0.0 for all other transitions (including falling in holes)

EPISODE TERMINATION:
    Episode ends when agent reaches Goal or falls in a Hole.
"""

import random
from typing import Any, Dict, List, Tuple

from backend.envs.base_env import BaseEnv


# Action constants
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Default 4x4 FrozenLake map
# S = Start, F = Frozen (safe), H = Hole, G = Goal
MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
}


class FrozenLake(BaseEnv):
    """
    FrozenLake environment with slippery (stochastic) transitions.
    
    The agent navigates a frozen lake, trying to reach a goal while
    avoiding holes. The ice is slippery, so movements are stochastic.
    
    CRITICAL FOR DP ALGORITHMS:
        This environment exposes transition probabilities via:
        - get_transition_prob(state, action): Returns {next_state: probability}
        - get_transition_table(): Returns full P[s][a] matrix
        
        These methods allow DP algorithms (Policy Evaluation, Policy Iteration,
        Value Iteration) to compute exact expected values without sampling.
    
    Attributes:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        is_slippery (bool): Whether transitions are stochastic.
        current_state (int): Current state ID of the agent.
        start_state (int): Starting state ID.
        goal_states (set): Set of goal state IDs.
        hole_states (set): Set of hole state IDs.
        P (dict): Transition probability table P[s][a] = [(prob, next_s, reward, done)]
    """
    
    def __init__(self, map_name: str = "4x4", is_slippery: bool = True, max_steps: int = 100):
        """
        Initialize the FrozenLake environment.
        
        Args:
            map_name: Name of the map to use ("4x4" or "8x8").
            is_slippery: If True, transitions are stochastic (70/15/15 split).
                        If False, transitions are deterministic.
            max_steps: Maximum steps per episode before termination.
        """
        if map_name not in MAPS:
            raise ValueError(f"Unknown map: {map_name}. Available: {list(MAPS.keys())}")
        
        self.map_desc = MAPS[map_name]
        self.rows = len(self.map_desc)
        self.cols = len(self.map_desc[0])
        self.is_slippery = is_slippery
        self.max_steps = max_steps
        
        # Parse the map to find special states
        self.start_state = None
        self.goal_states = set()
        self.hole_states = set()
        
        for row in range(self.rows):
            for col in range(self.cols):
                state_id = row * self.cols + col
                cell = self.map_desc[row][col]
                
                if cell == 'S':
                    self.start_state = state_id
                elif cell == 'G':
                    self.goal_states.add(state_id)
                elif cell == 'H':
                    self.hole_states.add(state_id)
        
        # Build the transition probability table
        # This is the KEY feature for DP algorithm compatibility
        self.P = self._build_transition_table()
        
        # Initialize episode state
        self.current_state = self.start_state
        self.steps_taken = 0
    
    def _build_transition_table(self) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """
        Build the complete transition probability table P[s][a].
        
        This method pre-computes all transition probabilities, making them
        fully accessible to DP algorithms. NO HIDDEN RANDOMNESS.
        
        Returns:
            Dict mapping state -> action -> list of (probability, next_state, reward, done)
            
        TRANSITION PROBABILITY EXPLANATION:
            For slippery ice:
                - Intended direction: 0.70 probability
                - Perpendicular slip (clockwise): 0.15 probability
                - Perpendicular slip (counter-clockwise): 0.15 probability
            
            For non-slippery ice:
                - Intended direction: 1.0 probability
        """
        n_states = self.rows * self.cols
        n_actions = 4
        
        P = {}
        
        for state in range(n_states):
            P[state] = {}
            
            row = state // self.cols
            col = state % self.cols
            
            # Check if this is a terminal state (hole or goal)
            is_terminal = state in self.hole_states or state in self.goal_states
            
            for action in range(n_actions):
                transitions = []
                
                if is_terminal:
                    # Terminal states: stay in place with 0 reward
                    transitions.append((1.0, state, 0.0, True))
                else:
                    if self.is_slippery:
                        # Slippery: 70% intended, 15% slip each perpendicular direction
                        # Perpendicular directions depend on the action
                        intended_probs = [(0.70, action)]
                        
                        # Get perpendicular actions
                        # For LEFT/RIGHT: perpendicular is UP/DOWN
                        # For UP/DOWN: perpendicular is LEFT/RIGHT
                        if action in [LEFT, RIGHT]:
                            perp_actions = [UP, DOWN]
                        else:  # UP or DOWN
                            perp_actions = [LEFT, RIGHT]
                        
                        intended_probs.append((0.15, perp_actions[0]))
                        intended_probs.append((0.15, perp_actions[1]))
                        
                        # Calculate transitions for each possible movement
                        for prob, move_action in intended_probs:
                            next_state = self._get_next_state(row, col, move_action)
                            reward = 1.0 if next_state in self.goal_states else 0.0
                            done = next_state in self.goal_states or next_state in self.hole_states
                            transitions.append((prob, next_state, reward, done))
                    else:
                        # Deterministic: 100% intended direction
                        next_state = self._get_next_state(row, col, action)
                        reward = 1.0 if next_state in self.goal_states else 0.0
                        done = next_state in self.goal_states or next_state in self.hole_states
                        transitions.append((1.0, next_state, reward, done))
                
                # Merge duplicate next_states (can happen when hitting walls)
                merged_transitions = self._merge_transitions(transitions)
                P[state][action] = merged_transitions
        
        return P
    
    def _merge_transitions(self, transitions: List[Tuple[float, int, float, bool]]) -> List[Tuple[float, int, float, bool]]:
        """
        Merge transitions that lead to the same next_state.
        
        This handles cases where multiple slip directions lead to the same
        state (e.g., hitting a wall from different directions).
        
        Args:
            transitions: List of (prob, next_state, reward, done) tuples.
            
        Returns:
            Merged list with unique next_states.
        """
        merged = {}
        for prob, next_state, reward, done in transitions:
            if next_state in merged:
                # Add probabilities for same next_state
                old_prob, _, _, _ = merged[next_state]
                merged[next_state] = (old_prob + prob, next_state, reward, done)
            else:
                merged[next_state] = (prob, next_state, reward, done)
        
        return list(merged.values())
    
    def _get_next_state(self, row: int, col: int, action: int) -> int:
        """
        Compute the next state given current position and action.
        
        If the action would move outside the grid, the agent stays in place.
        
        Args:
            row: Current row.
            col: Current column.
            action: Action to take.
            
        Returns:
            int: Next state ID.
        """
        new_row, new_col = row, col
        
        if action == LEFT:
            new_col = max(0, col - 1)
        elif action == DOWN:
            new_row = min(self.rows - 1, row + 1)
        elif action == RIGHT:
            new_col = min(self.cols - 1, col + 1)
        elif action == UP:
            new_row = max(0, row - 1)
        
        return new_row * self.cols + new_col
    
    def reset(self) -> int:
        """
        Reset the environment to the starting state.
        
        Returns:
            int: The starting state ID.
        """
        self.current_state = self.start_state
        self.steps_taken = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute an action using stochastic transitions.
        
        For model-free algorithms, this samples from the transition
        distribution. The same distribution is exposed via get_transition_prob()
        for DP algorithms.
        
        Args:
            action: Action to take (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP).
            
        Returns:
            Tuple containing:
                - next_state: New state ID after action.
                - reward: Reward received (+1 at goal, 0 otherwise).
                - done: True if episode ended (goal or hole).
                - info: Additional information about the transition.
        """
        if action not in [LEFT, DOWN, RIGHT, UP]:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")
        
        # Get transition probabilities for this state-action pair
        transitions = self.P[self.current_state][action]
        
        # Sample from the transition distribution
        probs = [t[0] for t in transitions]
        idx = random.choices(range(len(transitions)), weights=probs, k=1)[0]
        
        prob, next_state, reward, done = transitions[idx]
        
        self.current_state = next_state
        self.steps_taken += 1
        
        # Check max steps
        if self.steps_taken >= self.max_steps:
            done = True
        
        # Build info dict
        row = next_state // self.cols
        col = next_state % self.cols
        info = {
            "row": row,
            "col": col,
            "steps_taken": self.steps_taken,
            "prob": prob,
            "is_hole": next_state in self.hole_states,
            "is_goal": next_state in self.goal_states,
            "reason": ""
        }
        if next_state in self.goal_states:
            info["reason"] = "goal_reached"
        elif next_state in self.hole_states:
            info["reason"] = "fell_in_hole"
        elif self.steps_taken >= self.max_steps:
            info["reason"] = "max_steps_exceeded"
        
        return next_state, reward, done, info
    
    def get_state(self) -> int:
        """Get the current state ID."""
        return self.current_state
    
    def get_valid_actions(self) -> list:
        """Get all valid actions (always 0-3 in FrozenLake)."""
        return [LEFT, DOWN, RIGHT, UP]
    
    def get_state_space_size(self) -> int:
        """Get total number of states."""
        return self.rows * self.cols
    
    def get_action_space_size(self) -> int:
        """Get total number of actions (4)."""
        return 4
    
    # =====================================================================
    # DP ALGORITHM SUPPORT METHODS
    # These methods expose transition probabilities for exact computation
    # =====================================================================
    
    def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities for a state-action pair.
        
        THIS IS THE KEY METHOD FOR DP ALGORITHMS.
        It returns the exact probability distribution over next states,
        allowing Policy Evaluation, Policy Iteration, and Value Iteration
        to compute expected values without sampling.
        
        Args:
            state: Current state ID.
            action: Action ID.
            
        Returns:
            Dict mapping next_state -> probability.
            All probabilities sum to 1.0.
            
        Example:
            >>> env = FrozenLake(is_slippery=True)
            >>> probs = env.get_transition_prob(0, DOWN)  # State 0, action DOWN
            >>> # Returns: {4: 0.70, 0: 0.15, 1: 0.15}
            >>> # 70% move down to state 4
            >>> # 15% slip left (wall, stay at 0)
            >>> # 15% slip right to state 1
        """
        transitions = self.P[state][action]
        return {next_s: prob for prob, next_s, _, _ in transitions}
    
    def get_transition_reward(self, state: int, action: int) -> Dict[int, float]:
        """
        Get rewards for transitions from a state-action pair.
        
        Args:
            state: Current state ID.
            action: Action ID.
            
        Returns:
            Dict mapping next_state -> reward.
        """
        transitions = self.P[state][action]
        return {next_s: reward for _, next_s, reward, _ in transitions}
    
    def get_transition_table(self) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """
        Get the complete transition probability table.
        
        Returns:
            The full P[s][a] = [(prob, next_state, reward, done)] table.
            This is the same format used by OpenAI Gym's FrozenLake.
        """
        return self.P
    
    def is_terminal(self, state: int) -> bool:
        """
        Check if a state is terminal (hole or goal).
        
        Args:
            state: State ID to check.
            
        Returns:
            True if the state is a hole or goal.
        """
        return state in self.hole_states or state in self.goal_states
    
    def state_to_coords(self, state_id: int) -> Tuple[int, int]:
        """Convert state ID to (row, col) coordinates."""
        row = state_id // self.cols
        col = state_id % self.cols
        return row, col
    
    def coords_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to state ID."""
        return row * self.cols + col
    
    def get_cell_type(self, state: int) -> str:
        """
        Get the type of cell at a state.
        
        Returns:
            'S' (start), 'F' (frozen), 'H' (hole), or 'G' (goal).
        """
        row, col = self.state_to_coords(state)
        return self.map_desc[row][col]
    
    def render(self) -> str:
        """
        Render the current state of the environment as text.
        
        Returns:
            String representation of the grid with agent position marked.
        """
        output = []
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                state = row * self.cols + col
                if state == self.current_state:
                    line += "A"  # Agent
                else:
                    line += self.map_desc[row][col]
            output.append(line)
        return "\n".join(output)
