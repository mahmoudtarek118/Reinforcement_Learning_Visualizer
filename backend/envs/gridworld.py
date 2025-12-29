"""
GridWorld Environment Implementation.

A deterministic grid-based environment for reinforcement learning education.

STATE MAPPING:
    States are represented as integer IDs computed from (row, col) coordinates.
    
    For a grid of size (rows, cols):
        state_id = row * cols + col
        
    Example for a 4x4 grid:
        +----+----+----+----+
        |  0 |  1 |  2 |  3 |
        +----+----+----+----+
        |  4 |  5 |  6 |  7 |
        +----+----+----+----+
        |  8 |  9 | 10 | 11 |
        +----+----+----+----+
        | 12 | 13 | 14 | 15 |  <- State 15 is the goal (terminal) state
        +----+----+----+----+
    
    To convert state_id back to coordinates:
        row = state_id // cols
        col = state_id % cols

ACTIONS:
    0 = UP    (row - 1)
    1 = DOWN  (row + 1)
    2 = LEFT  (col - 1)
    3 = RIGHT (col + 1)
    
    If an action would move the agent outside the grid, the agent stays in place.

REWARDS:
    -1 for each step (encourages finding shortest path)
    0 at terminal state (no penalty for reaching goal)

EPISODE TERMINATION:
    - Agent reaches the goal state (bottom-right corner), OR
    - Maximum number of steps is exceeded
"""

from typing import Any, Dict, Tuple

from backend.envs.base_env import BaseEnv


# Action constants for clarity
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class GridWorld(BaseEnv):
    """
    A deterministic GridWorld environment.
    
    The agent starts at the top-left corner (state 0) and must navigate
    to the bottom-right corner (goal state). Each step incurs a reward of -1,
    except reaching the goal which gives 0 reward.
    
    Attributes:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        max_steps (int): Maximum steps before episode termination.
        current_state (int): Current state ID of the agent.
        goal_state (int): State ID of the goal (terminal) state.
        steps_taken (int): Number of steps taken in current episode.
    """
    
    def __init__(self, rows: int = 4, cols: int = 4, max_steps: int = 100):
        """
        Initialize the GridWorld environment.
        
        Args:
            rows: Number of rows in the grid (default: 4).
            cols: Number of columns in the grid (default: 4).
            max_steps: Maximum steps per episode before termination (default: 100).
        """
        # Validate inputs
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2x2")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        
        # Store grid dimensions
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        
        # Goal state is the bottom-right corner
        # Using state_id = row * cols + col formula
        self.goal_state = (rows - 1) * cols + (cols - 1)
        
        # Initialize episode state
        self.current_state = 0  # Start at top-left (row=0, col=0)
        self.steps_taken = 0
    
    def reset(self) -> int:
        """
        Reset the environment to initial state.
        
        The agent is placed at the top-left corner (state 0).
        
        Returns:
            int: Initial state ID (always 0).
        """
        self.current_state = 0  # Top-left corner
        self.steps_taken = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute an action and return the result.
        
        The transition is deterministic: the agent moves in the specified
        direction if possible, otherwise stays in place.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).
            
        Returns:
            Tuple containing:
                - next_state: New state ID after action.
                - reward: -1 for each step, 0 at terminal.
                - done: True if goal reached or max steps exceeded.
                - info: Dict with 'reason' key explaining termination.
        """
        # Validate action
        if action not in [UP, DOWN, LEFT, RIGHT]:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")
        
        # Convert current state ID to row, col coordinates
        current_row = self.current_state // self.cols
        current_col = self.current_state % self.cols
        
        # Calculate new position based on action
        new_row = current_row
        new_col = current_col
        
        if action == UP:
            new_row = current_row - 1
        elif action == DOWN:
            new_row = current_row + 1
        elif action == LEFT:
            new_col = current_col - 1
        elif action == RIGHT:
            new_col = current_col + 1
        
        # Check boundaries - if outside grid, stay in place
        if new_row < 0 or new_row >= self.rows:
            new_row = current_row  # Revert to original row
        if new_col < 0 or new_col >= self.cols:
            new_col = current_col  # Revert to original column
        
        # Convert new position back to state ID
        self.current_state = new_row * self.cols + new_col
        
        # Increment step counter
        self.steps_taken += 1
        
        # Check if episode is done
        reached_goal = self.current_state == self.goal_state
        exceeded_max_steps = self.steps_taken >= self.max_steps
        done = reached_goal or exceeded_max_steps
        
        # Determine reward: 0 at terminal, -1 otherwise
        if reached_goal:
            reward = 0.0
        else:
            reward = -1.0
        
        # Build info dict
        info = {
            "row": new_row,
            "col": new_col,
            "steps_taken": self.steps_taken,
            "reason": ""
        }
        if reached_goal:
            info["reason"] = "goal_reached"
        elif exceeded_max_steps:
            info["reason"] = "max_steps_exceeded"
        
        return self.current_state, reward, done, info
    
    def get_state(self) -> int:
        """
        Get the current state ID.
        
        Returns:
            int: Current state ID.
        """
        return self.current_state
    
    def get_valid_actions(self) -> list:
        """
        Get all valid actions from current state.
        
        In GridWorld, all 4 actions are always valid (the agent just
        stays in place if it tries to move outside the grid).
        
        Returns:
            list: List of valid action IDs [0, 1, 2, 3].
        """
        return [UP, DOWN, LEFT, RIGHT]
    
    def get_state_space_size(self) -> int:
        """
        Get total number of states.
        
        Returns:
            int: rows * cols (total grid cells).
        """
        return self.rows * self.cols
    
    def get_action_space_size(self) -> int:
        """
        Get total number of actions.
        
        Returns:
            int: 4 (UP, DOWN, LEFT, RIGHT).
        """
        return 4
    
    def state_to_coords(self, state_id: int) -> Tuple[int, int]:
        """
        Convert state ID to (row, col) coordinates.
        
        Args:
            state_id: The state ID to convert.
            
        Returns:
            Tuple[int, int]: (row, col) coordinates.
        """
        row = state_id // self.cols
        col = state_id % self.cols
        return row, col
    
    def coords_to_state(self, row: int, col: int) -> int:
        """
        Convert (row, col) coordinates to state ID.
        
        Args:
            row: Row index (0-indexed).
            col: Column index (0-indexed).
            
        Returns:
            int: Corresponding state ID.
        """
        return row * self.cols + col
