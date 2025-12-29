# RL Learning Tool - Technical Report

## Executive Summary

The RL Learning Tool is an interactive web-based application designed for learning and experimenting with reinforcement learning algorithms. The tool provides real-time visualization of training processes, allowing users to explore different algorithms and environments while adjusting parameters and observing their effects on learning dynamics.

---

## 1. Algorithms Implemented

### 1.1 Dynamic Programming (Model-Based)

These algorithms require complete knowledge of the environment's transition dynamics and are implemented using the Bellman equations.

#### 1.1.1 Policy Evaluation

- **Purpose**: Computes the state-value function V^π(s) for a given policy π
- **Bellman Equation**: V(s) = Σₐ π(a|s) × [R(s,a) + γ × V(s')]
- **Convergence**: Iterates until max|V_new(s) - V_old(s)| < θ

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| γ (gamma) | Discount factor | 0 - 1 | 0.99 |
| θ (theta) | Convergence threshold | 0.000001 - 0.1 | 0.0001 |
| maxIterations | Maximum iterations | 1 - 10000 | 1000 |

#### 1.1.2 Policy Iteration

- **Purpose**: Finds optimal policy by alternating evaluation and improvement
- **Policy Improvement**: π(s) = argmax_a [R(s,a) + γV(s')]
- **Guaranteed Convergence**: Converges to optimal policy in finite steps

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| γ (gamma) | Discount factor | 0 - 1 | 0.99 |
| θ (theta) | Convergence threshold | 0.000001 - 0.1 | 0.0001 |
| maxPolicyIterations | Max improvement steps | 1 - 100 | 100 |

#### 1.1.3 Value Iteration

- **Purpose**: Directly computes optimal value function V*
- **Bellman Optimality**: V(s) = max_a [R(s,a) + γ × V(s')]
- **Efficiency**: Combines evaluation and improvement in single step

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| γ (gamma) | Discount factor | 0 - 1 | 0.99 |
| θ (theta) | Convergence threshold | 0.000001 - 0.1 | 0.0001 |
| maxIterations | Maximum iterations | 1 - 10000 | 1000 |

---

### 1.2 Model-Free Methods

These algorithms learn from experience without requiring knowledge of transition dynamics.

#### 1.2.1 Monte Carlo Prediction

- **Purpose**: Learn value function from complete episode returns
- **Update Rule**: V(s) = average(G_t | S_t = s)
- **Characteristics**:
  - First-visit MC averages returns from complete episodes
  - Requires episodic environments
  - Unbiased but high variance

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| γ (gamma) | Discount factor | 0 - 1 | 0.99 |
| numEpisodes | Training episodes | 1 - 10000 | 100 |

#### 1.2.2 TD Learning (TD(0) and n-Step TD)

- **Purpose**: Bootstrap value estimates using temporal difference
- **TD(0) Update**: V(S) ← V(S) + α[R + γV(S') - V(S)]
- **n-Step TD**: Uses n-step returns for updates (bridges MC and TD)

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| tdType | TD(0) or n-step | select | td0 |
| nSteps | Steps for n-step TD | 1 - 20 | 3 |
| α (alpha) | Learning rate | 0.01 - 1 | 0.1 |
| γ (gamma) | Discount factor | 0 - 1 | 0.99 |
| numEpisodes | Training episodes | 1 - 10000 | 100 |

#### 1.2.3 SARSA (On-Policy TD Control)

- **Purpose**: Learn Q-values on-policy with ε-greedy exploration
- **Update Rule**: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
- **Characteristics**:
  - On-policy: learns about the policy being followed
  - Safer exploration (considers action actually taken)

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| α (alpha) | Learning rate | 0.01 - 1 | 0.1 |
| γ (gamma) | Discount factor | 0 - 1 | 0.99 |
| ε (epsilon) | Exploration rate | 0 - 1 | 0.1 |
| numEpisodes | Training episodes | 1 - 10000 | 100 |

#### 1.2.4 Q-Learning (Off-Policy TD Control)

- **Purpose**: Learn optimal Q-values off-policy
- **Update Rule**: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
- **Characteristics**:
  - Off-policy: learns optimal policy regardless of exploration
  - Uses max Q-value for TD target

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| α (alpha) | Learning rate | 0.01 - 1 | 0.1 |
| γ (gamma) | Discount factor | 0 - 1 | 0.99 |
| ε (epsilon) | Exploration rate | 0 - 1 | 0.1 |
| numEpisodes | Training episodes | 1 - 10000 | 100 |

---

## 2. Environments Implemented

### 2.1 GridWorld (Deterministic)

```
+----+----+----+----+
|  0 |  1 |  2 |  3 |
+----+----+----+----+
|  4 |  5 |  6 |  7 |
+----+----+----+----+
|  8 |  9 | 10 | 11 |
+----+----+----+----+
| 12 | 13 | 14 | G  |  ← Goal (state 15)
+----+----+----+----+
```

**Characteristics:**

- **State Space**: Configurable grid (rows × cols)
- **Action Space**: 4 actions (UP, DOWN, LEFT, RIGHT)
- **Transitions**: Deterministic - agent moves in intended direction
- **Rewards**: -1 per step (encourages shortest path), 0 at goal
- **Termination**: Goal reached OR max steps exceeded

| Configuration | Description | Range | Default |
|---------------|-------------|-------|---------|
| rows | Grid rows | 2 - 10 | 4 |
| cols | Grid columns | 2 - 10 | 4 |
| maxSteps | Max steps per episode | 10 - 500 | 100 |

---

### 2.2 FrozenLake (Stochastic)

```
+---+---+---+---+
| S | F | F | F |   S = Start (state 0)
+---+---+---+---+   F = Frozen (safe)
| F | H | F | H |   H = Hole (terminal)
+---+---+---+---+   G = Goal (terminal)
| F | F | F | H |
+---+---+---+---+
| H | F | F | G |
+---+---+---+---+
```

**Characteristics:**

- **State Space**: 4×4 grid (16 states) or 8×8 grid (64 states)
- **Action Space**: 4 actions (LEFT, DOWN, RIGHT, UP)
- **Transitions**: Stochastic (slippery ice)
  - 70% probability: Move in intended direction
  - 15% probability: Slip perpendicular (clockwise)
  - 15% probability: Slip perpendicular (counter-clockwise)
- **Rewards**: +1 at Goal, 0 otherwise
- **Termination**: Reach Goal OR fall in Hole

**DP Compatibility:**

- Transition probabilities fully exposed via `get_transition_prob(state, action)`
- Complete transition table accessible via `get_transition_table()`
- No hidden randomness - DP algorithms compute exact expected values

| Configuration | Description | Options | Default |
|---------------|-------------|---------|---------|
| mapName | Map size | "4x4", "8x8" | "4x4" |
| isSlippery | Stochastic transitions | true/false | true |

---

## 3. Parameter Adjustment Capabilities

### 3.1 Common Parameters

| Parameter | Symbol | Purpose | Impact |
|-----------|--------|---------|--------|
| **Discount Factor** | γ | Values future vs immediate rewards | Higher = more foresight |
| **Learning Rate** | α | Step size for updates | Higher = faster but unstable |
| **Exploration Rate** | ε | Random action probability | Higher = more exploration |
| **Convergence Threshold** | θ | Stopping criterion | Lower = more precise |

### 3.2 Real-Time Adjustment

- **Speed Control**: Adjustable animation speed (0.1x to 3x)
- **Step Mode**: Manual step-by-step execution
- **Run/Pause/Resume**: Full control over algorithm execution
- **Reset**: Restart with new parameters without page reload

### 3.3 Parameter Validation

The frontend validates all parameters:

- Range checking (min/max bounds)
- Type validation (int/float/boolean)
- Conditional parameters (e.g., nSteps only shown for n-step TD)

---

## 4. Visualization Techniques

### 4.1 GridWorld Canvas Visualization

**HTML5 Canvas-based rendering with:**

- **Animated agent movement**: Smooth interpolation between states
- **Value heatmap overlay**: Color-coded state values (green=positive, red=negative)
- **Policy arrows**: Directional arrows showing best actions
- **Environment markers**: Start (S), Goal (G), Holes (H) with distinct colors

**Color Scheme:**

| Element | Color | Description |
|---------|-------|-------------|
| Background | #1a1a2e | Dark theme |
| Start Cell | Cyan glow | rgba(0, 217, 255, 0.15) |
| Goal Cell | Green border | #00ff88 |
| Hole Cell | Dark icy blue | rgba(30, 60, 90, 0.9) |
| Agent | Cyan circle | #00d9ff with glow effect |
| Positive Values | Green | rgba(0, 255, 136) |
| Negative Values | Red | rgba(255, 107, 107) |

### 4.2 State Value Grid

Compact numeric display of V(s) or max_a Q(s,a) for all states:

- Color intensity proportional to value magnitude
- Real-time updates during training
- Responsive grid layout matching environment dimensions

### 4.3 Policy Visualization

Arrow-based policy display:

- ↑ ↓ ← → indicating best action per state
- Special markers for terminal states (G for goal)
- Updated after each policy improvement step

### 4.4 Learning Curves

Canvas-based line charts showing:

- **TD Error** (|δ|): For TD-based algorithms (TD, SARSA, Q-Learning)
- **Delta V** (ΔV): For DP algorithms (Policy Eval, Value Iteration)
- Auto-scaling axes
- Downsampling for large datasets (>200 points)

### 4.5 Diagnostics Panel

Real-time statistics display:

- **Iteration/Step counter**: Current progress
- **Episode counter**: For episodic algorithms
- **Convergence status**: ✓ Converged indicator
- **Mean TD Error**: Average absolute TD error
- **Max ΔV**: Maximum value change (convergence metric)
- **Total Reward**: Cumulative episode reward

---

## 5. System Architecture

### 5.1 Frontend (React + Vite)

```
frontend/
├── src/
│   ├── pages/           # Algorithm-specific pages
│   │   ├── PolicyEvaluation.jsx
│   │   ├── PolicyIteration.jsx
│   │   ├── ValueIteration.jsx
│   │   ├── MonteCarlo.jsx
│   │   ├── TDLearning.jsx
│   │   ├── SARSA.jsx
│   │   └── QLearning.jsx
│   ├── components/      # Reusable UI components
│   │   ├── GridWorldCanvas.jsx
│   │   ├── VisualizationPanel.jsx
│   │   ├── LearningCurve.jsx
│   │   ├── ParameterPanel.jsx
│   │   └── EnvironmentSelector.jsx
│   └── config/          # Configuration files
│       ├── algorithmParams.js
│       └── environments.js
```

### 5.2 Backend (FastAPI + Python)

```
backend/
├── api/
│   └── server.py        # REST API endpoints
├── algorithms/          # Algorithm implementations
│   ├── policy_evaluation.py
│   ├── policy_iteration.py
│   ├── value_iteration.py
│   ├── monte_carlo_prediction.py
│   ├── td_learning.py
│   ├── sarsa.py
│   └── q_learning.py
└── envs/                # Environment implementations
    ├── base_env.py
    ├── gridworld.py
    └── frozenlake.py
```

### 5.3 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | /api/sessions | Create new training session |
| POST | /api/sessions/{id}/step | Execute one algorithm step |
| POST | /api/sessions/{id}/pause | Pause execution |
| POST | /api/sessions/{id}/resume | Resume execution |
| POST | /api/sessions/{id}/reset | Reset to initial state |
| DELETE | /api/sessions/{id} | Delete session |

---

## 6. Algorithm-Environment Compatibility

| Algorithm | GridWorld | FrozenLake | Requirements |
|-----------|:---------:|:----------:|--------------|
| Policy Evaluation | ✓ | ✓ | Known transition dynamics |
| Policy Iteration | ✓ | ✓ | Known transition dynamics |
| Value Iteration | ✓ | ✓ | Known transition dynamics |
| Monte Carlo | ✓ | ✓ | Episodic environment |
| TD Learning | ✓ | ✓ | None |
| SARSA | ✓ | ✓ | None |
| Q-Learning | ✓ | ✓ | None |

---

## 7. Conclusion

The RL Learning Tool provides a comprehensive platform for understanding reinforcement learning concepts through:

1. **Seven core algorithms** spanning Dynamic Programming and Model-Free methods
2. **Two distinct environments** demonstrating deterministic and stochastic dynamics
3. **Extensive parameter control** for experimenting with algorithm behavior
4. **Rich visualizations** including animated grids, value heatmaps, policy arrows, and learning curves

The tool successfully bridges theory and practice, allowing users to observe the inner workings of RL algorithms in real-time while developing intuition for parameter effects and algorithm behavior.

---

*Report generated for the RL Learning Tool Project*
*December 2024*
