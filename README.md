# RL Learning Tool

An interactive web-based tool for learning reinforcement learning algorithms with real-time visualization.

## Features

- **7 RL Algorithms**: Policy Evaluation, Policy Iteration, Value Iteration, Monte Carlo, TD Learning, SARSA, Q-Learning
- **2 Environments**: GridWorld (deterministic), FrozenLake (stochastic)
- **Real-time Visualization**: Animated agent, value heatmaps, policy arrows, learning curves
- **Interactive Parameters**: Adjust γ, α, ε, θ and see effects in real-time

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/mahmoudtarek118/Reinforcement_Learning_Visualizer.git
cd Reinforcement_Learning_Visualizer
```

### Backend (FastAPI)

```bash
cd RLbouns
pip install fastapi uvicorn pydantic
python -m uvicorn backend.api.server:app --reload --port 8000
```

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173> in your browser.

## Project Structure

```
RLbouns/
├── backend/
│   ├── api/server.py          # FastAPI endpoints
│   ├── algorithms/            # RL algorithm implementations
│   └── envs/                  # Environment implementations
├── frontend/
│   ├── src/pages/             # Algorithm pages
│   └── src/components/        # UI components
└── REPORT.md                  # Technical documentation
```

## Documentation

See [REPORT.md](REPORT.md) for detailed documentation on algorithms, environments, and visualization techniques.
