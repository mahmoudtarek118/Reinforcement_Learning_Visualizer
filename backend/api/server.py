"""
FastAPI Server for the RL Learning Tool.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uuid
import random

from backend.envs.gridworld import GridWorld
from backend.envs.frozenlake import FrozenLake
from backend.algorithms.policy_evaluation import PolicyEvaluation
from backend.algorithms.policy_iteration import PolicyIteration
from backend.algorithms.value_iteration import ValueIteration
from backend.algorithms.monte_carlo_prediction import MonteCarloPrediction
from backend.algorithms.td_learning import TD0, NStepTD
from backend.algorithms.sarsa import SARSA
from backend.algorithms.q_learning import QLearning

app = FastAPI(title="RL Learning Tool API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Dict[str, Any]] = {}


class EnvironmentConfig(BaseModel):
    type: str = "gridworld"
    rows: int = 4
    cols: int = 4
    max_steps: int = 100
    map_name: str = "4x4"  # For FrozenLake
    is_slippery: bool = True  # For FrozenLake


class AlgorithmConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}


class CreateSessionRequest(BaseModel):
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig


def create_environment(config: EnvironmentConfig):
    if config.type == "gridworld":
        return GridWorld(rows=config.rows, cols=config.cols, max_steps=config.max_steps)
    elif config.type == "frozenlake":
        return FrozenLake(map_name=config.map_name, is_slippery=config.is_slippery, max_steps=config.max_steps)
    raise ValueError(f"Unknown environment: {config.type}")


def create_transition_func(env):
    """
    Create transition function for DP algorithms.
    
    For FrozenLake (stochastic): Uses expected values from transition probabilities.
    For GridWorld (deterministic): Simulates the step directly.
    """
    if isinstance(env, FrozenLake):
        # FrozenLake has stochastic transitions
        # Return expected next state based on probabilities
        def transition_func(state: int, action: int):
            # Get transition probabilities for this state-action
            probs = env.get_transition_prob(state, action)
            rewards = env.get_transition_reward(state, action)
            
            # For DP, we need to compute expected value over transitions
            # But the standard DP interface expects a single (next_state, reward, done)
            # So we return the most likely transition for visualization,
            # but the DP algorithm should use get_transition_prob directly
            
            # Find most likely next state
            most_likely = max(probs.keys(), key=lambda s: probs[s])
            reward = rewards[most_likely]
            done = env.is_terminal(most_likely)
            
            return most_likely, reward, done
        return transition_func
    else:
        # GridWorld is deterministic
        def transition_func(state: int, action: int):
            # Save current environment state
            old_state = env.current_state
            old_steps = env.steps_taken
            
            # Set environment to the query state
            env.current_state = state
            env.steps_taken = 0
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            
            # Restore environment state
            env.current_state = old_state
            env.steps_taken = old_steps
            
            return next_state, reward, done
        return transition_func


def create_uniform_policy(num_states: int, num_actions: int):
    """Create uniform random policy."""
    prob = 1.0 / num_actions
    return [[prob] * num_actions for _ in range(num_states)]


@app.get("/")
def root():
    return {"status": "ok", "service": "RL Learning Tool API"}


@app.get("/api/environments")
def list_environments():
    return {"environments": [
        {"id": "gridworld", "name": "GridWorld", "available": True},
        {"id": "frozenlake", "name": "FrozenLake", "available": True}
    ]}


@app.get("/api/algorithms")
def list_algorithms():
    return {"algorithms": [
        {"id": "policy-evaluation", "name": "Policy Evaluation"},
        {"id": "policy-iteration", "name": "Policy Iteration"},
        {"id": "value-iteration", "name": "Value Iteration"},
        {"id": "monte-carlo", "name": "Monte Carlo"},
        {"id": "td-learning", "name": "TD Learning"},
        {"id": "sarsa", "name": "SARSA"},
        {"id": "q-learning", "name": "Q-Learning"},
    ]}


@app.post("/api/sessions")
def create_session(request: CreateSessionRequest):
    session_id = str(uuid.uuid4())[:8]
    
    try:
        env = create_environment(request.environment)
        algo_type = request.algorithm.type
        params = request.algorithm.params
        
        num_states = env.get_state_space_size()
        num_actions = env.get_action_space_size()
        state = env.reset()
        
        # Create algorithm
        if algo_type == "policy-evaluation":
            algo = PolicyEvaluation(num_states, num_actions, params.get("gamma", 0.99), params.get("theta", 0.0001))
        elif algo_type == "policy-iteration":
            algo = PolicyIteration(num_states, num_actions, params.get("gamma", 0.99), params.get("theta", 0.0001))
        elif algo_type == "value-iteration":
            algo = ValueIteration(num_states, num_actions, params.get("gamma", 0.99), params.get("theta", 0.0001))
        elif algo_type == "monte-carlo":
            algo = MonteCarloPrediction(num_states, params.get("gamma", 0.99))
        elif algo_type == "td-learning":
            if params.get("tdType") == "nstep":
                algo = NStepTD(num_states, params.get("nSteps", 3), params.get("alpha", 0.1), params.get("gamma", 0.99))
            else:
                algo = TD0(num_states, params.get("alpha", 0.1), params.get("gamma", 0.99))
        elif algo_type == "sarsa":
            algo = SARSA(num_states, num_actions, params.get("alpha", 0.1), params.get("gamma", 0.99), params.get("epsilon", 0.1))
        elif algo_type == "q-learning":
            algo = QLearning(num_states, num_actions, params.get("alpha", 0.1), params.get("gamma", 0.99), params.get("epsilon", 0.1))
        else:
            raise ValueError(f"Unknown algorithm: {algo_type}")
        
        sessions[session_id] = {
            "id": session_id,
            "environment": env,
            "algorithm": algo,
            "algorithm_type": algo_type,
            "transition_func": create_transition_func(env),
            "policy": create_uniform_policy(num_states, num_actions),
            "step": 0,
            "episode": 0,
            "total_reward": 0,
            "current_state": state,
            "episode_history": [],
            "converged": False,
            "paused": False
        }
        
        return {
            "session_id": session_id,
            "status": "ready",
            "agent_position": {"row": state // env.cols, "col": state % env.cols},
            "grid_size": {"rows": env.rows, "cols": env.cols}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sessions/{session_id}/step")
def step_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    env = session["environment"]
    algo = session["algorithm"]
    algo_type = session["algorithm_type"]
    
    if session["paused"]:
        return {"status": "paused"}
    
    # DP algorithms
    if algo_type in ["policy-evaluation", "policy-iteration", "value-iteration"]:
        if not session["converged"]:
            policy = session["policy"]
            is_stochastic = isinstance(env, FrozenLake)
            
            old_values = algo.values.copy()
            max_delta = 0.0
            
            if algo_type == "value-iteration":
                # Bellman optimality update
                for state in range(algo.num_states):
                    q_values = []
                    for action in range(algo.num_actions):
                        if is_stochastic:
                            # FrozenLake: compute expected value over all transitions
                            transitions = env.P[state][action]
                            q_val = sum(
                                prob * (reward + (0 if done else algo.gamma * old_values[next_s]))
                                for prob, next_s, reward, done in transitions
                            )
                        else:
                            # GridWorld: deterministic transition
                            next_state, reward, done = session["transition_func"](state, action)
                            q_val = reward if done else reward + algo.gamma * old_values[next_state]
                        q_values.append(q_val)
                    algo.values[state] = max(q_values)
                    max_delta = max(max_delta, abs(algo.values[state] - old_values[state]))
                
                # Extract greedy policy
                new_policy = []
                for state in range(algo.num_states):
                    q_values = []
                    for action in range(algo.num_actions):
                        if is_stochastic:
                            transitions = env.P[state][action]
                            q_val = sum(
                                prob * (reward + (0 if done else algo.gamma * algo.values[next_s]))
                                for prob, next_s, reward, done in transitions
                            )
                        else:
                            next_state, reward, done = session["transition_func"](state, action)
                            q_val = reward if done else reward + algo.gamma * algo.values[next_state]
                        q_values.append(q_val)
                    best = q_values.index(max(q_values))
                    probs = [0.0] * algo.num_actions
                    probs[best] = 1.0
                    new_policy.append(probs)
                session["policy"] = new_policy
            else:
                # Policy evaluation / iteration
                for state in range(algo.num_states):
                    new_value = 0.0
                    for action in range(algo.num_actions):
                        prob = policy[state][action]
                        if prob == 0:
                            continue
                        if is_stochastic:
                            # FrozenLake: compute expected value over all transitions
                            transitions = env.P[state][action]
                            action_val = sum(
                                t_prob * (reward + (0 if done else algo.gamma * old_values[next_s]))
                                for t_prob, next_s, reward, done in transitions
                            )
                        else:
                            # GridWorld: deterministic transition
                            next_state, reward, done = session["transition_func"](state, action)
                            action_val = reward if done else reward + algo.gamma * old_values[next_state]
                        new_value += prob * action_val
                    algo.values[state] = new_value
                    max_delta = max(max_delta, abs(new_value - old_values[state]))
                
                if algo_type == "policy-iteration":
                    # Policy improvement
                    new_policy = []
                    for state in range(algo.num_states):
                        q_values = []
                        for action in range(algo.num_actions):
                            if is_stochastic:
                                transitions = env.P[state][action]
                                q_val = sum(
                                    prob * (reward + (0 if done else algo.gamma * algo.values[next_s]))
                                    for prob, next_s, reward, done in transitions
                                )
                            else:
                                next_state, reward, done = session["transition_func"](state, action)
                                q_val = reward if done else reward + algo.gamma * algo.values[next_state]
                            q_values.append(q_val)
                        best = q_values.index(max(q_values))
                        probs = [0.0] * algo.num_actions
                        probs[best] = 1.0
                        new_policy.append(probs)
                    session["policy"] = new_policy
            
            algo.iteration_history.append({"iteration": session["step"], "max_delta": max_delta})
            session["converged"] = max_delta < algo.theta
            session["step"] += 1
        
        policy = session["policy"]
        optimal_policy = [row.index(max(row)) for row in policy]
        
        return {
            "session_id": session_id,
            "status": "converged" if session["converged"] else "running",
            "step": session["step"],
            "agent_position": {"row": 0, "col": 0},
            "done": session["converged"],
            "diagnostics": {
                "values": algo.values,
                "policy": optimal_policy,
                "delta_history": [h["max_delta"] for h in algo.iteration_history],
                "max_delta": algo.iteration_history[-1]["max_delta"] if algo.iteration_history else 0
            }
        }
    
    # Model-free algorithms
    else:
        state = session["current_state"]
        
        if algo_type in ["sarsa", "q-learning"]:
            action = algo.select_action(state)
        else:
            action = random.randint(0, env.get_action_space_size() - 1)
        
        next_state, reward, done, _ = env.step(action)
        session["episode_history"].append((state, action, reward))
        session["total_reward"] += reward
        session["step"] += 1
        
        if algo_type == "sarsa":
            next_action = algo.select_action(next_state) if not done else 0
            algo.update(state, action, reward, next_state, next_action, done)
        elif algo_type == "q-learning":
            algo.update(state, action, reward, next_state, done)
        elif algo_type == "td-learning":
            algo.update(state, reward, next_state, done)
        
        if done:
            if algo_type == "monte-carlo":
                algo.process_episode(session["episode_history"])
            session["episode"] += 1
            session["episode_history"] = []
            session["current_state"] = env.reset()
        else:
            session["current_state"] = next_state
        
        return {
            "session_id": session_id,
            "status": "running",
            "step": session["step"],
            "episode": session["episode"],
            "agent_position": {"row": session["current_state"] // env.cols, "col": session["current_state"] % env.cols},
            "reward": reward,
            "total_reward": session["total_reward"],
            "done": done,
            "diagnostics": algo.to_diagnostics()
        }


@app.post("/api/sessions/{session_id}/pause")
def pause_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["paused"] = True
    return {"status": "paused"}


@app.post("/api/sessions/{session_id}/resume")
def resume_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["paused"] = False
    return {"status": "resumed"}


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    env = session["environment"]
    state = env.reset()
    
    # Reset algorithm values
    algo = session["algorithm"]
    algo.values = [0.0] * algo.num_states
    algo.iteration_history = []
    
    session["current_state"] = state
    session["step"] = 0
    session["episode"] = 0
    session["total_reward"] = 0
    session["episode_history"] = []
    session["converged"] = False
    session["paused"] = False
    session["policy"] = create_uniform_policy(algo.num_states, algo.num_actions)
    
    return {"status": "reset", "agent_position": {"row": 0, "col": 0}}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"status": "deleted"}
