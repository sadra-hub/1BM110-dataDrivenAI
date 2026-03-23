"""
Part 2.2: Training and Hyperparameter Tuning for Bounded Knapsack Environment
Fixed version: uses EvalCallback to avoid environment reset issues, reduces memory usage.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from knapsack_env import BoundedKnapsackEnv

# =============================================================================
# Configuration
# =============================================================================
ENV_CONFIG = {
    "n_items": 200,
    "max_weight": 200,
    "randomize_params_on_reset": False,
    "mask": False
}

TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 1000          # evaluate every 1000 steps
N_EVAL_EPISODES = 10
SEEDS = [42, 123, 456]

LOG_DIR = "./logs/"
FIGURE_DIR = "./figures/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# =============================================================================
# Helper functions
# =============================================================================
def make_env(seed: int) -> gym.Env:
    """Create and wrap the environment with Monitor."""
    env = BoundedKnapsackEnv(**ENV_CONFIG)
    env.reset(seed=seed)
    env = Monitor(env, filename=os.path.join(LOG_DIR, f"seed_{seed}"))
    return env

def evaluate_agent(model, env, n_episodes=10) -> float:
    """Evaluate the agent and return mean true reward (unscaled)."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        true_reward = total_reward * 100.0
        rewards.append(true_reward)
    return np.mean(rewards)

def train_and_get_curves(algorithm: str, hyperparams: Dict[str, Any],
                         seeds: List[int], total_timesteps: int,
                         eval_freq: int, n_eval_episodes: int) -> Dict[int, List[float]]:
    per_seed_curves = {seed: [] for seed in seeds}

    for seed in seeds:
        env = make_env(seed)

        if algorithm == "DQN":
            model = DQN("MlpPolicy", env, verbose=0, seed=seed, **hyperparams)
        elif algorithm == "PPO":
            model = PPO("MlpPolicy", env, verbose=0, seed=seed, **hyperparams)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        eval_callback = EvalCallback(
            env,
            best_model_save_path=None,
            log_path=LOG_DIR,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=0,
            callback_on_new_best=None
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        # Extract evaluation results from the callback
        if hasattr(eval_callback, '_evaluations') and eval_callback._evaluations is not None:
            # _evaluations shape: (n_evaluations, n_eval_episodes)
            mean_rewards = np.mean(eval_callback._evaluations, axis=1) * 100.0
            per_seed_curves[seed] = mean_rewards.tolist()
        else:
            # Fallback: evaluate at the end
            final = evaluate_agent(model, env, n_eval_episodes)
            per_seed_curves[seed] = [final]

        env.close()

    return per_seed_curves

def plot_curves(curves: Dict[int, List[float]], title: str, ylabel: str, save_path: str):
    """Plot mean and std of curves across seeds."""
    # Convert to numpy array: (n_seeds, n_points)
    seeds = list(curves.keys())
    # Ensure all lists have same length (if not, pad with last value)
    lengths = [len(curves[s]) for s in seeds]
    if len(set(lengths)) != 1:
        min_len = min(lengths)
        for s in seeds:
            if len(curves[s]) > min_len:
                curves[s] = curves[s][:min_len]
            elif len(curves[s]) < min_len:
                # pad with last value
                last_val = curves[s][-1]
                curves[s].extend([last_val] * (min_len - len(curves[s])))
    data = np.array([curves[s] for s in seeds])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(mean)) * (EVAL_FREQ // 1000)  # x-axis in thousands of steps
    plt.plot(x, mean, label='Mean')
    plt.fill_between(x, mean - std, mean + std, alpha=0.3)
    plt.xlabel('Training steps (thousands)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# 1. Default hyperparameters training
# =============================================================================
print("=== 1. Training with default hyperparameters ===")

# Default hyperparams from SB3, but reduced buffer size for DQN to avoid memory issues
default_dqn = {
    "learning_rate": 1e-4,
    "buffer_size": 50_000,          # reduced from 1,000,000
    "batch_size": 32,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
}
default_ppo = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "clip_range": 0.2,
}

dqn_curves = train_and_get_curves("DQN", default_dqn, SEEDS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES)
ppo_curves = train_and_get_curves("PPO", default_ppo, SEEDS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES)

# Plot individually
plot_curves(dqn_curves, "DQN - Default Hyperparameters", "True Reward", os.path.join(FIGURE_DIR, "default_dqn.png"))
plot_curves(ppo_curves, "PPO - Default Hyperparameters", "True Reward", os.path.join(FIGURE_DIR, "default_ppo.png"))

# Combined plot
plt.figure(figsize=(10, 6))
for name, curves in [("DQN", dqn_curves), ("PPO", ppo_curves)]:
    data = np.array([curves[s] for s in SEEDS])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.arange(len(mean)) * (EVAL_FREQ // 1000)
    plt.plot(x, mean, label=name)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
plt.xlabel('Training steps (thousands)')
plt.ylabel('True Reward')
plt.title('Default Hyperparameters Comparison')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIGURE_DIR, "default_comparison.png"))
plt.close()

# =============================================================================
# 2. Hyperparameter tuning (one-at-a-time)
# =============================================================================
print("\n=== 2. Hyperparameter tuning (one-at-a-time) ===")

def evaluate_final_reward(algorithm: str, hyperparams: Dict[str, Any], seeds: List[int]) -> Dict[int, float]:
    """Train and return the final true reward for each seed."""
    final_rewards = {}
    for seed in seeds:
        env = make_env(seed)
        if algorithm == "DQN":
            model = DQN("MlpPolicy", env, verbose=0, seed=seed, **hyperparams)
        else:
            model = PPO("MlpPolicy", env, verbose=0, seed=seed, **hyperparams)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        final = evaluate_agent(model, env, N_EVAL_EPISODES)
        final_rewards[seed] = final
        env.close()
    return final_rewards

# ------------------------------------------------------------
# DQN tuning
# ------------------------------------------------------------
print("\n--- DQN hyperparameter tuning ---")

# Start with default DQN hyperparameters (buffer size already reduced)
dqn_base = default_dqn.copy()

# 1. Tune learning rate
lr_vals = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
lr_results = {}
for lr in lr_vals:
    params = dqn_base.copy()
    params["learning_rate"] = lr
    final = evaluate_final_reward("DQN", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    lr_results[lr] = mean_final
    print(f"   DQN learning_rate={lr} -> mean final reward = {mean_final:.2f}")
best_lr = max(lr_results, key=lr_results.get)
print(f"Best DQN learning rate: {best_lr}")

# 2. Tune batch size (use best learning rate)
bs_vals = [16, 32, 64, 128, 256]
bs_results = {}
for bs in bs_vals:
    params = dqn_base.copy()
    params["learning_rate"] = best_lr
    params["batch_size"] = bs
    final = evaluate_final_reward("DQN", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    bs_results[bs] = mean_final
    print(f"   DQN batch_size={bs} -> mean final reward = {mean_final:.2f}")
best_bs = max(bs_results, key=bs_results.get)
print(f"Best DQN batch size: {best_bs}")

# Update base with best common hyperparams
dqn_base["learning_rate"] = best_lr
dqn_base["batch_size"] = best_bs

# 3. Tune buffer size
buffer_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
buffer_results = {}
for buf in buffer_sizes:
    params = dqn_base.copy()
    params["buffer_size"] = buf
    final = evaluate_final_reward("DQN", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    buffer_results[buf] = mean_final
    print(f"   DQN buffer_size={buf} -> mean final reward = {mean_final:.2f}")
best_buffer = max(buffer_results, key=buffer_results.get)
print(f"Best DQN buffer size: {best_buffer}")

# 4. Tune exploration initial epsilon
epsilons = [1.0, 0.9, 0.5, 0.3, 0.1]
eps_results = {}
for eps in epsilons:
    params = dqn_base.copy()
    params["buffer_size"] = best_buffer
    params["exploration_initial_eps"] = eps
    final = evaluate_final_reward("DQN", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    eps_results[eps] = mean_final
    print(f"   DQN exploration_initial_eps={eps} -> mean final reward = {mean_final:.2f}")
best_eps = max(eps_results, key=eps_results.get)

best_dqn_params = dqn_base.copy()
best_dqn_params["buffer_size"] = best_buffer
best_dqn_params["exploration_initial_eps"] = best_eps
print(f"\nBest DQN hyperparams: {best_dqn_params}")

# ------------------------------------------------------------
# PPO tuning
# ------------------------------------------------------------
print("\n--- PPO hyperparameter tuning ---")

ppo_base = default_ppo.copy()

# 1. Tune learning rate
lr_results = {}
for lr in lr_vals:
    params = ppo_base.copy()
    params["learning_rate"] = lr
    final = evaluate_final_reward("PPO", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    lr_results[lr] = mean_final
    print(f"   PPO learning_rate={lr} -> mean final reward = {mean_final:.2f}")
best_lr = max(lr_results, key=lr_results.get)
print(f"Best PPO learning rate: {best_lr}")

# 2. Tune batch size
bs_results = {}
for bs in bs_vals:
    params = ppo_base.copy()
    params["learning_rate"] = best_lr
    params["batch_size"] = bs
    final = evaluate_final_reward("PPO", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    bs_results[bs] = mean_final
    print(f"   PPO batch_size={bs} -> mean final reward = {mean_final:.2f}")
best_bs = max(bs_results, key=bs_results.get)
print(f"Best PPO batch size: {best_bs}")

ppo_base["learning_rate"] = best_lr
ppo_base["batch_size"] = best_bs

# 3. Tune n_steps
nsteps_vals = [512, 1024, 2048, 4096, 8192]
nsteps_results = {}
for nsteps in nsteps_vals:
    params = ppo_base.copy()
    params["n_steps"] = nsteps
    final = evaluate_final_reward("PPO", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    nsteps_results[nsteps] = mean_final
    print(f"   PPO n_steps={nsteps} -> mean final reward = {mean_final:.2f}")
best_nsteps = max(nsteps_results, key=nsteps_results.get)
print(f"Best PPO n_steps: {best_nsteps}")

# 4. Tune clip_range
clip_vals = [0.05, 0.1, 0.2, 0.3, 0.4]
clip_results = {}
for cr in clip_vals:
    params = ppo_base.copy()
    params["n_steps"] = best_nsteps
    params["clip_range"] = cr
    final = evaluate_final_reward("PPO", params, SEEDS)
    mean_final = np.mean(list(final.values()))
    clip_results[cr] = mean_final
    print(f"   PPO clip_range={cr} -> mean final reward = {mean_final:.2f}")
best_clip = max(clip_results, key=clip_results.get)

best_ppo_params = ppo_base.copy()
best_ppo_params["n_steps"] = best_nsteps
best_ppo_params["clip_range"] = best_clip
print(f"\nBest PPO hyperparams: {best_ppo_params}")

# =============================================================================
# 3. Final training with tuned hyperparameters
# =============================================================================
print("\n=== 3. Final training with tuned hyperparameters ===")

dqn_tuned_curves = train_and_get_curves("DQN", best_dqn_params, SEEDS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES)
ppo_tuned_curves = train_and_get_curves("PPO", best_ppo_params, SEEDS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES)

plot_curves(dqn_tuned_curves, "DQN - Tuned Hyperparameters", "True Reward", os.path.join(FIGURE_DIR, "tuned_dqn.png"))
plot_curves(ppo_tuned_curves, "PPO - Tuned Hyperparameters", "True Reward", os.path.join(FIGURE_DIR, "tuned_ppo.png"))

# Combined tuned comparison
plt.figure(figsize=(10, 6))
for name, curves in [("DQN", dqn_tuned_curves), ("PPO", ppo_tuned_curves)]:
    data = np.array([curves[s] for s in SEEDS])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.arange(len(mean)) * (EVAL_FREQ // 1000)
    plt.plot(x, mean, label=name)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
plt.xlabel('Training steps (thousands)')
plt.ylabel('True Reward')
plt.title('Tuned Hyperparameters Comparison')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIGURE_DIR, "tuned_comparison.png"))
plt.close()

# =============================================================================
# 4. Final stats
# =============================================================================
print("\n=== Final Results ===")
print(f"DQN default: mean final reward = {np.mean([dqn_curves[s][-1] for s in SEEDS]):.2f} ± {np.std([dqn_curves[s][-1] for s in SEEDS]):.2f}")
print(f"DQN tuned:  mean final reward = {np.mean([dqn_tuned_curves[s][-1] for s in SEEDS]):.2f} ± {np.std([dqn_tuned_curves[s][-1] for s in SEEDS]):.2f}")
print(f"PPO default: mean final reward = {np.mean([ppo_curves[s][-1] for s in SEEDS]):.2f} ± {np.std([ppo_curves[s][-1] for s in SEEDS]):.2f}")
print(f"PPO tuned:  mean final reward = {np.mean([ppo_tuned_curves[s][-1] for s in SEEDS]):.2f} ± {np.std([ppo_tuned_curves[s][-1] for s in SEEDS]):.2f}")

print("\nAll done. Figures saved to", FIGURE_DIR)