"""
Assignment 2 - Part 2

- Train DQN and PPO on BoundedKnapsackEnv
- Aggregate results across 3 seeds
- Tune 4 hyperparameters manually with 5 candidate values each
- Log metrics, tables, and figures to W&B
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import wandb
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from knapsack_env import BoundedKnapsackEnv


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

for directory in (LOG_DIR, FIGURE_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)


EXPERIMENT_CONFIG = {
    "env": {
        "n_items": 200,
        "max_weight": 200,
        "randomize_params_on_reset": False,
    },
    "total_timesteps": 50_000,
    "eval_freq": 1_000,
    "n_eval_episodes": 10,
    "seeds": [42, 123, 456],
    "wandb_project": os.getenv("WANDB_PROJECT", "assignment-2-bounded-knapsack"),
    "wandb_entity": os.getenv("WANDB_ENTITY"),
    "wandb_mode": os.getenv("WANDB_MODE", "online"),
}

DEFAULT_DQN = {
    "learning_rate": 1e-4,
    "buffer_size": 50_000,
    "batch_size": 32,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
}

DEFAULT_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "clip_range": 0.2,
}

PART2_SEARCH_SPACE = {
    "DQN": {
        "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64, 128, 256],
        "buffer_size": [10_000, 50_000, 100_000, 500_000, 1_000_000],
        "exploration_initial_eps": [1.0, 0.9, 0.5, 0.3, 0.1],
    },
    "PPO": {
        "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64, 128, 256],
        "n_steps": [512, 1024, 2048, 4096, 8192],
        "clip_range": [0.05, 0.1, 0.2, 0.3, 0.4],
    },
}


def make_env(seed: int, monitor_tag: str = "run") -> gym.Env:
    env = BoundedKnapsackEnv(**EXPERIMENT_CONFIG["env"], mask=False)
    env.reset(seed=seed)
    return Monitor(env, filename=str(LOG_DIR / f"{monitor_tag}_seed_{seed}"))


def init_wandb_run(name: str, group: str, config: Dict[str, Any], tags: List[str]) -> Any:
    try:
        return wandb.init(
            project=EXPERIMENT_CONFIG["wandb_project"],
            entity=EXPERIMENT_CONFIG["wandb_entity"],
            mode=EXPERIMENT_CONFIG["wandb_mode"],
            name=name,
            group=group,
            tags=tags,
            config=config,
            reinit=True,
            save_code=True,
        )
    except Exception as exc:
        print(f"W&B init failed ({exc}). Falling back to disabled mode.")
        return wandb.init(
            project=EXPERIMENT_CONFIG["wandb_project"],
            entity=EXPERIMENT_CONFIG["wandb_entity"],
            mode="disabled",
            name=name,
            group=group,
            tags=tags,
            config=config,
            reinit=True,
            save_code=False,
        )


def evaluate_agent(model: Any, seed: int, n_episodes: int) -> Tuple[float, float, List[float]]:
    eval_env = make_env(seed=seed, monitor_tag="part2_eval")
    rewards: List[float] = []

    for episode in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward * 100.0)

    eval_env.close()
    return float(np.mean(rewards)), float(np.std(rewards)), rewards


class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_seed: int, eval_freq: int, n_eval_episodes: int, metric_prefix: str) -> None:
        super().__init__(verbose=0)
        self.eval_seed = eval_seed
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.metric_prefix = metric_prefix
        self.history: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        mean_reward, std_reward, _ = evaluate_agent(
            self.model,
            seed=self.eval_seed,
            n_episodes=self.n_eval_episodes,
        )
        self.history.append(
            {
                "timesteps": float(self.num_timesteps),
                "mean_true_reward": mean_reward,
                "std_true_reward": std_reward,
            }
        )
        wandb.log(
            {
                f"{self.metric_prefix}/eval_mean_true_reward": mean_reward,
                f"{self.metric_prefix}/eval_std_true_reward": std_reward,
                "global_step": self.num_timesteps,
            },
            step=self.num_timesteps,
        )
        return True


def build_model(algorithm: str, env: gym.Env, seed: int, hyperparams: Dict[str, Any]) -> Any:
    if algorithm == "DQN":
        return DQN("MlpPolicy", env, verbose=0, seed=seed, **hyperparams)
    if algorithm == "PPO":
        return PPO("MlpPolicy", env, verbose=0, seed=seed, **hyperparams)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def train_single_seed(algorithm: str, hyperparams: Dict[str, Any], seed: int, stage: str) -> Dict[str, Any]:
    env = make_env(seed=seed, monitor_tag=f"{stage}_{algorithm.lower()}")
    callback = WandbEvalCallback(
        eval_seed=seed,
        eval_freq=EXPERIMENT_CONFIG["eval_freq"],
        n_eval_episodes=EXPERIMENT_CONFIG["n_eval_episodes"],
        metric_prefix=f"{stage}/{algorithm}/seed_{seed}",
    )
    model = build_model(algorithm, env, seed=seed, hyperparams=hyperparams)
    model.learn(total_timesteps=EXPERIMENT_CONFIG["total_timesteps"], callback=callback)

    final_mean, final_std, episode_rewards = evaluate_agent(
        model,
        seed=seed,
        n_episodes=EXPERIMENT_CONFIG["n_eval_episodes"],
    )
    env.close()

    wandb.log(
        {
            f"{stage}/{algorithm}/seed_{seed}/final_mean_true_reward": final_mean,
            f"{stage}/{algorithm}/seed_{seed}/final_std_true_reward": final_std,
        }
    )

    return {
        "seed": seed,
        "curve": callback.history,
        "final_mean_reward": final_mean,
        "final_std_reward": final_std,
        "final_episode_rewards": episode_rewards,
    }


def aggregate_seed_curves(seed_runs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    min_len = min(len(run["curve"]) for run in seed_runs)
    aggregate: List[Dict[str, float]] = []

    for idx in range(min_len):
        step = int(seed_runs[0]["curve"][idx]["timesteps"])
        rewards = [run["curve"][idx]["mean_true_reward"] for run in seed_runs]
        aggregate.append(
            {
                "timesteps": step,
                "mean_true_reward": float(np.mean(rewards)),
                "std_true_reward": float(np.std(rewards)),
            }
        )
    return aggregate


def plot_aggregate_curve(aggregate_curve: List[Dict[str, float]], title: str, filename: str, wandb_key: str) -> None:
    x = np.array([point["timesteps"] for point in aggregate_curve])
    mean = np.array([point["mean_true_reward"] for point in aggregate_curve])
    std = np.array([point["std_true_reward"] for point in aggregate_curve])

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, label="Mean true reward")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="Std across seeds")
    plt.xlabel("Training steps")
    plt.ylabel("True reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    wandb.log({wandb_key: wandb.Image(str(output_path))})


def plot_comparison_curves(named_curves: Dict[str, List[Dict[str, float]]], title: str, filename: str, wandb_key: str) -> None:
    plt.figure(figsize=(10, 6))

    for name, curve in named_curves.items():
        x = np.array([point["timesteps"] for point in curve])
        mean = np.array([point["mean_true_reward"] for point in curve])
        std = np.array([point["std_true_reward"] for point in curve])
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    plt.xlabel("Training steps")
    plt.ylabel("True reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    wandb.log({wandb_key: wandb.Image(str(output_path))})


def log_curve_table(table_name: str, aggregate_curve: List[Dict[str, float]]) -> None:
    table = wandb.Table(columns=["timesteps", "mean_true_reward", "std_true_reward"])
    for row in aggregate_curve:
        table.add_data(row["timesteps"], row["mean_true_reward"], row["std_true_reward"])
    wandb.log({table_name: table})


def run_multi_seed_experiment(algorithm: str, hyperparams: Dict[str, Any], stage: str) -> Dict[str, Any]:
    seed_runs = []
    for seed in EXPERIMENT_CONFIG["seeds"]:
        seed_runs.append(train_single_seed(algorithm=algorithm, hyperparams=hyperparams, seed=seed, stage=stage))

    aggregate_curve = aggregate_seed_curves(seed_runs)
    final_rewards = [run["final_mean_reward"] for run in seed_runs]
    summary = {
        "algorithm": algorithm,
        "hyperparams": hyperparams,
        "seed_runs": seed_runs,
        "aggregate_curve": aggregate_curve,
        "final_mean_reward": float(np.mean(final_rewards)),
        "final_std_reward": float(np.std(final_rewards)),
    }

    wandb.log(
        {
            f"{stage}/{algorithm}/aggregate_final_mean_true_reward": summary["final_mean_reward"],
            f"{stage}/{algorithm}/aggregate_final_std_true_reward": summary["final_std_reward"],
        }
    )
    return summary


def tune_hyperparameters(
    algorithm: str,
    base_hyperparams: Dict[str, Any],
    search_space: Dict[str, List[Any]],
    stage: str,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    best_params = dict(base_hyperparams)
    tuning_history: Dict[str, List[Dict[str, Any]]] = {}

    for hyperparam_name, candidates in search_space.items():
        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            candidate_params = dict(best_params)
            candidate_params[hyperparam_name] = candidate
            run_summary = run_multi_seed_experiment(
                algorithm=algorithm,
                hyperparams=candidate_params,
                stage=f"{stage}_{hyperparam_name}_{candidate}",
            )
            result = {
                "hyperparameter": hyperparam_name,
                "candidate": candidate,
                "mean_reward": run_summary["final_mean_reward"],
                "std_reward": run_summary["final_std_reward"],
            }
            results.append(result)
            print(
                f"{algorithm} {hyperparam_name}={candidate} -> "
                f"mean final reward = {run_summary['final_mean_reward']:.2f}"
            )

        best_result = max(results, key=lambda item: item["mean_reward"])
        best_params[hyperparam_name] = best_result["candidate"]
        tuning_history[hyperparam_name] = results

        table = wandb.Table(columns=["hyperparameter", "candidate", "mean_reward", "std_reward"])
        for row in results:
            table.add_data(row["hyperparameter"], str(row["candidate"]), row["mean_reward"], row["std_reward"])
        wandb.log({f"{stage}/{algorithm}/{hyperparam_name}_search": table})

    return best_params, tuning_history


def save_json(filename: str, payload: Dict[str, Any]) -> None:
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    root_run = init_wandb_run(
        name="group02-assignment2-part2",
        group="assignment2-part2",
        config=EXPERIMENT_CONFIG,
        tags=["assignment-2", "part-2", "bounded-knapsack"],
    )

    print("=== Part 2: Default training ===")
    dqn_default = run_multi_seed_experiment("DQN", DEFAULT_DQN, stage="part2_default_dqn")
    ppo_default = run_multi_seed_experiment("PPO", DEFAULT_PPO, stage="part2_default_ppo")

    plot_aggregate_curve(dqn_default["aggregate_curve"], "DQN with default hyperparameters", "part2_default_dqn.png", "figures/part2_default_dqn")
    plot_aggregate_curve(ppo_default["aggregate_curve"], "PPO with default hyperparameters", "part2_default_ppo.png", "figures/part2_default_ppo")
    plot_comparison_curves(
        {"DQN default": dqn_default["aggregate_curve"], "PPO default": ppo_default["aggregate_curve"]},
        "Part 2 default comparison",
        "part2_default_comparison.png",
        "figures/part2_default_comparison",
    )

    log_curve_table("tables/part2_default_dqn_curve", dqn_default["aggregate_curve"])
    log_curve_table("tables/part2_default_ppo_curve", ppo_default["aggregate_curve"])

    print("\n=== Part 2: Hyperparameter tuning ===")
    best_dqn_params, dqn_tuning = tune_hyperparameters("DQN", DEFAULT_DQN, PART2_SEARCH_SPACE["DQN"], "part2_tuning_dqn")
    best_ppo_params, ppo_tuning = tune_hyperparameters("PPO", DEFAULT_PPO, PART2_SEARCH_SPACE["PPO"], "part2_tuning_ppo")

    print("\n=== Part 2: Final tuned training ===")
    dqn_tuned = run_multi_seed_experiment("DQN", best_dqn_params, stage="part2_tuned_dqn")
    ppo_tuned = run_multi_seed_experiment("PPO", best_ppo_params, stage="part2_tuned_ppo")

    plot_aggregate_curve(dqn_tuned["aggregate_curve"], "DQN with tuned hyperparameters", "part2_tuned_dqn.png", "figures/part2_tuned_dqn")
    plot_aggregate_curve(ppo_tuned["aggregate_curve"], "PPO with tuned hyperparameters", "part2_tuned_ppo.png", "figures/part2_tuned_ppo")
    plot_comparison_curves(
        {"DQN tuned": dqn_tuned["aggregate_curve"], "PPO tuned": ppo_tuned["aggregate_curve"]},
        "Part 2 tuned comparison",
        "part2_tuned_comparison.png",
        "figures/part2_tuned_comparison",
    )

    final_summary = {
        "default_dqn": {
            "final_mean_reward": dqn_default["final_mean_reward"],
            "final_std_reward": dqn_default["final_std_reward"],
        },
        "default_ppo": {
            "final_mean_reward": ppo_default["final_mean_reward"],
            "final_std_reward": ppo_default["final_std_reward"],
        },
        "best_dqn_params": best_dqn_params,
        "best_ppo_params": best_ppo_params,
        "tuned_dqn": {
            "final_mean_reward": dqn_tuned["final_mean_reward"],
            "final_std_reward": dqn_tuned["final_std_reward"],
        },
        "tuned_ppo": {
            "final_mean_reward": ppo_tuned["final_mean_reward"],
            "final_std_reward": ppo_tuned["final_std_reward"],
        },
    }

    save_json(
        "assignment2_part2_summary.json",
        {
            "experiment_config": EXPERIMENT_CONFIG,
            "final_summary": final_summary,
            "dqn_tuning": dqn_tuning,
            "ppo_tuning": ppo_tuning,
        },
    )

    wandb.summary["part2/best_dqn_params"] = best_dqn_params
    wandb.summary["part2/best_ppo_params"] = best_ppo_params
    wandb.summary["part2/dqn_tuned_final_mean_reward"] = dqn_tuned["final_mean_reward"]
    wandb.summary["part2/ppo_tuned_final_mean_reward"] = ppo_tuned["final_mean_reward"]

    print("\n=== Final results ===")
    print(f"DQN default: {dqn_default['final_mean_reward']:.2f} ± {dqn_default['final_std_reward']:.2f}")
    print(f"DQN tuned:   {dqn_tuned['final_mean_reward']:.2f} ± {dqn_tuned['final_std_reward']:.2f}")
    print(f"PPO default: {ppo_default['final_mean_reward']:.2f} ± {ppo_default['final_std_reward']:.2f}")
    print(f"PPO tuned:   {ppo_tuned['final_mean_reward']:.2f} ± {ppo_tuned['final_std_reward']:.2f}")
    print(f"\nFigures saved to: {FIGURE_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")

    root_run.finish()


if __name__ == "__main__":
    main()
