"""
Assignment 2 - Part 3

- Train MaskablePPO on BoundedKnapsackEnv with invalid action masking
- Reuse tuned PPO hyperparameters from Part 2 if available
- Tune additional MaskablePPO hyperparameters
- Compare masked PPO against tuned PPO
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
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from knapsack_env import BoundedKnapsackEnv


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
PART2_RESULTS_PATH = RESULTS_DIR / "assignment2_part2_summary.json"

for directory in (LOG_DIR, FIGURE_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)


EXPERIMENT_CONFIG = {
    "env": {
        "n_items": 200,
        "max_weight": 200,
        "randomize_params_on_reset": False,
    },
    "total_timesteps": 50_000,
    "eval_freq": 2_000,
    "n_eval_episodes": 5,
    "seeds": [21, 31, 41],
    "wandb_project": "assignment-2-bounded-knapsack",
    "wandb_entity": None,
    "wandb_mode": "online",
}

DEFAULT_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 32,
    "n_epochs": 5,
    "clip_range": 0.2,
}

MASKED_PPO_GRID = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps": [128, 256, 512],
    "ent_coef": [0.0, 0.005, 0.01],
}


def init_wandb_run(name: str, group: str, config: Dict[str, Any], tags: List[str]) -> Any:
    try:
        run = wandb.init(
            project=EXPERIMENT_CONFIG["wandb_project"],
            entity=EXPERIMENT_CONFIG["wandb_entity"],
            mode=EXPERIMENT_CONFIG["wandb_mode"],
            name=name,
            group=group,
            tags=tags,
            config=config,
            reinit="finish_previous",
            save_code=True,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        return run
    except Exception as exc:
        print(f"W&B init failed ({exc}). Falling back to disabled mode.")
        run = wandb.init(
            project=EXPERIMENT_CONFIG["wandb_project"],
            entity=EXPERIMENT_CONFIG["wandb_entity"],
            mode="disabled",
            name=name,
            group=group,
            tags=tags,
            config=config,
            reinit="finish_previous",
            save_code=False,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        return run


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_mask()


def make_env(seed: int, monitor_tag: str = "run") -> gym.Env:
    env = BoundedKnapsackEnv(**EXPERIMENT_CONFIG["env"], mask=True)
    env.reset(seed=seed)
    env = ActionMasker(env, mask_fn)
    return Monitor(env, filename=str(LOG_DIR / f"{monitor_tag}_seed_{seed}"))


def load_part2_best_ppo_params() -> Tuple[Dict[str, Any], Dict[str, float]]:
    if PART2_RESULTS_PATH.exists():
        payload = json.loads(PART2_RESULTS_PATH.read_text(encoding="utf-8"))
        summary = payload.get("final_summary", {})
        best_ppo_params = summary.get("best_ppo_params", DEFAULT_PPO)
        tuned_ppo = summary.get("tuned_ppo", {})
        print(f"Loaded tuned PPO hyperparameters from {PART2_RESULTS_PATH}")
        return best_ppo_params, tuned_ppo

    print("Part 2 summary not found. Falling back to default PPO hyperparameters.")
    return dict(DEFAULT_PPO), {}


def predict_action(model: Any, obs: np.ndarray, env: gym.Env) -> int:
    action_masks = env.unwrapped.get_mask()
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    return int(action)


def evaluate_agent(model: Any, seed: int, n_episodes: int) -> Tuple[float, float, List[float]]:
    eval_env = make_env(seed=seed, monitor_tag="part3_eval")
    rewards: List[float] = []

    for episode in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action = predict_action(model, obs, eval_env)
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
                "global_step": self.num_timesteps,
                f"{self.metric_prefix}/eval_mean_true_reward": mean_reward,
                f"{self.metric_prefix}/eval_std_true_reward": std_reward,
            }
        )
        return True


def train_single_seed(hyperparams: Dict[str, Any], seed: int, stage: str) -> Dict[str, Any]:
    env = make_env(seed=seed, monitor_tag=f"{stage}_maskableppo")
    callback = WandbEvalCallback(
        eval_seed=seed,
        eval_freq=EXPERIMENT_CONFIG["eval_freq"],
        n_eval_episodes=EXPERIMENT_CONFIG["n_eval_episodes"],
        metric_prefix=f"{stage}/MaskablePPO/seed_{seed}",
    )
    model = MaskablePPO("MlpPolicy", env, verbose=0, seed=seed, **hyperparams)
    model.learn(total_timesteps=EXPERIMENT_CONFIG["total_timesteps"], callback=callback)

    final_mean, final_std, episode_rewards = evaluate_agent(
        model,
        seed=seed,
        n_episodes=EXPERIMENT_CONFIG["n_eval_episodes"],
    )
    env.close()

    wandb.log(
        {
            f"{stage}/MaskablePPO/seed_{seed}/final_mean_true_reward": final_mean,
            f"{stage}/MaskablePPO/seed_{seed}/final_std_true_reward": final_std,
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


def run_multi_seed_experiment(hyperparams: Dict[str, Any], stage: str) -> Dict[str, Any]:
    seed_runs = []
    for seed in EXPERIMENT_CONFIG["seeds"]:
        seed_runs.append(train_single_seed(hyperparams=hyperparams, seed=seed, stage=stage))

    aggregate_curve = aggregate_seed_curves(seed_runs)
    final_rewards = [run["final_mean_reward"] for run in seed_runs]
    summary = {
        "hyperparams": hyperparams,
        "seed_runs": seed_runs,
        "aggregate_curve": aggregate_curve,
        "final_mean_reward": float(np.mean(final_rewards)),
        "final_std_reward": float(np.std(final_rewards)),
    }

    wandb.log(
        {
            f"{stage}/MaskablePPO/aggregate_final_mean_true_reward": summary["final_mean_reward"],
            f"{stage}/MaskablePPO/aggregate_final_std_true_reward": summary["final_std_reward"],
        }
    )
    return summary


def tune_hyperparameters(base_hyperparams: Dict[str, Any], search_space: Dict[str, List[Any]], stage: str) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    best_params = dict(base_hyperparams)
    tuning_history: Dict[str, List[Dict[str, Any]]] = {}

    for hyperparam_name, candidates in search_space.items():
        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            candidate_params = dict(best_params)
            candidate_params[hyperparam_name] = candidate
            run_summary = run_multi_seed_experiment(candidate_params, stage=f"{stage}_{hyperparam_name}_{candidate}")
            result = {
                "hyperparameter": hyperparam_name,
                "candidate": candidate,
                "mean_reward": run_summary["final_mean_reward"],
                "std_reward": run_summary["final_std_reward"],
            }
            results.append(result)
            print(
                f"MaskablePPO {hyperparam_name}={candidate} -> "
                f"mean final reward = {run_summary['final_mean_reward']:.2f}"
            )

        best_result = max(results, key=lambda item: item["mean_reward"])
        best_params[hyperparam_name] = best_result["candidate"]
        tuning_history[hyperparam_name] = results

        table = wandb.Table(columns=["hyperparameter", "candidate", "mean_reward", "std_reward"])
        for row in results:
            table.add_data(row["hyperparameter"], str(row["candidate"]), row["mean_reward"], row["std_reward"])
        wandb.log({f"{stage}/MaskablePPO/{hyperparam_name}_search": table})

    return best_params, tuning_history


def save_json(filename: str, payload: Dict[str, Any]) -> None:
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    best_ppo_params, tuned_ppo_summary = load_part2_best_ppo_params()

    root_run = init_wandb_run(
        name="group02-assignment2-part3",
        group="assignment2-part3",
        config={**EXPERIMENT_CONFIG, "part2_best_ppo_params": best_ppo_params},
        tags=["assignment-2", "part-3", "bounded-knapsack", "maskableppo"],
    )

    print("=== Part 3: MaskablePPO with PPO hyperparameters ===")
    masked_ppo_base = run_multi_seed_experiment(best_ppo_params, stage="part3_masked_base")

    print("\n=== Part 3: Additional MaskablePPO tuning ===")
    best_masked_params, masked_tuning = tune_hyperparameters(best_ppo_params, MASKED_PPO_GRID, "part3_tuning_maskedppo")
    masked_ppo_tuned = run_multi_seed_experiment(best_masked_params, stage="part3_masked_tuned")

    plot_aggregate_curve(masked_ppo_base["aggregate_curve"], "MaskablePPO with PPO tuned hyperparameters", "part3_masked_base.png", "figures/part3_masked_base")
    plot_aggregate_curve(masked_ppo_tuned["aggregate_curve"], "MaskablePPO with tuned masked hyperparameters", "part3_masked_tuned.png", "figures/part3_masked_tuned")

    comparison_curves = {
        "MaskablePPO base": masked_ppo_base["aggregate_curve"],
        "MaskablePPO tuned": masked_ppo_tuned["aggregate_curve"],
    }
    if tuned_ppo_summary:
        tuned_ppo_reward = tuned_ppo_summary.get("final_mean_reward")
        tuned_ppo_std = tuned_ppo_summary.get("final_std_reward", 0.0)
        if tuned_ppo_reward is not None:
            comparison_curves["PPO tuned (final)"] = [
                {"timesteps": 0, "mean_true_reward": tuned_ppo_reward, "std_true_reward": tuned_ppo_std},
                {"timesteps": EXPERIMENT_CONFIG["total_timesteps"], "mean_true_reward": tuned_ppo_reward, "std_true_reward": tuned_ppo_std},
            ]

    plot_comparison_curves(comparison_curves, "Part 3 comparison against PPO", "part3_comparison.png", "figures/part3_comparison")

    log_curve_table("tables/part3_masked_base_curve", masked_ppo_base["aggregate_curve"])
    log_curve_table("tables/part3_masked_tuned_curve", masked_ppo_tuned["aggregate_curve"])

    final_summary = {
        "part2_best_ppo_params": best_ppo_params,
        "part2_tuned_ppo_summary": tuned_ppo_summary,
        "masked_base": {
            "final_mean_reward": masked_ppo_base["final_mean_reward"],
            "final_std_reward": masked_ppo_base["final_std_reward"],
        },
        "best_masked_params": best_masked_params,
        "masked_tuned": {
            "final_mean_reward": masked_ppo_tuned["final_mean_reward"],
            "final_std_reward": masked_ppo_tuned["final_std_reward"],
        },
    }

    save_json(
        "assignment2_part3_summary.json",
        {
            "experiment_config": EXPERIMENT_CONFIG,
            "final_summary": final_summary,
            "masked_tuning": masked_tuning,
        },
    )

    wandb.summary["part3/part2_best_ppo_params"] = best_ppo_params
    wandb.summary["part3/best_masked_params"] = best_masked_params
    wandb.summary["part3/masked_base_final_mean_reward"] = masked_ppo_base["final_mean_reward"]
    wandb.summary["part3/masked_tuned_final_mean_reward"] = masked_ppo_tuned["final_mean_reward"]

    print("\n=== Final results ===")
    if tuned_ppo_summary:
        print(
            f"PPO tuned from part 2: {tuned_ppo_summary.get('final_mean_reward', float('nan')):.2f} ± "
            f"{tuned_ppo_summary.get('final_std_reward', float('nan')):.2f}"
        )
    print(f"MaskablePPO base:  {masked_ppo_base['final_mean_reward']:.2f} ± {masked_ppo_base['final_std_reward']:.2f}")
    print(f"MaskablePPO tuned: {masked_ppo_tuned['final_mean_reward']:.2f} ± {masked_ppo_tuned['final_std_reward']:.2f}")
    print(f"\nFigures saved to: {FIGURE_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")

    root_run.finish()


if __name__ == "__main__":
    main()
