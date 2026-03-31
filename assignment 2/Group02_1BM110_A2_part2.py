"""
Assignment 2 - Part 2

- Train DQN and PPO on BoundedKnapsackEnv
- Aggregate results across 3 seeds
- Tune 4 hyperparameters manually with 3 candidate values each
- Log metrics, tables, and figures to W&B
"""

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

# make these folders once at the start so saving logs and plots does not break later
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

DEFAULT_DQN = {
    "learning_rate": 1e-4,
    "buffer_size": 10_000,
    "batch_size": 16,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
}

DEFAULT_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 32,
    "n_epochs": 5,
    "clip_range": 0.2,
}

# reduce the search space to keep the total runtime manageable, but still have some variety to see what works best
PART2_SEARCH_SPACE = {
    "DQN": {
        "learning_rate": [1e-5, 3e-5, 1e-4],
        "gamma": [0.90, 0.95, 0.97],
        "buffer_size": [2_000, 5_000, 10_000],
        "exploration_initial_eps": [1.0, 0.9, 0.5],
    },
    "PPO": {
        "learning_rate": [1e-5, 3e-5, 1e-4],
        "gamma": [0.90, 0.95, 0.97],
        "n_steps": [128, 256, 512],
        "clip_range": [0.05, 0.1, 0.2],
    },
}


def make_env(seed: int, monitor_tag: str = "run") -> gym.Env:
    # part 2 is the no-mask baseline, so masking stays off here on purpose
    env = BoundedKnapsackEnv(**EXPERIMENT_CONFIG["env"], mask=False)
    env.reset(seed=seed)
    return Monitor(env, filename=str(LOG_DIR / f"{monitor_tag}_seed_{seed}"))

# using wandb helps with montioring the results online as they come in, and it also makes it easy to log tables and figures in a way that keeps them organized and shareable
def init_wandb_run(name: str, group: str, config: Dict[str, Any], tags: List[str]) -> Any:
    try:
        # use wandb when possible because the assignment asks for tracked results and figures
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
        # if wandb fails, still run the experiment and keep local outputs
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


def evaluate_agent(model: Any, seed: int, n_episodes: int) -> Tuple[float, float, List[float]]:
    eval_env = make_env(seed=seed, monitor_tag="part2_eval")
    rewards: List[float] = []

    for episode in range(n_episodes):
        # change the seed per episode a bit so evaluation is not just one repeated reset
        obs, _ = eval_env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        # the env scales reward down, so bring it back to the more readable true reward
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

        # evaluate during training so later we can plot the learning curve, not only the final score
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


def build_model(algorithm: str, env: gym.Env, seed: int, hyperparams: Dict[str, Any]) -> Any:
    # keep model creation in one place so switching between DQN and PPO stays simple
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
    # train each seed separately so the final result is not based on one lucky run
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
    # use the shortest history so averaging lines up correctly across all seeds
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

    # plot both the average and the spread so seed differences are visible too
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
        # draw each method with its own shaded range so the comparison is easier to read
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
    # tables are useful because they keep the exact curve values behind the plots
    table = wandb.Table(columns=["timesteps", "mean_true_reward", "std_true_reward"])
    for row in aggregate_curve:
        table.add_data(row["timesteps"], row["mean_true_reward"], row["std_true_reward"])
    wandb.log({table_name: table})


def run_multi_seed_experiment(algorithm: str, hyperparams: Dict[str, Any], stage: str) -> Dict[str, Any]:
    seed_runs = []
    for seed in EXPERIMENT_CONFIG["seeds"]:
        seed_runs.append(train_single_seed(algorithm=algorithm, hyperparams=hyperparams, seed=seed, stage=stage))

    # average across the assignment seeds so the score is more stable and fair
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
            # change one hyperparameter at a time so it is easier to see what helped
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

        # keep the best value before moving on to tune the next setting
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
    # save a local summary too so the main outputs are easy to reuse in part 3
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
    # start with the default settings so there is a baseline before any tuning
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
    # after the baseline, try a few candidate values to see which setup works better
    best_dqn_params, dqn_tuning = tune_hyperparameters("DQN", DEFAULT_DQN, PART2_SEARCH_SPACE["DQN"], "part2_tuning_dqn")
    best_ppo_params, ppo_tuning = tune_hyperparameters("PPO", DEFAULT_PPO, PART2_SEARCH_SPACE["PPO"], "part2_tuning_ppo")

    print("\n=== Part 2: Final tuned training ===")
    # train once more with the best settings so the final comparison uses the selected configs
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
