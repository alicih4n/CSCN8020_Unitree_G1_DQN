from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from g1_rl import G1ElbowTargetEnv


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_configuration(
    config_name: str,
    epsilon_decay: float,
    num_episodes: int = 600,
    seed: int = 42,
) -> tuple[dict, float]:
    """
    Train a single DQN configuration on the G1 Elbow environment.

    Returns:
        metrics: Dictionary of list of metrics recorded per episode.
        training_time: Total wall-clock time in seconds.
    """
    print(f"\n==================================================")
    print(f"TRAINING CONFIGURATION {config_name.upper()}")
    print(f"Epsilon Decay: {epsilon_decay:.3f} | Total Episodes: {num_episodes}")
    print(f"==================================================")

    # Set seeds
    set_seed(seed)

    # Instantiate headless environment
    env = G1ElbowTargetEnv(render_mode=None)
    
    # Initialize DQN Agent using standard hyperparameters
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=0.001,
        gamma=0.95,
        target_update_interval=250,
    )
    
    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(capacity=50000)

    # Hyperparameters
    batch_size = 64
    warmup_steps = 500
    epsilon_start = 1.00
    epsilon_min = 0.05
    epsilon = epsilon_start

    # Metrics collections
    metrics = {
        "episode": [],
        "reward": [],
        "success": [],
        "length": [],
        "final_error": [],
        "epsilon": [],
        "avg_loss": [],
        "wall_clock_time": [],
    }

    # Tracking variables
    total_steps = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_losses = []
        episode_steps = 0

        while True:
            # Action selection (epsilon-greedy)
            action = agent.select_action(state, epsilon)
            
            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            # Push transition into the replay buffer
            # Note: We store terminated which correctly masks out bootstrapping on success,
            # but we bootstrap on truncation (time-limit) because we do not set terminated to True
            # unless the success condition is met.
            replay_buffer.push(state, action, reward, next_state, terminated)

            # Move to next state
            state = next_state

            # Perform optimization step if out of warm-up phase
            if total_steps >= warmup_steps:
                loss = agent.optimize_model(replay_buffer, batch_size)
                if loss is not None:
                    episode_losses.append(loss)

            if terminated or truncated:
                break

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Record metrics
        episode_avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        current_wall_time = time.time() - start_time
        success_indicator = 1.0 if info["success_streak"] >= env.required_success_steps else 0.0

        metrics["episode"].append(episode)
        metrics["reward"].append(episode_reward)
        metrics["success"].append(success_indicator)
        metrics["length"].append(episode_steps)
        metrics["final_error"].append(info["absolute_error"])
        metrics["epsilon"].append(epsilon)
        metrics["avg_loss"].append(episode_avg_loss)
        metrics["wall_clock_time"].append(current_wall_time)

        # Periodic print statements
        if episode % 25 == 0 or episode == 1:
            recent_rewards = metrics["reward"][-25:]
            recent_success = metrics["success"][-25:]
            print(
                f"Episode {episode:3d}/{num_episodes} | "
                f"Reward: {episode_reward:7.2f} (Avg25: {np.mean(recent_rewards):7.2f}) | "
                f"Success: {success_indicator} (Avg25 Success: {np.mean(recent_success):.2f}) | "
                f"Steps: {episode_steps:3d} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Loss: {episode_avg_loss:.6f}"
            )

    total_training_time = time.time() - start_time
    print(f"\nConfiguration {config_name.upper()} training complete in {total_training_time:.2f} seconds.")
    
    # Save the checkpoint
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    agent.save(checkpoint_dir / f"dqn_config_{config_name.lower()}.pt")

    # Save metrics to CSV
    results_dir = Path("results") / f"config_{config_name.lower()}"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "training_metrics.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))
    print(f"Metrics saved to: {csv_path}")

    # Generate configuration plots
    generate_plots(config_name, metrics, results_dir)

    env.close()
    return metrics, total_training_time


def generate_plots(config_name: str, metrics: dict, output_dir: Path) -> None:
    """Generate and save the required training visualization plots."""
    episodes = metrics["episode"]
    rewards = metrics["reward"]
    successes = metrics["success"]
    epsilons = metrics["epsilon"]
    losses = metrics["avg_loss"]

    # 1. Rewards (Raw & Moving Average)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Raw Reward", color="skyblue")
    
    # Compute moving average
    window = 20
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(episodes[window-1:], moving_avg, label=f"{window}-Episode Moving Avg", color="navy", linewidth=2)
    
    plt.title(f"Configuration {config_name.upper()} - Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_reward.png", dpi=150)
    plt.close()

    # 2. Rolling Success Rate
    plt.figure(figsize=(10, 5))
    success_window = 50
    rolling_success = np.convolve(successes, np.ones(success_window)/success_window, mode="valid")
    plt.plot(episodes[success_window-1:], rolling_success, color="forestgreen", linewidth=2)
    plt.title(f"Configuration {config_name.upper()} - Rolling Success Rate (Window={success_window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_success_rate.png", dpi=150)
    plt.close()

    # 3. Epsilon Decay
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, epsilons, color="darkorange", linewidth=2)
    plt.title(f"Configuration {config_name.upper()} - Epsilon Exploration Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_decay.png", dpi=150)
    plt.close()

    # 4. Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, losses, color="crimson", alpha=0.6)
    plt.title(f"Configuration {config_name.upper()} - Mean Temporal-Difference Loss")
    plt.xlabel("Episode")
    plt.ylabel("Huber Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()


def generate_comparison_plots(
    metrics_a: dict,
    metrics_b: dict,
    output_dir: Path,
) -> None:
    """Generate combined comparison plots overlaying Configuration A and B."""
    output_dir.mkdir(exist_ok=True)
    episodes = metrics_a["episode"]

    # 1. Compare Rewards
    plt.figure(figsize=(10, 6))
    window = 20
    moving_avg_a = np.convolve(metrics_a["reward"], np.ones(window)/window, mode="valid")
    moving_avg_b = np.convolve(metrics_b["reward"], np.ones(window)/window, mode="valid")
    
    plt.plot(episodes[window-1:], moving_avg_a, label="Config A (Decay=0.995, Longer Exploration)", color="navy", linewidth=2)
    plt.plot(episodes[window-1:], moving_avg_b, label="Config B (Decay=0.985, Faster Exploitation)", color="crimson", linewidth=2)
    
    plt.title("Epsilon Decay Comparison - Mean Training Reward (Moving Average)")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_decay_reward_comparison.png", dpi=150)
    plt.close()

    # 2. Compare Rolling Success Rates
    plt.figure(figsize=(10, 6))
    success_window = 50
    rolling_success_a = np.convolve(metrics_a["success"], np.ones(success_window)/success_window, mode="valid")
    rolling_success_b = np.convolve(metrics_b["success"], np.ones(success_window)/success_window, mode="valid")
    
    plt.plot(episodes[success_window-1:], rolling_success_a, label="Config A (Decay=0.995)", color="navy", linewidth=2)
    plt.plot(episodes[success_window-1:], rolling_success_b, label="Config B (Decay=0.985)", color="crimson", linewidth=2)
    
    plt.title("Epsilon Decay Comparison - Rolling Success Rate (Window=50)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_decay_success_comparison.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN Agents for G1 Left Elbow Control.")
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["a", "b", "all"],
        help="Configuration to train (a, b, or all).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
        help="Number of episodes to train for.",
    )
    args = parser.parse_args()

    # Create root results folder
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)

    metrics_a, time_a = None, None
    metrics_b, time_b = None, None

    if args.config in ["a", "all"]:
        metrics_a, time_a = train_configuration(
            config_name="a",
            epsilon_decay=0.995,
            num_episodes=args.episodes,
        )

    if args.config in ["b", "all"]:
        metrics_b, time_b = train_configuration(
            config_name="b",
            epsilon_decay=0.985,
            num_episodes=args.episodes,
        )

    # Generate comparison curves if both were run
    if metrics_a is not None and metrics_b is not None:
        generate_comparison_plots(metrics_a, metrics_b, results_root)
        
        # Output summary comparison table to terminal
        print("\n==================================================")
        print("SUMMARY COMPARISON")
        print("==================================================")
        
        # Compute stats over last 20 training episodes
        last_20_reward_a = np.mean(metrics_a["reward"][-20:])
        last_20_reward_b = np.mean(metrics_b["reward"][-20:])
        
        # Compute success rate over last 50 episodes
        last_50_success_a = np.mean(metrics_a["success"][-50:])
        last_50_success_b = np.mean(metrics_b["success"][-50:])

        print(f"{'Metric':35s} | {'Config A':10s} | {'Config B':10s}")
        print("-" * 63)
        print(f"{'Epsilon Decay Rate':35s} | {'0.995':10s} | {'0.985':10s}")
        print(f"{'Total Training Episodes':35s} | {len(metrics_a['episode']):10d} | {len(metrics_b['episode']):10d}")
        print(f"{'Total Wall-Clock Training Time (s)':35s} | {time_a:10.2f} | {time_b:10.2f}")
        print(f"{'Final Epsilon':35s} | {metrics_a['epsilon'][-1]:10.4f} | {metrics_b['epsilon'][-1]:10.4f}")
        print(f"{'Mean Reward (Final 20 Episodes)':35s} | {last_20_reward_a:10.2f} | {last_20_reward_b:10.2f}")
        print(f"{'Training Success Rate (Final 50 Ep)':35s} | {last_50_success_a*100:9.1f}% | {last_50_success_b*100:9.1f}%")
        print("==================================================")


if __name__ == "__main__":
    main()
