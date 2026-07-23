from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn.agent import DQNAgent
from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action


BENCHMARK_GOALS = [-0.8, -0.4, 0.4, 0.8]
EPISODES_PER_GOAL = 5
TOTAL_EPISODES = len(BENCHMARK_GOALS) * EPISODES_PER_GOAL
EVAL_SEED_OFFSET = 10000  # Unique seeds for evaluation


def evaluate_dqn_agent(
    model_path: Path,
    env: G1ElbowTargetEnv,
) -> tuple[list[dict], dict]:
    """
    Evaluate a DQN agent greedily (epsilon=0.0) over the 20 benchmark episodes.
    """
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    agent.load(model_path)
    agent.online_net.eval()

    episode_records = []
    goal_stats = {goal: {"successes": 0, "rewards": []} for goal in BENCHMARK_GOALS}

    episode_idx = 0
    for goal in BENCHMARK_GOALS:
        for run in range(EPISODES_PER_GOAL):
            seed = EVAL_SEED_OFFSET + episode_idx
            
            # Reset environment with specific goal
            # Pass goal_angle in reset options to override standard goal sampling
            state, info = env.reset(seed=seed, options={"goal_angle": goal})
            
            episode_reward = 0.0
            steps = 0
            
            while True:
                # Epsilon = 0.0 for greedy evaluation
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                steps += 1
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break

            success = 1.0 if info["success_streak"] >= env.required_success_steps else 0.0
            
            record = {
                "episode": episode_idx + 1,
                "goal": goal,
                "reward": episode_reward,
                "steps": steps,
                "final_error": info["absolute_error"],
                "success": success,
            }
            episode_records.append(record)
            
            # Record goal specific stats
            goal_stats[goal]["rewards"].append(episode_reward)
            if success > 0:
                goal_stats[goal]["successes"] += 1
                
            episode_idx += 1

    return episode_records, goal_stats


def evaluate_rule_based(
    env: G1ElbowTargetEnv,
) -> tuple[list[dict], dict]:
    """
    Evaluate the rule-based policy over the same 20 benchmark episodes.
    """
    episode_records = []
    goal_stats = {goal: {"successes": 0, "rewards": []} for goal in BENCHMARK_GOALS}

    episode_idx = 0
    for goal in BENCHMARK_GOALS:
        for run in range(EPISODES_PER_GOAL):
            seed = EVAL_SEED_OFFSET + episode_idx
            state, info = env.reset(seed=seed, options={"goal_angle": goal})
            
            episode_reward = 0.0
            steps = 0
            
            while True:
                # Select rule-based action
                action = choose_rule_based_action(
                    observation=state,
                    controller_target=float(info["controller_target"]),
                    action_increment=env.action_increment,
                )
                
                next_state, reward, terminated, truncated, info = env.step(action)
                
                steps += 1
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break

            success = 1.0 if info["success_streak"] >= env.required_success_steps else 0.0
            
            record = {
                "episode": episode_idx + 1,
                "goal": goal,
                "reward": episode_reward,
                "steps": steps,
                "final_error": info["absolute_error"],
                "success": success,
            }
            episode_records.append(record)
            
            # Record goal specific stats
            goal_stats[goal]["rewards"].append(episode_reward)
            if success > 0:
                goal_stats[goal]["successes"] += 1
                
            episode_idx += 1

    return episode_records, goal_stats


def write_eval_csv(filepath: Path, records: list[dict]) -> None:
    """Save evaluation metrics to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(records[0].keys())
        for r in records:
            writer.writerow(r.values())


def compute_summary_metrics(records: list[dict]) -> dict:
    """Compute aggregate evaluation metrics across all episodes."""
    successes = [r["success"] for r in records]
    rewards = [r["reward"] for r in records]
    lengths = [r["steps"] for r in records]
    errors = [r["final_error"] for r in records]

    return {
        "success_rate": np.mean(successes),
        "success_count": int(np.sum(successes)),
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "mean_final_error": np.mean(errors),
    }


def print_markdown_comparison_table(
    rule_metrics: dict,
    dqn_a_metrics: dict,
    dqn_b_metrics: dict,
) -> None:
    """Print the final comparison table in Markdown format."""
    print("\n==================================================")
    print("FINAL BENCHMARK COMPARISON TABLE (20 EPISODES)")
    print("==================================================")
    print(f"| {'Metric':30s} | {'Rule-Based Policy':18s} | {'DQN Config A':15s} | {'DQN Config B':15s} |")
    print(f"|{'-'*32}|{'-'*20}|{'-'*17}|{'-'*17}|")
    print(
        f"| {'Successes / 20':30s} | "
        f"{rule_metrics['success_count']:18d} | "
        f"{dqn_a_metrics['success_count']:15d} | "
        f"{dqn_b_metrics['success_count']:15d} |"
    )
    print(
        f"| {'Success Rate':30s} | "
        f"{rule_metrics['success_rate']*100:16.1f}% | "
        f"{dqn_a_metrics['success_rate']*100:13.1f}% | "
        f"{dqn_b_metrics['success_rate']*100:13.1f}% |"
    )
    print(
        f"| {'Mean Cumulative Reward':30s} | "
        f"{rule_metrics['mean_reward']:18.2f} | "
        f"{dqn_a_metrics['mean_reward']:15.2f} | "
        f"{dqn_b_metrics['mean_reward']:15.2f} |"
    )
    print(
        f"| {'Mean Episode Length (steps)':30s} | "
        f"{rule_metrics['mean_length']:18.1f} | "
        f"{dqn_a_metrics['mean_length']:15.1f} | "
        f"{dqn_b_metrics['mean_length']:15.1f} |"
    )
    print(
        f"| {'Mean Final Absolute Error':30s} | "
        f"{rule_metrics['mean_final_error']:18.4f} | "
        f"{dqn_a_metrics['mean_final_error']:15.4f} | "
        f"{dqn_b_metrics['mean_final_error']:15.4f} |"
    )
    print("==================================================\n")


def print_goal_summary_table(
    config_name: str,
    goal_stats: dict,
) -> None:
    """Print the per-goal statistics table requested by the rubric."""
    print(f"=== FINAL EVALUATION TABLE - CONFIGURATION {config_name.upper()} ===")
    print(f"| {'Goal (rad)':10s} | {'Episodes':8s} | {'Successes':9s} | {'Success Rate':12s} | {'Mean Reward':11s} |")
    print(f"|{'-'*12}|{'-'*10}|{'-'*11}|{'-'*14}|{'-'*13}|")
    
    total_successes = 0
    total_rewards = []
    
    for goal in BENCHMARK_GOALS:
        stats = goal_stats[goal]
        successes = stats["successes"]
        success_rate = successes / EPISODES_PER_GOAL
        mean_reward = np.mean(stats["rewards"])
        
        total_successes += successes
        total_rewards.extend(stats["rewards"])
        
        print(
            f"| {goal:+.1f} rad   | "
            f"{EPISODES_PER_GOAL:8d} | "
            f"{successes:9d} | "
            f"{success_rate*100:11.1f}% | "
            f"{mean_reward:11.2f} |"
        )
        
    overall_success_rate = total_successes / TOTAL_EPISODES
    print(
        f"| {'Overall':10s} | "
        f"{TOTAL_EPISODES:8d} | "
        f"{total_successes:9d} | "
        f"{overall_success_rate*100:11.1f}% | "
        f"{np.mean(total_rewards):11.2f} |"
    )
    print("===============================================================\n")


def generate_success_by_angle_plot(
    rule_stats: dict,
    dqn_a_stats: dict,
    dqn_b_stats: dict,
    output_dir: Path,
) -> None:
    """Generate bar plot comparing success rates by target angle."""
    x = np.arange(len(BENCHMARK_GOALS))
    width = 0.25

    rule_rates = [rule_stats[g]["successes"] / EPISODES_PER_GOAL for g in BENCHMARK_GOALS]
    dqn_a_rates = [dqn_a_stats[g]["successes"] / EPISODES_PER_GOAL for g in BENCHMARK_GOALS]
    dqn_b_rates = [dqn_b_stats[g]["successes"] / EPISODES_PER_GOAL for g in BENCHMARK_GOALS]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, rule_rates, width, label="Rule-Based Policy", color="darkgrey")
    rects2 = ax.bar(x, dqn_a_rates, width, label="DQN Config A (Decay=0.995)", color="navy")
    rects3 = ax.bar(x + width, dqn_b_rates, width, label="DQN Config B (Decay=0.985)", color="crimson")

    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Target Angle (Goal)")
    ax.set_title("Greedy Evaluation Success Rate by Target Angle")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g:+.1f} rad" for g in BENCHMARK_GOALS])
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend()

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height*100:.0f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_success_by_angle.png", dpi=150)
    plt.close()


def main() -> None:
    print("Initializing environment for evaluation...")
    env = G1ElbowTargetEnv(render_mode=None)

    checkpoint_a = Path("models/dqn_config_a.pt")
    checkpoint_b = Path("models/dqn_config_b.pt")

    if not checkpoint_a.is_file() or not checkpoint_b.is_file():
        raise RuntimeError("Missing model checkpoints. Run training before evaluation.")

    # 1. Evaluate Rule-based baseline
    print("Evaluating Rule-based policy...")
    rule_records, rule_goal_stats = evaluate_rule_based(env)
    rule_summary = compute_summary_metrics(rule_records)
    write_eval_csv(Path("results/rule_based_evaluation_metrics.csv"), rule_records)

    # 2. Evaluate DQN Configuration A
    print("Evaluating DQN Configuration A...")
    dqn_a_records, dqn_a_goal_stats = evaluate_dqn_agent(checkpoint_a, env)
    dqn_a_summary = compute_summary_metrics(dqn_a_records)
    write_eval_csv(Path("results/config_a/evaluation_metrics.csv"), dqn_a_records)

    # 3. Evaluate DQN Configuration B
    print("Evaluating DQN Configuration B...")
    dqn_b_records, dqn_b_goal_stats = evaluate_dqn_agent(checkpoint_b, env)
    dqn_b_summary = compute_summary_metrics(dqn_b_records)
    write_eval_csv(Path("results/config_b/evaluation_metrics.csv"), dqn_b_records)

    # Close the environment
    env.close()

    # 4. Generate Markdown Tables
    print_markdown_comparison_table(rule_summary, dqn_a_summary, dqn_b_summary)
    print_goal_summary_table("a", dqn_a_goal_stats)
    print_goal_summary_table("b", dqn_b_goal_stats)

    # 5. Generate Success by Target Angle Plot
    generate_success_by_angle_plot(
        rule_goal_stats,
        dqn_a_goal_stats,
        dqn_b_goal_stats,
        Path("results"),
    )
    print("Evaluation plot generated at: results/evaluation_success_by_angle.png")

    # 6. Save the superior model to models/selected_dqn.pt
    # Select best model based on overall success rate, using mean reward as tie breaker.
    if dqn_a_summary["success_rate"] > dqn_b_summary["success_rate"]:
        best_config = "A"
        shutil.copy(checkpoint_a, "models/selected_dqn.pt")
    elif dqn_b_summary["success_rate"] > dqn_a_summary["success_rate"]:
        best_config = "B"
        shutil.copy(checkpoint_b, "models/selected_dqn.pt")
    else:
        # Tie breaker: choose the one with higher mean reward
        if dqn_a_summary["mean_reward"] >= dqn_b_summary["mean_reward"]:
            best_config = "A"
            shutil.copy(checkpoint_a, "models/selected_dqn.pt")
        else:
            best_config = "B"
            shutil.copy(checkpoint_b, "models/selected_dqn.pt")

    print(f"\n>>> Selected Configuration {best_config} as the superior model. Saved as models/selected_dqn.pt")


if __name__ == "__main__":
    main()
