from __future__ import annotations

import argparse
import time
from pathlib import Path

from dqn.agent import DQNAgent
from g1_rl import G1ElbowTargetEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render G1 DQN policy in a continuous back-and-forth movement.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/selected_dqn.pt"),
        help="Path to the trained PyTorch checkpoint.",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print("Initializing environment in human render mode...")
    env = G1ElbowTargetEnv(render_mode="human")

    # Load agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    agent.load(model_path)
    agent.online_net.eval()

    print("\n=== CONTINUOUS ELBOW MOVEMENT DEMO ===")
    print("The robot will automatically swing its left elbow back and forth.")
    print("Close the viewer window or press Ctrl+C in the terminal to exit.")
    print("--------------------------------------------------")

    # Targets to alternate between
    goals = [-0.8, 0.8]
    goal_idx = 0
    state, info = env.reset(seed=42, options={"goal_angle": goals[goal_idx]})
    
    steps = 0
    
    try:
        while env.viewer is not None and env.viewer.is_running():
            # Select greedy action
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            steps += 1
            state = next_state
            
            # Check if success streak achieved or target reached
            # If the joint is close to the current goal, switch to the other goal!
            current_error = abs(info["angle_error"])
            
            # Print state details periodically
            if steps % 10 == 0:
                print(
                    f"Step {steps:4d} | "
                    f"Elbow Angle: {info['elbow_angle']:+.4f} rad | "
                    f"Goal: {info['goal_angle']:+.4f} rad | "
                    f"Error: {info['angle_error']:+.4f} rad"
                )

            # Switch goals if the current goal is reached and held
            if terminated or (current_error <= env.success_tolerance and info["success_streak"] >= 5):
                goal_idx = (goal_idx + 1) % len(goals)
                new_goal = goals[goal_idx]
                print(f"\n[Goal Reached!] Switching target to: {new_goal:+.1f} rad\n")
                
                # Keep the same physics state, but update the goal angle in the env
                env.goal_angle = new_goal
                # Update observation with the new goal
                state = env._get_observation()
                env.success_streak = 0
                
            # If the episode length runs too long, reset the environment to keep physics stable
            if steps % 300 == 0:
                # Reset to current goal
                state, info = env.reset(seed=42 + steps, options={"goal_angle": goals[goal_idx]})
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
