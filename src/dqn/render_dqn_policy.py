from __future__ import annotations

import argparse
import time
from pathlib import Path

from dqn.agent import DQNAgent
from g1_rl import G1ElbowTargetEnv


def wait_for_user(goal: float) -> None:
    print()
    print("==================================================")
    print(f"NEXT TARGET GOAL: {goal:+.1f} rad")
    print("==================================================")
    print("Adjust the camera position (pan/zoom/rotate) in the MuJoCo window.")
    print("When ready, return here and press Enter to start the simulation.")
    print("==================================================")
    input("Press Enter to begin...")
    print("Starting simulation...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trained G1 DQN policy in MuJoCo.")
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

    # Initialize environment in human rendering mode
    print("Initializing environment in human render mode...")
    env = G1ElbowTargetEnv(render_mode="human")

    # Load agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    agent.load(model_path)
    agent.online_net.eval()

    # We will show the agent controlling the elbow to at least two benchmark goals: -0.8 rad and +0.8 rad
    # as required by video requirements (Show at least two different target angles).
    test_goals = [-0.8, 0.8]
    
    try:
        for idx, goal in enumerate(test_goals):
            wait_for_user(goal)
            
            # Reset env with target goal
            state, info = env.reset(seed=123 + idx, options={"goal_angle": goal})
            env.render()
            
            # Run simulation episode
            cumulative_reward = 0.0
            steps = 0
            
            # Run simulation for a fixed number of steps (300 steps = 6 seconds at 50 FPS)
            # to show the holding stability of the learned policy and allow recording.
            max_steps = 300
            for step_idx in range(max_steps):
                # Epsilon = 0.0 for greedy execution
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                steps += 1
                cumulative_reward += reward
                state = next_state
                
                # Print real-time state for recording console metrics
                print(
                    f"step={steps:3d} | "
                    f"action={action} | "
                    f"angle={info['elbow_angle']:+.4f} rad | "
                    f"target={info['controller_target']:+.4f} rad | "
                    f"goal={info['goal_angle']:+.4f} rad | "
                    f"error={info['angle_error']:+.4f} rad | "
                    f"streak={info['success_streak']:2d} | "
                    f"reward={reward:+.4f}"
                )
                
                # Check if viewer closed
                if env.viewer is not None and not env.viewer.is_running():
                    print("Viewer window closed by user.")
                    break

            print()
            print("==================================================")
            print("EPISODE RESULTS")
            print("==================================================")
            print(f"Success Streak Reached: {info['success_streak'] >= env.required_success_steps}")
            print(f"Total Steps run:       {steps}")
            print(f"Final Angle:       {info['elbow_angle']:.4f} rad")
            print(f"Goal Angle:        {info['goal_angle']:.4f} rad")
            print(f"Final Error:       {info['angle_error']:.4f} rad")
            print(f"Cumulative Reward: {cumulative_reward:.4f}")
            print("==================================================")
            
            # Brief pause before next run if there is one
            if idx < len(test_goals) - 1:
                time.sleep(2.0)
                
    finally:
        env.close()
        print("\nSimulation session finished.")


if __name__ == "__main__":
    main()
