from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.q_network import QNetwork, get_device
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent for G1 left elbow control.

    Contains the online and target networks, action selection logic (epsilon-greedy),
    Bellman temporal-difference learning step, target-network synchronization,
    and checkpoint saving/loading.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 3,
        lr: float = 0.001,
        gamma: float = 0.95,
        target_update_interval: int = 250,
        device: torch.device | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        # Device selection (CPU, CUDA, or macOS M1 MPS)
        self.device = device if device is not None else get_device()
        print(f"DQNAgent initialized on device: {self.device}")

        # Instantiate networks
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)

        # Initialize target network from online network (hard weight copy)
        self.update_target_network()
        self.target_net.eval()  # Target net is only used in evaluation/bootstrap

        # Optimizer: Adam is standard and robust
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Loss function: Huber loss is robust to outliers and standard for DQN stability
        self.loss_fn = nn.SmoothL1Loss()

        self.optimization_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using the epsilon-greedy strategy.
        If epsilon is 0.0, the action is selected greedily.
        """
        if random.random() < epsilon:
            # Exploration: select a random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: select the greedy action (highest Q-value)
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_net(state_t)
                action = q_values.argmax(dim=1).item()
            return int(action)

    def optimize_model(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 64,
    ) -> float | None:
        """
        Perform a single gradient descent step to optimize the online network.
        
        Uses temporal-difference (TD) learning targets based on the Bellman equation
        with target-network bootstrapping.
        """
        if len(replay_buffer) < batch_size:
            return None

        # Sample a mini-batch of transitions
        states, actions, rewards, next_states, terminateds = replay_buffer.sample(
            batch_size=batch_size,
            device=self.device,
        )

        # Compute Q(s, a) using the online network
        state_action_values = self.online_net(states).gather(1, actions)

        # Compute max_a' Q_target(s', a') using the target network
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Bellman equation: target = reward + gamma * max Q * (1 - terminated)
            # Terminated mask is crucial: prevents bootstrapping from true terminal states
            expected_state_action_values = rewards + (1 - terminateds) * self.gamma * next_state_values

        # Compute Huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for training stability (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.optimization_steps += 1

        # Check target network synchronization condition
        if self.optimization_steps % self.target_update_interval == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Perform a hard copy of online network weights to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, filepath: str | Path) -> None:
        """Save Q-network state dictionary to a checkpoint file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net_state_dict": self.online_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
            },
            filepath,
        )
        print(f"Agent checkpoint saved successfully to: {filepath}")

    def load(self, filepath: str | Path) -> None:
        """Load Q-network state dictionary from a checkpoint file."""
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.optimization_steps = checkpoint["optimization_steps"]
        print(f"Agent checkpoint loaded successfully from: {filepath}")
