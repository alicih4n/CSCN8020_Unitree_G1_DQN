from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay memory buffer for training the DQN agent.
    
    Uses a double-ended queue (deque) of a fixed capacity. Once full,
    the oldest transition is automatically discarded when a new one is added.
    """

    def __init__(self, capacity: int = 50000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store a new experience transition."""
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample a random mini-batch of transitions and move them to the target device.
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, terminateds = zip(*batch)

        # Convert to numpy arrays first for efficiency, then PyTorch tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        terminateds_t = torch.tensor(np.array(terminateds), dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, terminateds_t

    def __len__(self) -> int:
        return len(self.buffer)
