from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    DQN Q-Network for Unitree G1 elbow joint control.

    Architecture:
        Input: 4 dimensions (observation vector: [angle, velocity, goal, error])
        Hidden Layer 1: 64 units, ReLU activation
        Hidden Layer 2: 64 units, ReLU activation
        Output: 3 dimensions (unconstrained action-value estimates)
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 3) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        # Weight initialization for stability
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute action values."""
        return self.net(state)


def get_device() -> torch.device:
    """
    Select the best available device for training.
    Optimized for MacBook Pro M1 (MPS), standard NVIDIA GPUs (CUDA), and CPU fallback.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
