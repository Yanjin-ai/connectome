import math

import torch.nn as nn

from src.connectome_gnn import ConnectomeController


class MLPController(nn.Module):
    """Two-layer MLP baseline with roughly matched parameter count."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 32):
        super().__init__()

        ref = ConnectomeController(obs_dim=obs_dim, act_dim=act_dim)
        target_params = ref.count_params()

        h = max(hidden_dim, int(math.sqrt(target_params / 2)))

        self.net = nn.Sequential(
            nn.Linear(obs_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, act_dim),
        )
        actual = self.count_params()
        print(f"MLP hidden={h}, params={actual:,} (target={target_params:,})")

    def forward(self, obs):
        return self.net(obs)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
