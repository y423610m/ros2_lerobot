"""MLP policy networks for BC and policy distillation.

Both the BC ("teacher" trained on oracle data) and the distilled student
policy use the same class; you just pass different hidden sizes.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        act_limit: float = 1.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), act_cls()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.act_limit = float(act_limit)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_limit * self.net(obs)

    @torch.no_grad()
    def act_numpy(self, obs):
        import numpy as np
        x = torch.as_tensor(obs, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        a = self.forward(x).cpu().numpy()
        return a[0] if a.shape[0] == 1 else a

    # ----- (de)serialization helpers used by the scripts -----------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_sizes": list(self.hidden_sizes),
                "act_limit": self.act_limit,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str | None = None) -> "MLPPolicy":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(
            obs_dim=blob["obs_dim"],
            act_dim=blob["act_dim"],
            hidden_sizes=blob["hidden_sizes"],
            act_limit=blob["act_limit"],
        )
        net.load_state_dict(blob["state_dict"])
        return net
