"""Build the env on CPU and step it a few times — used to validate the config.

Run with::

  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import torch

import mjlab_rl  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-SO101-Block-Picking"


def main(num_envs: int = 2, num_steps: int = 16) -> None:
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cpu")

  obs, _ = env.reset()
  print(f"reset OK: actor={tuple(obs['actor'].shape)} critic={tuple(obs['critic'].shape)}")

  action = torch.zeros(
    (env.num_envs, env.action_manager.total_action_dim), device="cpu"
  )
  total = torch.zeros(env.num_envs, device="cpu")
  for t in range(num_steps):
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if (terminated | truncated).any():
      print(f"  step {t}: term={terminated.tolist()} trunc={truncated.tolist()}")
  print(f"stepped {num_steps} steps; cumulative reward per env = {total.tolist()}")


if __name__ == "__main__":
  main()
