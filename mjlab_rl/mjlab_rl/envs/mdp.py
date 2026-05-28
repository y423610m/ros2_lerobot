"""Custom MDP terms for the SO-101 block pick-and-place task.

Conventions match :mod:`mjlab.tasks.manipulation.mdp`: every term takes the
:class:`mjlab.envs.ManagerBasedRlEnv` as the first argument plus any term-
specific keyword params declared in the ``ObservationTermCfg`` / ``RewardTermCfg``.
All returned tensors have batch dimension ``num_envs``.

Geometry note: the SO-101's flange-fixed ``gripperframe`` site is the most
stable choice for an end-effector reference because it does not move when the
gripper opens or closes (the moving jaw site does). Reward shaping and the
success check use this site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Observations.
# ---------------------------------------------------------------------------


def ee_to_block(
  env: "ManagerBasedRlEnv",
  block_name: str = "block",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Vector from the gripper site to the block, in world frame. ``(B, 3)``."""
  robot: Entity = env.scene[asset_cfg.name]
  block: Entity = env.scene[block_name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return block.data.root_link_pos_w - ee_pos


def block_to_container(
  env: "ManagerBasedRlEnv",
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """Vector from the block to the container's interior site, in world. ``(B, 3)``."""
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]
  # Container is a floating-base entity with a single registered site
  # ("container_site") that sits at the centre of the interior.
  container_site_pos = container.data.site_pos_w[:, 0]
  return container_site_pos - block.data.root_link_pos_w


def block_pose(
  env: "ManagerBasedRlEnv",
  block_name: str = "block",
) -> torch.Tensor:
  """Block ``(pos_xyz, quat_wxyz)``. ``(B, 7)``."""
  block: Entity = env.scene[block_name]
  return torch.cat(
    (block.data.root_link_pos_w, block.data.root_link_quat_w), dim=-1
  )


# ---------------------------------------------------------------------------
# Rewards.
# ---------------------------------------------------------------------------


def reach_block_reward(
  env: "ManagerBasedRlEnv",
  std: float = 0.10,
  block_name: str = "block",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Gaussian kernel of ``||ee - block||`` (closer ⇒ ≈1)."""
  robot: Entity = env.scene[asset_cfg.name]
  block: Entity = env.scene[block_name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  d2 = torch.sum((ee_pos - block.data.root_link_pos_w) ** 2, dim=-1)
  return torch.exp(-d2 / (std**2))


def lift_block_reward(
  env: "ManagerBasedRlEnv",
  min_lift: float = 0.02,
  block_name: str = "block",
) -> torch.Tensor:
  """Binary 0/1 reward: block lifted at least ``min_lift`` above its initial z."""
  block: Entity = env.scene[block_name]
  initial_z = block.data.default_root_state[:, 2]
  current_z = block.data.root_link_pos_w[:, 2]
  return (current_z - initial_z > min_lift).float()


def place_block_reward(
  env: "ManagerBasedRlEnv",
  std: float = 0.10,
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """Gaussian reward on block→container distance, gated by lift."""
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]
  container_site_pos = container.data.site_pos_w[:, 0]
  d2 = torch.sum(
    (block.data.root_link_pos_w - container_site_pos) ** 2, dim=-1
  )
  return torch.exp(-d2 / (std**2))


def success_bonus(
  env: "ManagerBasedRlEnv",
  xy_tol: float = 0.045,
  z_max_above_floor: float = 0.05,
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """``1`` when the block has settled inside the container's interior."""
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]

  block_pos = block.data.root_link_pos_w
  container_pos = container.data.root_link_pos_w  # bottom of container

  dxy = block_pos[:, :2] - container_pos[:, :2]
  in_xy = torch.linalg.norm(dxy, dim=-1) < xy_tol

  rel_z = block_pos[:, 2] - container_pos[:, 2]
  in_z = (rel_z > 0.0) & (rel_z < z_max_above_floor)

  return (in_xy & in_z).float()


# ---------------------------------------------------------------------------
# Terminations.
# ---------------------------------------------------------------------------


def block_dropped(
  env: "ManagerBasedRlEnv",
  min_z: float = -0.05,
  block_name: str = "block",
) -> torch.Tensor:
  """Terminate if the block falls below the floor by more than ``|min_z|``."""
  block: Entity = env.scene[block_name]
  return block.data.root_link_pos_w[:, 2] < min_z
