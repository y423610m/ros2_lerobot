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
  max_lift: float = 0.10,
  block_name: str = "block",
) -> torch.Tensor:
  """Linear ramp 0 → 1 as the block rises from its initial z up to
  ``max_lift`` above it. Saturates at 1 for any lift ≥ max_lift, returns 0
  for negative lift (block pushed into the table)."""
  block: Entity = env.scene[block_name]
  initial_z = block.data.default_root_state[:, 2]
  current_z = block.data.root_link_pos_w[:, 2]
  return ((current_z - initial_z) / max_lift).clamp(0.0, 1.0)


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
  xy_tol: float = 0.020,
  z_max_above_floor: float = 0.055,
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """``1`` when the block has settled inside the container's interior.

  Defaults assume the bundled container mesh (interior half-width 3.5 cm,
  rim at 5 cm above the body origin) and a 1.5 cm-half-edge block.
  ``xy_tol`` is what's left of the interior after the block takes up its
  share (3.5 − 1.5 = 2.0 cm); ``z_max_above_floor`` is the rim height
  with a little slack.
  """
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]

  block_pos = block.data.root_link_pos_w
  container_pos = container.data.root_link_pos_w  # body origin == floor

  dxy = block_pos[:, :2] - container_pos[:, :2]
  in_xy = torch.linalg.norm(dxy, dim=-1) < xy_tol

  rel_z = block_pos[:, 2] - container_pos[:, 2]
  in_z = (rel_z > 0.0) & (rel_z < z_max_above_floor)

  return (in_xy & in_z).float()


# ---------------------------------------------------------------------------
# Container "stay-put" buffer + snapshot event + penalty reward.
#
# We want to penalise the policy for pushing the container around, but only
# relative to wherever the container ended up after the per-env randomized
# reset — not relative to its un-randomized default. So an event snapshots
# the post-reset XY position into a per-env buffer, and the reward term
# reads from that buffer.
#
# Buffer is keyed by ``id(env)`` so multiple envs in the same process don't
# clobber each other. In practice there's one env per process under mjlab's
# normal usage; the dict will hold a single entry whose value is a
# ``(num_envs, 2)`` tensor.
# ---------------------------------------------------------------------------


_container_init_xy: dict[int, torch.Tensor] = {}
_container_init_quat: dict[int, torch.Tensor] = {}


def snapshot_container_xy(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  container_name: str = "container",
) -> None:
  """Reset-mode event: capture each env's container XY *and quaternion*
  after it has been randomized. Must be ordered after
  ``reset_container_pose``.

  Reset events run before mjlab triggers the next ``sim.forward()``, so the
  entity's data cache is stale when we read it here. Calling ``sim.forward()``
  ourselves refreshes ``root_link_pos_w`` against the freshly-written sim
  state.
  """
  env.sim.forward()

  container: Entity = env.scene[container_name]
  current_xy = container.data.root_link_pos_w[:, :2]
  current_quat = container.data.root_link_quat_w  # (B, 4) wxyz

  key = id(env)

  buf_xy = _container_init_xy.get(key)
  if buf_xy is None or buf_xy.shape[0] != env.num_envs:
    buf_xy = current_xy.clone()
    _container_init_xy[key] = buf_xy

  buf_q = _container_init_quat.get(key)
  if buf_q is None or buf_q.shape[0] != env.num_envs:
    buf_q = current_quat.clone()
    _container_init_quat[key] = buf_q

  if env_ids is None:
    buf_xy.copy_(current_xy)
    buf_q.copy_(current_quat)
  else:
    buf_xy[env_ids] = current_xy[env_ids]
    buf_q[env_ids] = current_quat[env_ids]


def container_xy_displacement_sq(
  env: "ManagerBasedRlEnv",
  container_name: str = "container",
) -> torch.Tensor:
  """Squared XY displacement of the container from its snapshotted reset
  position. Pair with a negative reward weight.
  """
  container: Entity = env.scene[container_name]
  current_xy = container.data.root_link_pos_w[:, :2]

  snapshot = _container_init_xy.get(id(env))
  if snapshot is None:
    # No reset has happened yet — return zeros so we don't poison the very
    # first reward read (rsl-rl calls reward functions during env warm-up).
    return torch.zeros(env.num_envs, device=current_xy.device)

  return torch.sum((current_xy - snapshot) ** 2, dim=-1)


def container_rotation_metric(
  env: "ManagerBasedRlEnv",
  container_name: str = "container",
) -> torch.Tensor:
  """Smooth [0, 1] rotation penalty against the snapshotted reset quat.

  Returns ``1 - (q · q_init)^2`` per env. The square makes the metric
  invariant to quaternion double cover (``q`` and ``-q`` give the same
  rotation). Scale of the metric:
      ``θ = 0°   → 0.0``
      ``θ = 30°  → 0.067``
      ``θ = 90°  → 0.5``
      ``θ = 180° → 1.0``
  Pair with a negative reward weight.
  """
  container: Entity = env.scene[container_name]
  current_q = container.data.root_link_quat_w

  snapshot = _container_init_quat.get(id(env))
  if snapshot is None:
    return torch.zeros(env.num_envs, device=current_q.device)

  dot = torch.sum(current_q * snapshot, dim=-1)
  return 1.0 - dot * dot


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
