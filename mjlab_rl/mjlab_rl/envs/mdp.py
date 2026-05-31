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


def gripper_open_above_cup_bonus(
  env: "ManagerBasedRlEnv",
  gripper_open_threshold: float = 0.3,
  xy_tol: float = 0.04,
  z_min_above_floor: float = 0.04,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=("gripper",)),
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """Behavioral bonus: 1 when (gripper is open) AND (block is over the cup).

  Pays for *the act of opening the gripper at a useful moment*, independent
  of whether the block then actually lands inside the cup. Without this,
  the policy has no gradient teaching it to ever try opening the gripper —
  the deposit/success rewards only fire after release succeeds, which is
  itself the action the policy needs to learn.

  Triggers when ALL three hold:
    * gripper joint position > ``gripper_open_threshold`` (jaws open),
    * block xy is within ``xy_tol`` of the container body's xy,
    * block z is at least ``z_min_above_floor`` above the container body
      (i.e. block is above the rim, where a release would actually drop
      it into the cup).
  """
  robot: Entity = env.scene[asset_cfg.name]
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]

  gripper_pos = robot.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
  gripper_open = gripper_pos > gripper_open_threshold

  block_pos = block.data.root_link_pos_w
  container_pos = container.data.root_link_pos_w
  xy_close = (
    torch.linalg.norm(block_pos[:, :2] - container_pos[:, :2], dim=-1) < xy_tol
  )
  z_above_rim = (block_pos[:, 2] - container_pos[:, 2]) > z_min_above_floor

  return (gripper_open & xy_close & z_above_rim).float()


def block_deposit_reward(
  env: "ManagerBasedRlEnv",
  std_xy: float = 0.025,
  z_high: float = 0.08,
  z_low: float = 0.02,
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """Dense "block has been deposited inside the cup" reward.

  Product of two smooth gates:
    * ``gauss(xy)`` — peaks at 1 when the block is centered above the
      container body's XY, falls off with ``std_xy``.
    * ``inv_smoothstep(z_above_floor)`` — 0 when the block is at or above
      ``z_high`` (= above the rim, where the policy can hold the block
      while gripping), 1 when the block is at or below ``z_low`` (= settled
      on the cup floor), smooth Hermite in between.

  The crucial property: the product is non-zero only when the block is
  *low and centered relative to the container body*. The closed gripper
  is wider than the cup interior, so this region is physically
  unreachable while held. The only way to claim this reward is to
  release the block above the cup. That gives the release action explicit
  dense credit and bridges the gradient gap leading into ``success_bonus``.
  """
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]

  block_pos = block.data.root_link_pos_w
  container_pos = container.data.root_link_pos_w

  dxy_sq = torch.sum(
    (block_pos[:, :2] - container_pos[:, :2]) ** 2, dim=-1
  )
  xy_gauss = torch.exp(-dxy_sq / (std_xy**2))

  # Inverted Hermite smoothstep: t=0 at z_high, t=1 at z_low, smooth between.
  z_rel = block_pos[:, 2] - container_pos[:, 2]
  t = ((z_high - z_rel) / (z_high - z_low)).clamp(0.0, 1.0)
  z_gate = t * t * (3.0 - 2.0 * t)

  return xy_gauss * z_gate


def post_success_home_pose_reward(
  env: "ManagerBasedRlEnv",
  std: float = 0.6,
  xy_tol: float = 0.020,
  z_max_above_floor: float = 0.055,
  min_z_axis_alignment: float = 0.9,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """Pulls the arm back to its home pose *after* the block is deposited.

  Returns ``deposited * gauss(rms(q - q_home))``, where ``deposited`` is
  the same boolean condition as :func:`success_bonus` (block in cup,
  upright) and ``q_home`` is the entity's ``default_joint_pos``. The
  Gaussian is on the **mean** squared per-joint deviation (not the sum),
  so ``std`` is interpreted in rad-per-joint terms and the reward scale
  doesn't shrink to zero as the joint count grows.

  Sample values with ``std=0.6``:
    * every joint at home          ⇒ 1.0
    * every joint 0.3 rad off (~17°) ⇒ 0.78
    * every joint 0.6 rad off (~34°) ⇒ 0.37
    * every joint 1.0 rad off (~57°) ⇒ 0.06
  """
  robot: Entity = env.scene[asset_cfg.name]
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]

  # Same gating logic as success_bonus.
  block_pos = block.data.root_link_pos_w
  container_pos = container.data.root_link_pos_w
  dxy = block_pos[:, :2] - container_pos[:, :2]
  in_xy = torch.linalg.norm(dxy, dim=-1) < xy_tol
  rel_z = block_pos[:, 2] - container_pos[:, 2]
  in_z = (rel_z > 0.0) & (rel_z < z_max_above_floor)
  q = container.data.root_link_quat_w
  upright = (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)) > min_z_axis_alignment
  deposited = in_xy & in_z & upright

  joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  joint_home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  d2_mean = torch.mean((joint_pos - joint_home) ** 2, dim=-1)
  home_proximity = torch.exp(-d2_mean / (std**2))

  return deposited.float() * home_proximity


def success_bonus(
  env: "ManagerBasedRlEnv",
  xy_tol: float = 0.020,
  z_max_above_floor: float = 0.055,
  min_z_axis_alignment: float = 0.9,
  block_name: str = "block",
  container_name: str = "container",
) -> torch.Tensor:
  """``1`` when the block has settled inside the container's interior
  *and* the container is still upright.

  Defaults assume the bundled container mesh (interior half-width 3.5 cm,
  rim at 5 cm above the body origin) and a 1.5 cm-half-edge block.

  ``xy_tol`` is what's left of the interior after the block takes up its
  share (3.5 − 1.5 = 2.0 cm); ``z_max_above_floor`` is the rim height
  with a little slack; ``min_z_axis_alignment`` is the cos of the max
  allowed tilt from vertical (0.9 ≈ 26°). Yaw rotations don't affect
  success — only tipping does.
  """
  block: Entity = env.scene[block_name]
  container: Entity = env.scene[container_name]

  block_pos = block.data.root_link_pos_w
  container_pos = container.data.root_link_pos_w  # body origin == floor

  dxy = block_pos[:, :2] - container_pos[:, :2]
  in_xy = torch.linalg.norm(dxy, dim=-1) < xy_tol

  rel_z = block_pos[:, 2] - container_pos[:, 2]
  in_z = (rel_z > 0.0) & (rel_z < z_max_above_floor)

  # World-frame z component of the container's local +z axis. For a
  # (w, x, y, z) quaternion the rotation matrix's bottom-right element is
  # ``1 - 2(x² + y²)``: =1 when upright, =-1 when fully upside-down,
  # =0 when tipped 90°.
  q = container.data.root_link_quat_w
  z_axis_dot_world_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
  upright = z_axis_dot_world_z > min_z_axis_alignment

  return (in_xy & in_z & upright).float()


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
# Contact-sensor-based rewards.
# ---------------------------------------------------------------------------


def sensor_contact_penalty(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
) -> torch.Tensor:
  """Returns 1.0 per env if any contact registered by the named
  ContactSensor is active this step, else 0.0. Pair with a negative
  weight to penalise the contact.
  """
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.any(found, dim=-1).float()


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
