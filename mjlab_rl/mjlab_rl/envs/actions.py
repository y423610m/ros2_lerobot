"""Action terms specific to this project.

Currently just :class:`RateLimitedJointPositionActionCfg`, a drop-in
replacement for ``mjlab.envs.mdp.actions.JointPositionActionCfg`` that
mirrors the safety clamp the real lerobot SO-follower applies to every
``send_action`` call:

    diff = goal_pos - present_pos
    diff = clamp(diff, -max_relative_target, +max_relative_target)
    goal_pos = present_pos + diff

This prevents the policy from commanding a target wildly far from the
current joint position in a single env step, which keeps the simulated
trajectory consistent with what the real arm can physically execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  JointPositionActionCfg,
)
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
  """``JointPositionActionCfg`` + per-step delta clamp from present pos.

  ``max_relative_target`` is the largest absolute per-joint change (in
  radians) between the present joint position and the commanded target
  in a single env step. Pass a float for a uniform cap, or a dict mapping
  joint-name regex → cap for per-joint values.

  Optional servo-realism model (sim-to-real): a first-order low-pass filter on
  the commanded target (limited servo bandwidth) plus an integer transport
  delay (latency). Both default to *off* (``lpf_alpha_default=1.0`` = no
  filtering, ``max_action_lag=0`` = no delay) so behavior is unchanged unless
  the ``randomize_actuator_lag`` DR event turns them on per-episode. The filter
  and the delay buffer operate at the physics-substep rate (``apply_actions`` is
  called once per substep), so lag units are *substeps* (= ``physics_dt`` each).
  """

  max_relative_target: float | dict[str, float] = 0.1
  max_action_lag: int = 0
  """Delay-buffer capacity in physics substeps. 0 = no delay buffer allocated."""
  lpf_alpha_default: float = 1.0
  """Default per-substep low-pass coefficient. 1.0 = no filtering (instant)."""

  def build(self, env: "ManagerBasedRlEnv") -> "RateLimitedJointPositionAction":
    return RateLimitedJointPositionAction(self, env)


class RateLimitedJointPositionAction(JointPositionAction):
  """Joint-position action with per-step relative clamping and an optional
  servo-realism model.

  Subclasses :class:`mjlab.envs.mdp.actions.JointPositionAction`. Overrides
  ``apply_actions`` to (1) clamp the sent target to
  ``present_pos ± max_relative_target`` per joint, mirroring
  ``lerobot.robots.utils.ensure_safe_goal_position``, then (2) optionally pass
  it through a per-joint low-pass filter and a per-env transport delay so the
  simulated arm tracks targets with the finite bandwidth + latency of the real
  STS-3215 servos. The filter/delay strengths are per-env state set by the
  ``randomize_actuator_lag`` event; they stay inert (alpha 1, lag 0) otherwise.
  """

  def __init__(
    self, cfg: RateLimitedJointPositionActionCfg, env: "ManagerBasedRlEnv"
  ) -> None:
    super().__init__(cfg, env)

    if isinstance(cfg.max_relative_target, (float, int)):
      self._max_rel: torch.Tensor | float = float(cfg.max_relative_target)
    else:
      buf = torch.zeros(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.max_relative_target, self._target_names
      )
      buf[:, index_list] = torch.tensor(value_list, device=self.device)
      self._max_rel = buf

    # Servo-realism state (all joints). Per-(env, joint) low-pass coefficient;
    # per-env transport lag (substeps). Defaults = inert.
    self._lpf_alpha = torch.full(
      (self.num_envs, self.action_dim),
      float(cfg.lpf_alpha_default),
      device=self.device,
    )
    self._filtered_target: torch.Tensor | None = None  # lazy init on first reset
    self._max_action_lag = int(cfg.max_action_lag)
    if self._max_action_lag > 0:
      self._delay_hist = CircularBuffer(
        max_len=self._max_action_lag + 1,
        batch_size=self.num_envs,
        device=str(self.device),
      )
      self._action_lag = torch.zeros(
        self.num_envs, dtype=torch.long, device=self.device
      )
    else:
      self._delay_hist = None
      self._action_lag = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    # Reset the filter to track the present pose so a new episode starts without
    # carrying over the previous episode's lag state.
    present = self._entity.data.joint_pos[:, self._target_ids]
    if self._filtered_target is None:
      self._filtered_target = present.clone()
    else:
      idx = slice(None) if env_ids is None else env_ids
      self._filtered_target[idx] = present[idx]
    if self._delay_hist is not None:
      self._delay_hist.reset(batch_ids=env_ids)

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    present = self._entity.data.joint_pos[:, self._target_ids]

    if isinstance(self._max_rel, float):
      diff = (target - present).clamp(-self._max_rel, self._max_rel)
    else:
      diff = torch.maximum(
        torch.minimum(target - present, self._max_rel), -self._max_rel
      )
    target = present + diff

    # Servo bandwidth: first-order low-pass on the target (per joint). With
    # alpha == 1 this is a no-op (filtered == target).
    if self._filtered_target is None:
      self._filtered_target = present.clone()
    self._filtered_target = (
      self._lpf_alpha * target + (1.0 - self._lpf_alpha) * self._filtered_target
    )
    out = self._filtered_target

    # Servo latency: return the target from `lag` substeps ago (per env). With
    # lag == 0 this returns the just-pushed target (no delay).
    if self._delay_hist is not None:
      self._delay_hist.append(out)
      lag = torch.minimum(
        self._action_lag, self._delay_hist.current_length - 1
      ).clamp_min(0)
      out = self._delay_hist[lag]

    self._entity.set_joint_position_target(out, joint_ids=self._target_ids)
