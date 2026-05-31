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
  """

  max_relative_target: float | dict[str, float] = 0.1

  def build(self, env: "ManagerBasedRlEnv") -> "RateLimitedJointPositionAction":
    return RateLimitedJointPositionAction(self, env)


class RateLimitedJointPositionAction(JointPositionAction):
  """Joint-position action with per-step relative clamping.

  Subclasses :class:`mjlab.envs.mdp.actions.JointPositionAction` and only
  overrides ``apply_actions`` to clamp the sent target to
  ``present_pos ± max_relative_target`` per joint, mirroring
  ``lerobot.robots.utils.ensure_safe_goal_position``.
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

    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
