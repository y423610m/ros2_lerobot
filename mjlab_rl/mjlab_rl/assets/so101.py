"""SO-101 arm entity config for mjlab.

Wraps the existing SO-101 MJCF (shipped with ``lerobot_robots_description``)
into an :class:`mjlab.entity.EntityCfg`. The MJCF already defines six
``<position>`` actuators (5 arm joints + 1 gripper finger), so we reuse them
via :class:`mjlab.actuator.XmlActuatorCfg` rather than re-specifying gains.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

SO101_XML: Path = (
  Path(__file__).resolve().parents[3]
  / "src"
  / "lerobot_robots_description"
  / "urdf"
  / "SO101"
  / "so101_new_calib_stable_grasp.xml"
)
assert SO101_XML.exists(), f"SO101 MJCF not found at {SO101_XML}"

SO101_JOINT_NAMES: tuple[str, ...] = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
)

ARM_JOINT_NAMES: tuple[str, ...] = SO101_JOINT_NAMES[:5]
GRIPPER_JOINT_NAME: str = SO101_JOINT_NAMES[5]

# Joint position targets at episode start. Values were copied from the original
# scripted oracle's home pose; keeps the arm comfortably above the table.
HOME_JOINT_POS: dict[str, float] = {
  "shoulder_pan": 0.0,
  "shoulder_lift": -0.3,
  "elbow_flex": 0.6,
  "wrist_flex": 0.3,
  "wrist_roll": 0.0,
  "gripper": 0.02,
}

# Approximate distance (radians) each joint can move per env step. The PPO
# action is multiplied by this scale before being added to the home pose, so
# smaller values give the policy a tighter operating envelope around home.
SO101_ACTION_SCALE: dict[str, float] = {
  "shoulder_pan": 0.5,
  "shoulder_lift": 0.5,
  "elbow_flex": 0.5,
  "wrist_flex": 0.5,
  "wrist_roll": 0.5,
  "gripper": 0.8,
}


def _strip_world_floor(spec: mujoco.MjSpec) -> None:
  """Remove the placeholder floor geom the SO-101 XML drops into the world."""
  for geom in list(spec.worldbody.geoms):
    if geom.name == "__mjv_default_floor":
      spec.delete(geom)


def get_so101_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SO101_XML))
  # The XML declares meshdir="assets" *and* mesh files prefixed with "assets/",
  # so the resolver doubles up. Clear meshdir to fix.
  spec.meshdir = ""
  _strip_world_floor(spec)
  return spec


# Reuse the XML <position> actuators (kp=17.8 from sts3215 class).
# The pattern targets the six SO-101 joints explicitly to avoid matching the
# arm's named sites (baseframe, gripperframe, ee_*, ...).
SO101_ACTUATORS = (
  XmlActuatorCfg(target_names_expr=SO101_JOINT_NAMES),
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=SO101_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_so101_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      joint_pos=HOME_JOINT_POS,
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_so101_spec,
    articulation=ARTICULATION,
  )
