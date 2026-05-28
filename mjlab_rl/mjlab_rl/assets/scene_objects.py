"""MjSpec builders for the table, manipulable block, and goal container."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

# Block dimensions (half-extents) and color.
BLOCK_HALF_SIZE: float = 0.015
BLOCK_MASS: float = 0.05

# Container mesh paths — same STLs the original block_picking.xml used.
_LEROBOT_MESHES = (
  Path(__file__).resolve().parents[3]
  / "src"
  / "lerobot_robots_description"
  / "meshes"
)
CONTAINER_VISUAL_STL = _LEROBOT_MESHES / "container_visual.stl"
CONTAINER_COLLISION_STL = _LEROBOT_MESHES / "container_collision.stl"
assert CONTAINER_VISUAL_STL.exists(), f"missing {CONTAINER_VISUAL_STL}"
assert CONTAINER_COLLISION_STL.exists(), f"missing {CONTAINER_COLLISION_STL}"

# Approximate rim height of the container mesh above its own origin. Used to
# place the ``container_site`` at the interior so the success check has a
# reasonable goal point. Calibrated to match the original scene.
CONTAINER_RIM_HEIGHT: float = 0.046


def _make_block_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="block")
  body.add_freejoint(name="block_freejoint")
  body.add_geom(
    name="block_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(BLOCK_HALF_SIZE,) * 3,
    mass=BLOCK_MASS,
    rgba=(0.90, 0.30, 0.10, 1.0),
    friction=(1.5, 0.5, 0.5),
  )
  body.add_site(name="block_site", pos=(0, 0, 0), size=(0.003,))
  return spec


def _make_container_spec() -> mujoco.MjSpec:
  """Container using the real ``container_*.stl`` meshes (same as the
  original block_picking.xml the project shipped before the mjlab rewrite).

  * ``container_visual.stl`` is the high-detail visual mesh (no collision).
  * ``container_collision.stl`` is the cheap collision proxy (no visual).
  """
  spec = mujoco.MjSpec()
  spec.meshdir = ""  # we pass absolute paths to add_mesh

  spec.add_mesh(name="container_visual", file=str(CONTAINER_VISUAL_STL))
  spec.add_mesh(name="container_collision", file=str(CONTAINER_COLLISION_STL))

  body = spec.worldbody.add_body(name="container")
  body.add_freejoint(name="container_freejoint")

  body.add_geom(
    name="container_vis",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="container_visual",
    rgba=(0.10, 0.80, 0.30, 0.35),
    contype=0,
    conaffinity=0,
    mass=0.0,
  )
  body.add_geom(
    name="container_collision",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="container_collision",
    rgba=(0.0, 0.0, 0.0, 0.0),  # invisible collision proxy
    mass=0.15,
  )
  body.add_site(
    name="container_site",
    pos=(0.0, 0.0, CONTAINER_RIM_HEIGHT * 0.5),
    size=(0.003,),
  )
  return spec


# Height of the task surface in the scene; the SO-101 MJCF anchors its base
# at this Z too, so this value must agree with the robot MJCF.
TABLE_TOP_Z: float = 0.825
TABLE_HALF_SIZE: tuple[float, float, float] = (0.6, 0.275, 0.025)


def _make_table_spec() -> mujoco.MjSpec:
  """Static, fixed-base, non-articulated table — gets replicated per env."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_HALF_SIZE,
    pos=(0.0, 0.0, TABLE_TOP_Z - TABLE_HALF_SIZE[2]),
    rgba=(0.60, 0.50, 0.40, 1.0),
  )
  return spec


def get_table_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    spec_fn=_make_table_spec,
  )


# Workspace defaults. The SO-101 base sits at world (-0.4, 0.25, 0.825) and
# the arm is rotated -90° about z, so its workspace extends in the world -y
# direction (toward smaller y) from x ≈ -0.4. The pre-mjlab env sampled
# block.x ∈ [-0.50, -0.30] and block.y ∈ [0.00, 0.15]; container.x ∈
# [-0.45, -0.30] and container.y ∈ [-0.20, -0.05]. Default positions below
# are the midpoints of those ranges so randomization deltas in tasks/
# block_picking.py stay inside the original reachable region.
def get_block_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(-0.40, 0.075, TABLE_TOP_Z + BLOCK_HALF_SIZE + 0.001),
    ),
    spec_fn=_make_block_spec,
  )


def get_container_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(-0.375, -0.125, TABLE_TOP_Z + 0.001),
    ),
    spec_fn=_make_container_spec,
  )
