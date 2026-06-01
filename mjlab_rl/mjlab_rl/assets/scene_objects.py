"""MjSpec builders for the table, manipulable block, and goal container."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

# Block dimensions (half-extents) and color.
BLOCK_HALF_SIZE: float = 0.015
BLOCK_MASS: float = 0.05

# Container visual mesh (rendered only — collision below uses primitive
# boxes because MuJoCo Warp treats <geom type="mesh"> as the geom's convex
# hull, which seals the concave cup with an invisible lid).
_LEROBOT_MESHES = (
  Path(__file__).resolve().parents[3]
  / "src"
  / "lerobot_robots_description"
  / "meshes"
)
CONTAINER_VISUAL_STL = _LEROBOT_MESHES / "container_visual.stl"
assert CONTAINER_VISUAL_STL.exists(), f"missing {CONTAINER_VISUAL_STL}"

# Container dimensions, measured from container_collision.stl. Match the
# visual mesh so the invisible collision boxes line up with what the user
# sees in the viewer.
#   * Floor at z ≈ 0.000-0.003 (tapered slab)
#   * Full perimeter rim at z ≈ 0.050  ← this is what bounds the cup
#   * Two cylindrical posts on one wall extending up to z ≈ 0.077
#   * Outer wall at 7.6 cm wide, inner cavity 7.0 cm wide → wall ≈ 3 mm
CONTAINER_RIM_HEIGHT: float = 0.050
CONTAINER_POLE_HEIGHT: float = 0.077
CONTAINER_INTERIOR_HALF_WIDTH: float = 0.035
CONTAINER_WALL_THICKNESS: float = 0.003
CONTAINER_FLOOR_THICKNESS: float = 0.003
# Drop-here site z above container floor. Needs to clear the pole tops so
# the gripper can approach from any direction without colliding with them.
CONTAINER_DROP_SITE_Z: float = 0.10


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
  """Container = visual STL (for rendering) + primitive box walls (for
  collision). MuJoCo Warp treats ``<geom type="mesh">`` as the geom's
  convex hull, which seals a concave cup with an invisible lid. We
  therefore use the STL only for what the user sees, and build the cup's
  *collision* shape out of five thin boxes (four walls + one floor) sized
  to match the visual mesh's interior cavity.

  Total collision mass adds up to ~0.15 kg, matching the prior single
  mesh-collision geom.
  """
  spec = mujoco.MjSpec()
  spec.meshdir = ""  # we pass absolute paths to add_mesh
  spec.add_mesh(name="container_visual", file=str(CONTAINER_VISUAL_STL))

  body = spec.worldbody.add_body(name="container")
  body.add_freejoint(name="container_freejoint")

  # Visual only — group 2 by MuJoCo convention. Toggle in the native
  # viewer with keyboard '2'.
  body.add_geom(
    name="container_vis",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="container_visual",
    rgba=(0.10, 0.80, 0.30, 0.35),
    contype=0,
    conaffinity=0,
    mass=0.0,
    group=2,
  )

  # Collision geometry — primitive boxes only, all convex, total ≈ 0.15 kg.
  # Group 3 by MuJoCo convention. Hidden by default; toggle on with
  # keyboard '3' in the native viewer to inspect the collision proxy.
  iw = CONTAINER_INTERIOR_HALF_WIDTH
  wt = CONTAINER_WALL_THICKNESS
  ft = CONTAINER_FLOOR_THICKNESS
  rim_z = CONTAINER_RIM_HEIGHT  # walls span 0 → rim_z above the body origin
  outer = iw + wt               # outer wall half-extent in the long direction
  collision_rgba = (0.95, 0.20, 0.20, 0.35)  # translucent red, only visible
                                              # when group 3 is enabled

  # Thin floor slab covering the inner footprint.
  body.add_geom(
    name="container_floor_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(outer, outer, ft / 2),
    pos=(0.0, 0.0, ft / 2),
    rgba=collision_rgba,
    mass=0.05,
    group=3,
  )
  wall_half_size_xy = (wt / 2, outer, rim_z / 2)  # short side x, long side y
  wall_half_size_yx = (outer, wt / 2, rim_z / 2)
  for name, pos, size in (
    ("container_wall_xp",
     (iw + wt / 2, 0.0, rim_z / 2), wall_half_size_xy),
    ("container_wall_xn",
     (-(iw + wt / 2), 0.0, rim_z / 2), wall_half_size_xy),
    ("container_wall_yp",
     (0.0, iw + wt / 2, rim_z / 2), wall_half_size_yx),
    ("container_wall_yn",
     (0.0, -(iw + wt / 2), rim_z / 2), wall_half_size_yx),
  ):
    body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=pos,
      rgba=collision_rgba,
      mass=0.025,
      group=3,
    )

  # Site above the rim — see geometry comments at the top of the file.
  body.add_site(
    name="container_site",
    pos=(0.0, 0.0, CONTAINER_DROP_SITE_Z),
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
    rgba=(0.1, 0.1, 0.1, 0.9),
  )
  # White backdrop wall along the -y edge of the table. Gives the cameras a
  # clean uniform background behind the workspace (helps sim-to-real
  # contrast for vision policies). Slightly outside the table footprint so
  # block/container can sit flush at the edge if randomization pushes them.
  body.add_geom(
    name="backdrop_wall",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(TABLE_HALF_SIZE[0], 0.01, 0.25),  # 1.2 m wide × 2 cm thick × 60 cm tall
    pos=(0.0, -(TABLE_HALF_SIZE[1] + 0.01), TABLE_TOP_Z + 0.30),
    rgba=(1.0, 1.0, 1.0, 1.0),
  )

  # Three extra directional lights aimed at the workspace centre. Combined
  # with the existing "sun" light, the scene has 4 lights — randomly
  # toggling each on/off per episode gives a wide range of brightness
  # (1 light = dim, 4 lights = bright) and direction (front/back/side
  # shadows). See `mjlab_rl.envs.mdp.randomize_light_active`.
  light_z = TABLE_TOP_Z + 0.80
  for name, lpos, ldir in (
    ("table_light_overhead",  (-0.40,  0.00, light_z), (0.0,  0.0, -1.0)),
    ("table_light_front",     (-0.40, +0.40, light_z), (0.0, -0.4, -1.0)),
    ("table_light_back",      (-0.40, -0.40, light_z), (0.0, +0.4, -1.0)),
  ):
    body.add_light(name=name, pos=lpos, dir=ldir,
                   type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
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
      # 180° about world z (mjlab quat order = w, x, y, z), so the mesh's
      # asymmetric front faces the robot rather than away.
      rot=(0.0, 0.0, 0.0, 1.0),
    ),
    spec_fn=_make_container_spec,
  )
