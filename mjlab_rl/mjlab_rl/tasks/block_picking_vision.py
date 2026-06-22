"""Vision variant of the SO-101 block-picking task.

Same env as ``Mjlab-SO101-Block-Picking``, but:

* Two RGB cameras (64×64 each):
    - ``wrist_cam`` wraps the existing ``hand_eye`` camera in the SO-101 MJCF
      (already wrist-mounted, pointing toward the gripper).
    - ``top_cam`` is a new world-fixed top-down camera centred over the
      workspace, attached to the table body so it replicates per-env.
* **Asymmetric observations**:
    - ``actor``  = joint_pos + joint_vel + last_action  (~18-dim) **+** the
      ``camera`` group. No privileged distances.
    - ``critic`` = the unmodified state-only group from the base task
      (includes privileged ``ee_to_block``, ``block_to_container``,
      ``block_pose``).
* **Asymmetric models**: actor is :class:`SpatialSoftmaxCNNModel` (state
  + image CNN), critic is the default MLP, no camera, no CNN.
"""

from __future__ import annotations

import math

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from mjlab_rl.envs import mdp as task_mdp
from mjlab_rl.tasks.block_picking import (
  DR_ORDER,
  make_block_picking_env_cfg,
  make_block_picking_ppo_cfg,
)

# ----------------------------------------------------------------------------
# CNN cfg + class spec — same as YAM lift-cube-vision.
# ----------------------------------------------------------------------------

_VISION_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_CNN_CLASS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


# ----------------------------------------------------------------------------
# Env cfg.
# ----------------------------------------------------------------------------


def make_block_picking_vision_env_cfg(play: bool = False, dr_level: str = "dr3"):
  order = DR_ORDER[dr_level]
  cfg = make_block_picking_env_cfg(play=play, dr_level=dr_level)

  # --- Cameras ----------------------------------------------------------
  shared = dict(
    height=64,
    width=64,
    data_types=("rgb",),
    # Visual-only rendering:
    #   * 0 = default group — block, table.
    #   * 2 = visual class — SO-101 mesh geoms (full jaw shape included)
    #         and the container's STL visual.
    # Group 3 (collision class) is intentionally excluded, so the
    # container's translucent-red collision boxes and the SO-101's
    # invisible finger collision primitives don't appear in the camera
    # frames. The viewer keyboard toggle for group 3 still works as a
    # debug tool.
    enabled_geom_groups=(0, 2),
    use_textures=True,
    # mujoco-warp ray-traces shadows, so they are hard-edged / directional by
    # construction (the OpenGL shadowsize/shadowscale knobs don't apply here).
    # Adds GPU cost per camera per step and changes the policy's input
    # distribution — retrain after flipping this.
    use_shadows=False,
  )
  wrist_cam = CameraSensorCfg(
    name="wrist_cam",
    camera_name="robot/hand_eye",  # wraps the existing MJCF camera
    **shared,
  )
  # New world-fixed top-down camera. Anchored to the table body so the
  # camera replicates per-env (the table is a fixed-base mocap entity
  # whose body sits at the per-env origin). pos is table-local; the
  # workspace centroid (block range mid ≈ (-0.40, 0.075), container range
  # mid ≈ (-0.375, -0.125)) is around (-0.39, -0.02) → use that as the xy.
  top_cam = CameraSensorCfg(
    name="top_cam",
    parent_body="table/table",
    pos=(-0.34, -0.02, 1.30),  # 1.30 m above the table-mocap origin
                                # = ~48 cm above the 0.825 m tabletop
    # Still looks straight down (local -z = world -z), but rotated 90° about
    # the world z axis so the image's vertical axis tracks the table's x
    # axis rather than y. quat = cos(45°), 0, 0, sin(45°).
    quat=(0.7071068, 0.0, 0.0, 0.7071068),
    **shared,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (wrist_cam, top_cam)

  # --- Camera extrinsics randomization ----------------------------------
  # Per-episode jitter of each camera's mount pose so the policy is robust to
  # the real rig not matching the sim extrinsics exactly (a cause of the
  # residual ee-placement error on hardware).
  # Camera-pose jitter ramps across dr2->dr3 (half at dr2, full at dr3; none
  # below dr2). "Full" = reduced ranges (top ±5°/±3 cm, wrist ±3°/±1.5 cm): the
  # original ±15°/±10 cm is a large spatial image shift that collapsed the grasp,
  # and is larger than realistic mounting error anyway. Introducing it at dr2
  # alongside the image realism (rather than alone at dr3) gives a gentler ramp.
  cam_frac = (order - 1) / 2.0 if order >= 2 else 0.0  # dr2 -> 0.5, dr3 -> 1.0
  _WRIST_RPY = (-math.radians(3.0) * cam_frac, math.radians(3.0) * cam_frac)
  _TOP_RPY = (-math.radians(5.0) * cam_frac, math.radians(5.0) * cam_frac)
  _WRIST_UV = 0.015 * cam_frac
  _TOP_POS = 0.03 * cam_frac

  # Wrist cam ("robot/hand_eye"), rigidly bolted to the wrist. Position is
  # jittered ONLY within its image plane (local x/y), NOT along the optical
  # axis (depth) — its optical axis is tilted, so plain cam_pos can't isolate
  # that. Full ±1.5 cm in-plane, ±3° orientation (× cam_frac).
  cfg.events["wrist_cam_pos"] = EventTermCfg(
    func=task_mdp.randomize_cam_pos_in_image_plane,
    mode="reset",
    params={
      "camera_name": "robot/hand_eye",
      "u_range": (-_WRIST_UV, _WRIST_UV),
      "v_range": (-_WRIST_UV, _WRIST_UV),
    },
  )
  cfg.events["wrist_cam_quat"] = EventTermCfg(
    func=dr.cam_quat,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", camera_names=("hand_eye",)),
      "roll_range": _WRIST_RPY,
      "pitch_range": _WRIST_RPY,
      "yaw_range": _WRIST_RPY,
    },
  )

  # Top cam ("top_cam"), free-standing overhead rig. Full ±3 cm position offset
  # (all axes) and ±5° orientation (× cam_frac), reduced from ±10 cm / ±15°.
  cfg.events["top_cam_pos"] = EventTermCfg(
    func=dr.cam_pos,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("table", camera_names=("top_cam",)),
      "ranges": (-_TOP_POS, _TOP_POS),
      "operation": "add",
    },
  )
  cfg.events["top_cam_quat"] = EventTermCfg(
    func=dr.cam_quat,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("table", camera_names=("top_cam",)),
      "roll_range": _TOP_RPY,
      "pitch_range": _TOP_RPY,
      "yaw_range": _TOP_RPY,
    },
  )

  # Camera-pose jitter active from dr2 (ramped via cam_frac). dr0/dr1 keep fixed
  # cameras so the grasp and color are learned against a stable view first.
  if order < 2:
    for _k in ("wrist_cam_pos", "wrist_cam_quat", "top_cam_pos", "top_cam_quat"):
      cfg.events.pop(_k, None)

  # --- Observations -----------------------------------------------------
  # Strip privileged distance terms from the actor; it must learn them
  # from pixels. Also drop joint_vel: lerobot's SOFollower.get_observation
  # only reads Present_Position from the motors, so the real arm doesn't
  # expose a velocity channel. Removing it here keeps the sim actor's
  # input shape consistent with what's available at deploy time.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_block")
  actor_obs.terms.pop("block_to_container")
  actor_obs.terms.pop("joint_vel")
  # Actor keeps: joint_pos, actions  (12-dim).
  # Critic is untouched and still has the full privileged state (incl.
  # joint_vel — fine, the critic only runs in sim).

  # Camera group — concatenated along channel dim (6 = 2 cams × 3 RGB).
  # Image realism (sim2real) at soft/full: motion blur + pixel noise +
  # brightness/contrast via camera_rgb_aug, and camera latency via the obs
  # term's built-in delay buffer. At none the aug params are 0 (identity wrapper
  # of camera_rgb) and there is no delay, matching the sharp clean sim image.
  img_aug = order >= 2
  aug_params = dict(
    blur_strength=2.0 if img_aug else 0.0,  # max Gaussian sigma (px) at full motion
    noise_std=0.02 if img_aug else 0.0,
    brightness=0.10 if img_aug else 0.0,
    contrast=0.10 if img_aug else 0.0,
  )
  cam_lag = 1 if img_aug else 0  # camera latency, control steps (0-20 ms)
  cam_terms = {
    "wrist_rgb": ObservationTermCfg(
      func=task_mdp.camera_rgb_aug,
      params={"sensor_name": "wrist_cam", **aug_params},
      delay_max_lag=cam_lag,
    ),
    "top_rgb": ObservationTermCfg(
      func=task_mdp.camera_rgb_aug,
      params={"sensor_name": "top_cam", **aug_params},
      delay_max_lag=cam_lag,
    ),
  }
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  return cfg


# ----------------------------------------------------------------------------
# RL runner cfg — asymmetric: CNN actor, MLP critic.
# ----------------------------------------------------------------------------


def make_block_picking_vision_ppo_cfg(run_name: str = "") -> RslRlOnPolicyRunnerCfg:
  base = make_block_picking_ppo_cfg()
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name=_CNN_CLASS,
      cnn_cfg=_VISION_CNN_CFG,
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      # Default MLP — no CNN, no camera.
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic",),
    },
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,  # a bit lower than state-only; vision is noisier
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="so101_block_picking_vision",
    # Appended to the run dir (<timestamp>_<run_name>) to tag the DR level while
    # keeping one shared experiment_name so curriculum resumes find prior runs.
    run_name=run_name,
    save_interval=1000,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=10_000,  # vision needs more iters
  )


# ----------------------------------------------------------------------------
# Registration.
# ----------------------------------------------------------------------------


# 4-level curriculum, all sharing one experiment_name ("so101_block_picking_vision")
# so each stage's resume finds the prior run; run_name tags the run dir
# (<timestamp>_dr<level>):
#   DR0 : non-visual DR only (servo lag, encoder bias, wide ±0.3 starts) — vivid
#         pink, fixed cameras, sharp images.
#   DR1 : + block/table color.
#   DR2 : + image motion blur/noise/brightness + camera/proprioception latency
#         + camera-pose jitter @ 1/2 (±2.5°/±1.5 cm).
#   DR3 : + camera-pose jitter @ full (±5°/±3 cm). Jitter ramps 1/2->full dr2-3.
# The bare ``Mjlab-SO101-Block-Picking-Rgb`` is kept as the default/full (= DR3)
# task id for standalone tooling (export, deploy, play-zero, render).
_BLOCK_PICKING_RGB_LEVELS = {
  "Mjlab-SO101-Block-Picking-Rgb": "dr3",
  "Mjlab-SO101-Block-Picking-Rgb-DR0": "dr0",
  "Mjlab-SO101-Block-Picking-Rgb-DR1": "dr1",
  "Mjlab-SO101-Block-Picking-Rgb-DR2": "dr2",
  "Mjlab-SO101-Block-Picking-Rgb-DR3": "dr3",
}
for _task_id, _lvl in _BLOCK_PICKING_RGB_LEVELS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=make_block_picking_vision_env_cfg(dr_level=_lvl),
    play_env_cfg=make_block_picking_vision_env_cfg(play=True, dr_level=_lvl),
    rl_cfg=make_block_picking_vision_ppo_cfg(run_name=_lvl),
    runner_cls=ManipulationOnPolicyRunner,
  )
