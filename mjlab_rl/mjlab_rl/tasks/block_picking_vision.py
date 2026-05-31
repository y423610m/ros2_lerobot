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

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from mjlab_rl.tasks.block_picking import (
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


def make_block_picking_vision_env_cfg(play: bool = False):
  cfg = make_block_picking_env_cfg(play=play)

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
    pos=(-0.39, -0.02, 1.30),  # 1.30 m above the table-mocap origin
                                # = ~48 cm above the 0.825 m tabletop
    quat=(1.0, 0.0, 0.0, 0.0),  # identity → camera local -z = world -z
                                # = looks straight down
    **shared,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (wrist_cam, top_cam)

  # --- Observations -----------------------------------------------------
  # Strip privileged distance terms from the actor; it must learn them
  # from pixels.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_block")
  actor_obs.terms.pop("block_to_container")
  # Actor keeps: joint_pos, joint_vel, actions  (18-dim).
  # Critic is untouched and still has the full privileged state.

  # Camera group — concatenated along channel dim (6 = 2 cams × 3 RGB).
  cam_terms = {
    "wrist_rgb": ObservationTermCfg(
      func=manipulation_mdp.camera_rgb,
      params={"sensor_name": "wrist_cam"},
    ),
    "top_rgb": ObservationTermCfg(
      func=manipulation_mdp.camera_rgb,
      params={"sensor_name": "top_cam"},
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


def make_block_picking_vision_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
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
    save_interval=100,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=10_000,  # vision needs more iters
  )


# ----------------------------------------------------------------------------
# Registration.
# ----------------------------------------------------------------------------


register_mjlab_task(
  task_id="Mjlab-SO101-Block-Picking-Rgb",
  env_cfg=make_block_picking_vision_env_cfg(),
  play_env_cfg=make_block_picking_vision_env_cfg(play=True),
  rl_cfg=make_block_picking_vision_ppo_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
