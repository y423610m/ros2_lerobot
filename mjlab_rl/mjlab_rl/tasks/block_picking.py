"""Block pick-and-place task: SO-101 arm picks a cube and drops it into a bin.

Manager-based RL env config wired up the same way as the YAM lift-cube task in
``mjlab.tasks.manipulation``, but with custom rewards/terminations defined in
``mjlab_rl.envs.mdp`` (since this is a place-into-container task, not a
lift-to-target-pose task — no ``LiftingCommand`` involved).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import actions as mdp_actions  # noqa: F401  (kept for compat)
from mjlab.envs.mdp import events as mdp_events
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.envs.mdp import rewards as mdp_rewards
from mjlab.envs.mdp import terminations as mdp_term
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_rl.assets import (
  SO101_ACTION_SCALE,
  TABLE_TOP_Z,
  get_block_cfg,
  get_container_cfg,
  get_so101_robot_cfg,
  get_table_cfg,
)
from mjlab_rl.envs import mdp as task_mdp
from mjlab_rl.envs.actions import RateLimitedJointPositionActionCfg

# ----------------------------------------------------------------------------
# Geometry / randomization ranges. All numbers are in meters / radians.
# ----------------------------------------------------------------------------

# XY randomization offsets (added to each entity's default pos by
# ``reset_root_state_uniform`` — these are *deltas*, not absolute positions).
# Together with the defaults set in scene_objects.py these reproduce the
# original pre-mjlab env's sample regions:
#   block.x ∈ [-0.50, -0.30],  block.y ∈ [0.00, 0.15]
#   container.x ∈ [-0.45, -0.30], container.y ∈ [-0.20, -0.05]
# The whole region sits within the SO-101's ~0.35 m reach from base
# (-0.4, 0.25, 0.825), with the arm pointing in world -y.
BLOCK_XY_RANGE = {
  "x": (-0.10, 0.10),  # default x = -0.40 → [-0.50, -0.30]
  "y": (-0.075, 0.075),  # default y = 0.075 → [0.00, 0.15]
}
CONTAINER_XY_RANGE = {
  "x": (-0.075, 0.075),  # default x = -0.375 → [-0.45, -0.30]
  "y": (-0.075, 0.075),  # default y = -0.125 → [-0.20, -0.05]
}


# ----------------------------------------------------------------------------
# Environment configuration.
# ----------------------------------------------------------------------------


def make_block_picking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  robot_cfg = SceneEntityCfg("robot", site_names=("gripperframe",))

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp_obs.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp_obs.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "ee_to_block": ObservationTermCfg(
      func=task_mdp.ee_to_block,
      params={"block_name": "block", "asset_cfg": robot_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "block_to_container": ObservationTermCfg(
      func=task_mdp.block_to_container,
      params={"block_name": "block", "container_name": "container"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "actions": ObservationTermCfg(func=mdp_obs.last_action),
  }
  critic_terms = {
    **{k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor_terms.items()},
    "block_pose": ObservationTermCfg(
      func=task_mdp.block_pose,
      params={"block_name": "block"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": RateLimitedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=SO101_ACTION_SCALE,
      use_default_offset=True,
      # Per-step (env-step = 100 ms at decimation=20, timestep=0.005) max
      # delta from the present joint position. Mirrors lerobot's SOFollower
      # send_action safety clamp. 0.5 rad/step ≈ 5 rad/s, in line with what
      # the real STS-3215 servos can comfortably do.
      max_relative_target={
        "shoulder_pan": 0.5,
        "shoulder_lift": 0.5,
        "elbow_flex": 0.5,
        "wrist_flex": 0.5,
        "wrist_roll": 0.5,
        "gripper": 1.5,  # gripper snaps faster
      },
    ),
  }

  events = {
    # Fixed-base mocap entities won't follow env_origins without an explicit
    # reset event — they otherwise stack on top of each other at world (0,0,0).
    "reset_table_pose": EventTermCfg(
      func=mdp_events.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("table"),
      },
    ),
    "reset_robot_base": EventTermCfg(
      func=mdp_events.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp_events.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_block_pose": EventTermCfg(
      func=mdp_events.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": BLOCK_XY_RANGE,
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("block"),
      },
    ),
    "reset_container_pose": EventTermCfg(
      func=mdp_events.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": CONTAINER_XY_RANGE,
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("container"),
      },
    ),
    # Must run AFTER reset_container_pose so the snapshot captures the
    # randomized position, not the un-randomized default.
    "snapshot_container_xy": EventTermCfg(
      func=task_mdp.snapshot_container_xy,
      mode="reset",
      params={"container_name": "container"},
    ),
    # Randomize which scene lights are active per env — gives bright/dim/
    # angled-shadow variation each episode. 4 lights × Bernoulli(0.6) →
    # ~2.4 on average. ensure_one_on prevents pitch-black scenes.
    "randomize_lights": EventTermCfg(
      func=task_mdp.randomize_light_active,
      mode="reset",
      params={"p_on": 0.6, "ensure_one_on": True},
    ),
  }

  rewards = {
    "reach": RewardTermCfg(
      func=task_mdp.reach_block_reward,
      weight=1.0,
      params={"std": 0.10, "asset_cfg": robot_cfg, "block_name": "block"},
    ),
    "lift": RewardTermCfg(
      func=task_mdp.lift_block_reward,
      weight=2.0,
      params={"max_lift": 0.10, "block_name": "block"},
    ),
    "place": RewardTermCfg(
      func=task_mdp.place_block_reward,
      weight=3.0,
      params={
        "std": 0.10,
        "block_name": "block",
        "container_name": "container",
      },
    ),
    # Behavioral bonus: rewards the act of opening the gripper while the
    # block is positioned over the container. Pays even if the block then
    # misses the cup. This is the missing causal signal — without it the
    # policy never samples a release at the right moment, so it never
    # discovers the downstream deposit/success rewards.
    "gripper_open_above_cup": RewardTermCfg(
      func=task_mdp.gripper_open_above_cup_bonus,
      weight=5.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=("gripper",)),
        "block_name": "block",
        "container_name": "container",
      },
    ),
    # Smooth shaping for "block has cleared the rim and is settling into
    # the cup". Earnable only by releasing the block above the container —
    # the closed gripper is wider than the cup interior, so this region is
    # physically unreachable while held. Bridges the gradient gap that was
    # leaving the policy stuck at a "hover-and-hold" local optimum.
    "deposit": RewardTermCfg(
      func=task_mdp.block_deposit_reward,
      weight=6.0,
      params={
        "block_name": "block",
        "container_name": "container",
      },
    ),
    "success": RewardTermCfg(
      func=task_mdp.success_bonus,
      weight=15.0,
      params={
        "xy_tol": 0.030,         # 3cm tolerance for the 4.5×2×2 cm block
        "z_max_above_floor": 0.055,  # rim is at 5cm above container body origin
        "block_name": "block",
        "container_name": "container",
      },
    ),
    # Pulls the arm back to the home pose AFTER the block is deposited. Zero
    # before deposit, so doesn't interfere with reach/lift/place shaping.
    # Stops the policy from wandering during the post-success steps.
    "home_pose_after_success": RewardTermCfg(
      func=task_mdp.post_success_home_pose_reward,
      weight=10.0,
      params={
        "cutoff": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "block_name": "block",
        "container_name": "container",
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp_rewards.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp_rewards.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Discourage knocking the container around. Weight is -10 on squared
    # XY displacement: a 5 cm push costs 0.025 reward; a 15 cm push costs
    # 0.225 — comparable to the place reward's peak, so dragging the
    # container toward the block is not a winning strategy.
    "container_displacement": RewardTermCfg(
      func=task_mdp.container_xy_displacement_sq,
      weight=-10.0,
      params={"container_name": "container"},
    ),
    # Discourage spinning the container in place. Metric is 0 at no
    # rotation, 0.5 at 90°, 1 at 180°. weight=-2 makes a 90° rotation cost
    # -1.0 per step (comparable to the lift reward's peak) so the policy
    # has a strong reason not to use the container as a fidget toy.
    "container_rotation": RewardTermCfg(
      func=task_mdp.container_rotation_metric,
      weight=-2.0,
      params={"container_name": "container"},
    ),
    # Penalise any gripper-on-table contact. Sensor primary = gripper subtree
    # (catches both the wrist body and the moving-jaw child body), secondary
    # = table_top geom. Fires 1.0/step while in contact.
    "gripper_table_contact": RewardTermCfg(
      func=task_mdp.sensor_contact_penalty,
      weight=-5.0,
      params={"sensor_name": "gripper_table_contact"},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp_term.time_out, time_out=True),
    "block_dropped": TerminationTermCfg(
      func=task_mdp.block_dropped,
      params={"min_z": TABLE_TOP_Z - 0.05, "block_name": "block"},
    ),
    # Without this, an occasional MuJoCo Warp solver blow-up (one env in
    # thousands, usually during contact-heavy gripper-on-mesh interactions)
    # produces a NaN in qpos/qvel → NaN in observations → rsl-rl aborts
    # training. Listing it as a termination resets just the affected env
    # before the post-step observation is computed.
    "nan": TerminationTermCfg(func=mdp_term.nan_detection),
  }

  gripper_table_contact_cfg = ContactSensorCfg(
    name="gripper_table_contact",
    primary=ContactMatch(mode="subtree", pattern="gripper", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="table_top", entity="table"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    history_length=1,
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=2.0,
      entities={
        "table": get_table_cfg(),
        "robot": get_so101_robot_cfg(),
        "block": get_block_cfg(),
        "container": get_container_cfg(),
      },
      sensors=(gripper_table_contact_cfg,),
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=1.2,
      elevation=-20.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=20,  # 20 * 5 ms physics step = 100 ms env step = 10 Hz control
    episode_length_s=8.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


# ----------------------------------------------------------------------------
# RL runner configuration.
# ----------------------------------------------------------------------------


def make_block_picking_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
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
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.02,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="so101_block_picking",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


# ----------------------------------------------------------------------------
# Registration.
# ----------------------------------------------------------------------------


register_mjlab_task(
  task_id="Mjlab-SO101-Block-Picking",
  env_cfg=make_block_picking_env_cfg(),
  play_env_cfg=make_block_picking_env_cfg(play=True),
  rl_cfg=make_block_picking_ppo_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
