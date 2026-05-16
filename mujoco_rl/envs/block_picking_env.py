"""
Block Picking Environment
=========================
MuJoCo/Gymnasium environment for training a SO-101 robot (6 DOF) to pick a
block from a table and place it on a target zone.

All 6 joints are treated uniformly:
  shoulder_pan  - base yaw       (hinge, Z)
  shoulder_lift - shoulder pitch (hinge, Y)
  elbow_flex    - elbow pitch    (hinge, Y)
  wrist_flex    - wrist pitch    (hinge, Y)
  wrist_roll    - wrist roll     (hinge, X)
  gripper       - gripper finger (slide, Y)

Observation space (25-dim):
  joint_pos        (6)   joint positions
  joint_vel        (6)   joint velocities
  ee_pos           (3)   end-effector position [m]
#   ee_quat          (4)   end-effector orientation (quaternion)
  block_pos        (3)   block position [m]
  block_vel        (3)   block linear velocity [m/s]
  ee_to_block      (3)   vector from EE to block [m]
  block_to_target  (1)   distance from block to target [m]

Action space (6-dim, continuous [-1, 1]):
  [0:6]  torque/force commands for shoulder_pan..gripper
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

XML_PATH = Path(__file__).parent.parent / "mujoco_models" / "block_picking.xml"

TARGET_POS = np.array([-0.4, -0.1, 0.84], dtype=np.float64)
TABLE_Z = 0.825
LIFT_THRESHOLD = 0.05
PLACE_THRESHOLD = 0.04
MAX_FINGER_SLIDE = 0.040

REWARD_WEIGHTS: dict[str, float] = {
    "reach":        1.0,
    "grasp":        3.0,
    "lift":         5.0,
    "transport":    8.0,
    "release":     10.0,
    "place":       10.0,
    "success":     500.0,
    "alive_penalty":-0.5,
    "ctrl_penalty": -0.1,
}


class BlockPickingEnv(MujocoEnv):
    """
    SO-101 block picking task with 6 uniform DOF.

    The robot must reach the block, grasp it via the gripper joint,
    lift it, transport it to the target zone, and release.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        random_block_pos: bool = True,
        reward_type: str = "dense",
    ):
        self._max_episode_steps = max_episode_steps
        self._random_block_pos = random_block_pos
        self._reward_type = reward_type
        self._step_count = 0
        self._success_count = 0
        self._episode_count = 0

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        super().__init__(
            model_path=str(XML_PATH),
            frame_skip=2,          # 2 * 10ms = 20ms per step -> 50 fps
            observation_space=observation_space,
            render_mode=render_mode,
        )

        self._cache_ids()

    def _cache_ids(self) -> None:
        m = self.model
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self._joint_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        # ee_pos
        self._ee_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._ee_gripper_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_gripper")
        self._ee_wrist_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_wrist")
        self._ee_wrist_base_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_wrist_base")
        self._block_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "block_site")

        blk_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
        self._blk_qpos_adr = m.jnt_qposadr[blk_jid]
        self._blk_qvel_adr = m.jnt_dofadr[blk_jid]

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset_model(self) -> np.ndarray:
        self._step_count = 0
        self._episode_count += 1

        self.data.qvel[:] = 0

        # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        init_qpos = np.array([0.0, -0.3, 0.6, 0.3, 0.0, 0.02])
        for i, jid in enumerate(self._joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = init_qpos[i]

        if self._random_block_pos:
            bx = self.np_random.uniform(-0.5, -0.3)
            # keep block on the +y side of the target (target_y = -0.1) so it
            # never spawns already inside PLACE_THRESHOLD. upper bound 0.15
            # stays inside the robot's reachable range.
            by = self.np_random.uniform(0.0, 0.15)
        else:
            bx, by = 0.30, 0.00
        bz = TABLE_Z + 0.025
        self.data.qpos[self._blk_qpos_adr:self._blk_qpos_adr + 3] = [bx, by, bz]
        self.data.qpos[self._blk_qpos_adr + 3:self._blk_qpos_adr + 7] = [1, 0, 0, 0]

        self.prev_action = np.zeros(6, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def normalize_joint_values(self, raw_joint_pos: np.ndarray):
        jnt_low  = np.array([self.model.jnt_range[jid, 0] for jid in self._joint_ids])
        jnt_high = np.array([self.model.jnt_range[jid, 1] for jid in self._joint_ids])
        normalized_joint_values = 2.0 * (raw_joint_pos - jnt_low) / (jnt_high - jnt_low) - 1.0
        return normalized_joint_values

    def unnormalize_joint_values(self, normalized_joint_values: np.ndarray):
        ctrl_low  = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        return ctrl_low + (normalized_joint_values + 1.0) * 0.5 * (ctrl_high - ctrl_low)

    def step(self, action: np.ndarray):
        self.do_simulation(action, self.frame_skip)
        self._step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        terminated = bool(info["is_success"])
        truncated  = self._step_count >= self._max_episode_steps

        if terminated:
            self._success_count += 1

        info["step"]         = self._step_count
        info["success_rate"] = self._success_count / max(1, self._episode_count)
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        d, m = self.data, self.model

        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids])
        joint_vel = np.array([d.qvel[m.jnt_dofadr[jid]]  for jid in self._joint_ids])

        # ee_pos  = d.site_xpos[self._ee_sid].copy()
        ee_gripper = d.site_xpos[self._ee_gripper_sid].copy()
        ee_wrist = d.site_xpos[self._ee_wrist_sid].copy()
        ee_pos = (ee_gripper + ee_wrist) / 2.0
        # ee_xmat = d.site_xmat[self._ee_sid].reshape(3, 3)
        # ee_quat = np.zeros(4)
        # mujoco.mju_mat2Quat(ee_quat, ee_xmat.flatten())

        block_pos = d.site_xpos[self._block_sid].copy()
        block_vel = d.qvel[self._blk_qvel_adr:self._blk_qvel_adr + 3].copy()

        ee_to_block     = block_pos - ee_pos
        block_to_target = np.array([np.linalg.norm(TARGET_POS - block_pos)])

        return np.concatenate([
            joint_pos,        # 6
            joint_vel,        # 6
            ee_pos,           # 3
            # ee_quat,          # 4
            block_pos,        # 3
            block_vel,        # 3
            ee_to_block,      # 3
            block_to_target,  # 1
        ]).astype(np.float32)

    def _has_contact(self, geom_a_name: str, geom_b_name: str) -> bool:
        """
        Return True if geom_a and geom_b are in contact.
        """
        geom_a = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_a_name,
        )
        geom_b = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_b_name,
        )

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            g1 = contact.geom1
            g2 = contact.geom2

            if (
                (g1 == geom_a and g2 == geom_b)
                or
                (g1 == geom_b and g2 == geom_a)
            ):
                return True

        return False

    def _point_to_line_distance(self, A, B, C):
        '''
        distance between line AB and point C
        '''
        AB = B - A
        AC = C - A

        # Distance = |AB x AC| / |AB|
        distance = np.linalg.norm(np.cross(AB, AC)) / np.linalg.norm(AB)
        return distance

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        # ee_pos    = self.data.site_xpos[self._ee_sid].copy()
        ee_gripper = self.data.site_xpos[self._ee_gripper_sid].copy()
        ee_wrist = self.data.site_xpos[self._ee_wrist_sid].copy()
        ee_wrist_base = self.data.site_xpos[self._ee_wrist_base_sid].copy()
        ee_pos = (ee_gripper + ee_wrist) / 2.0
        block_pos = self.data.site_xpos[self._block_sid].copy()

        d_ee_block     = float(np.linalg.norm(block_pos - ee_pos))
        d_block_target_xy = float(np.linalg.norm(block_pos[:2] - TARGET_POS[:2]))
        block_height   = float(block_pos[2] - TABLE_Z)

        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids])
        normalized_gripper_pos = self.normalize_joint_values(joint_pos)[5]

        # is_grasping = d_ee_block < 0.02 and normalized_gripper_pos < 0.010
        is_gripper_touching = self._has_contact("gripper_finger_collision_inner", "block_geom")
        is_finger_touching = self._has_contact("moving_jaw_finger_collision_inner", "block_geom")
        is_grasping = is_gripper_touching and is_finger_touching and self._point_to_line_distance(ee_wrist, ee_gripper, block_pos) < 0.01
        # is_lifted is height-only so transport/lift shaping doesn't collapse
        # the instant the gripper opens above the target.
        is_lifted   = block_height > LIFT_THRESHOLD
        is_above_target = d_block_target_xy < PLACE_THRESHOLD
        # Widened placement height window from 1cm to 3cm for robustness.
        is_success  =  is_above_target and (block_height < LIFT_THRESHOLD + 0.03) and not is_gripper_touching and not is_finger_touching
        # print(f"{is_gripper_touching=} {is_finger_touching=}  {normalized_gripper_pos=} {is_grasping=} {d_ee_block=} {block_height=} {is_above_target=} {LIFT_THRESHOLD-block_height}")
        # print(f"{block_height-LIFT_THRESHOLD}")
        if self._reward_type == "sparse":
            reward = 1.0 if is_success else 0.0
            info: dict = {}
        else:
            r_reach     = (0.3 * (-d_ee_block) + np.exp(-20 * d_ee_block)) * REWARD_WEIGHTS["reach"]
            r_touch_gripper     = float(is_gripper_touching) * 0.5
            r_touch_finger     = float(is_finger_touching) * 0.5
            r_touch     = float(is_gripper_touching) * float(is_finger_touching) * 1.0
            # Zero out holding rewards once above target so they don't
            # outcompete the release incentive over a long episode.
            r_grasp     = float(is_grasping) * float(not is_above_target) * (1.0 - normalized_gripper_pos) * REWARD_WEIGHTS["grasp"]
            r_lift      = float(is_lifted) * float(not is_above_target) * REWARD_WEIGHTS["lift"]
            r_transport = (0.3 * (-d_block_target_xy) + np.exp(-20 * d_block_target_xy)) * REWARD_WEIGHTS["transport"] if (is_lifted and not is_above_target) else 0.0
            # Once above target XY, shape descent toward TARGET_POS in 3D.
            d_block_target_3d = float(np.linalg.norm(block_pos - TARGET_POS))
            r_descend   = 0.0 # float(is_above_target) * np.exp(-20 * d_block_target_3d) * REWARD_WEIGHTS["transport"]
            r_release   = float(is_above_target) * (normalized_gripper_pos + 1.0) * REWARD_WEIGHTS["release"]
            r_success   = float(is_success) * REWARD_WEIGHTS["success"]
            r_alive     = REWARD_WEIGHTS["alive_penalty"]
            # Only penalize action smoothness (action - prev_action); comparing
            # action [-1,1] to joint_pos [rad] was a scale mismatch.
            r_ctrl      = float(np.linalg.norm(action - self.prev_action)) * REWARD_WEIGHTS["ctrl_penalty"]

            reward = r_reach + r_touch_gripper + r_touch_finger + r_touch + r_grasp + r_lift + r_transport + r_descend + r_release + r_success + r_alive + r_ctrl
            info = {
                "r_reach":     r_reach,
                "r_touch_gripper": r_touch_gripper,
                "r_touch_finger": r_touch_finger,
                "r_touch": r_touch,
                "r_grasp":     r_grasp,
                "r_lift":      r_lift,
                "r_transport": r_transport,
                "r_descend":   r_descend,
                "r_release":   r_release,
                "r_ctrl":      r_ctrl,
            }

        info.update({
            "ee_to_block":     d_ee_block,
            "block_to_target": d_block_target_xy,
            "block_height":    block_height,
            "normalized_gripper_pos":     float(normalized_gripper_pos),
            "is_grasping":     bool(is_grasping),
            "is_lifted":       bool(is_lifted),
            "is_success":      bool(is_success),
            "is_above_target": bool(is_above_target),
        })
        self.prev_action = action.copy()
        return float(reward), info

    @property
    def success_rate(self) -> float:
        return self._success_count / max(1, self._episode_count)
