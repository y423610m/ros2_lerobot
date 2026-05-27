"""
SO‑101 block‑picking Gymnasium environment.

A clean, minimal env that wraps the existing MuJoCo scene
(`mjlab_rl/mujoco_models/block_picking.xml`). Used by the scripted oracle,
behavior cloning, and policy distillation pipelines.

Observation (np.float32, 25-dim):
    joint_pos (6) | joint_vel (6) | ee_pos (3) | block_pos (3) |
    block_quat (4) | container_pos (3)

Action (np.float32, 6-dim, [-1, 1]):
    Normalized position targets for shoulder_pan, shoulder_lift, elbow_flex,
    wrist_flex, wrist_roll, gripper.

Success: block xy distance to container site < 0.04 m AND block height
within container interior AND gripper not touching block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv


# ---------------------------------------------------------------------------
# Geometry constants (must agree with mujoco_models/block_picking.xml).
# ---------------------------------------------------------------------------
TABLE_Z = 0.825
CONTAINER_RIM_HEIGHT = 0.046
CONTAINER_INTERIOR_HALF_WIDTH = 0.030
SUCCESS_XY_TOL = CONTAINER_INTERIOR_HALF_WIDTH + 0.005
SUCCESS_Z_MAX = CONTAINER_RIM_HEIGHT + 0.02

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

INIT_QPOS = np.array([0.0, -0.3, 0.6, 0.3, 0.0, 0.02], dtype=np.float64)

DEFAULT_XML_PATH = (
    Path(__file__).resolve().parents[2] / "mujoco_models" / "block_picking.xml"
)


class BlockPickingEnv(MujocoEnv):
    """SO‑101 block‑picking MuJoCo env."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    OBS_DIM = 6 + 6 + 3 + 3 + 4 + 3   # = 25
    ACT_DIM = 6

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 400,
        random_block_pos: bool = True,
        random_container_pos: bool = True,
        xml_path: str | Path = DEFAULT_XML_PATH,
        seed: int | None = None,
    ) -> None:
        self._max_episode_steps = int(max_episode_steps)
        self._random_block_pos = bool(random_block_pos)
        self._random_container_pos = bool(random_container_pos)
        self._step_count = 0
        self._success_count = 0
        self._episode_count = 0

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

        super().__init__(
            model_path=str(Path(xml_path).resolve()),
            frame_skip=2,                  # 2 * 10 ms = 20 ms / 50 Hz control
            observation_space=observation_space,
            render_mode=render_mode,
        )
        # gymnasium’s MujocoEnv builds an action_space from actuator ctrlrange,
        # but we explicitly want normalized [-1, 1] (so BC targets stay scaled).
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.ACT_DIM,), dtype=np.float32
        )

        self._cache_ids()
        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Mujoco bookkeeping
    # ------------------------------------------------------------------
    def _cache_ids(self) -> None:
        m = self.model
        self._joint_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES
        ]
        self._jnt_qposadr = np.array(
            [m.jnt_qposadr[j] for j in self._joint_ids], dtype=np.int64
        )
        self._jnt_dofadr = np.array(
            [m.jnt_dofadr[j] for j in self._joint_ids], dtype=np.int64
        )
        self._jnt_low = m.jnt_range[self._joint_ids, 0].astype(np.float64)
        self._jnt_high = m.jnt_range[self._joint_ids, 1].astype(np.float64)

        self._ctrl_low = m.actuator_ctrlrange[:, 0].astype(np.float64)
        self._ctrl_high = m.actuator_ctrlrange[:, 1].astype(np.float64)

        # Use the fixed flange frame ("gripperframe") as the IK reference.
        # gripperframe is rigidly bolted to the flange, so its world pose
        # depends only on the arm joints (1-5), not on the gripper opening.
        # We expose the actual *grasp point* (ee_pos) as
        #     ee_pos = gripperframe + R_gripperframe @ _grasp_local_offset
        # where _grasp_local_offset is a constant offset, calibrated once from
        # the URDF geometry of the FIXED finger (it never moves with the
        # gripper DoF), then rotated into the world by the gripperframe's
        # orientation each step. This way:
        #   * IK stays anchored to a rigid frame (good conditioning), and
        #   * the reported EE position is where the jaws actually grasp,
        #     so reward, success checks, and oracle distances are correct.
        self._ee_sid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self._block_sid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "block_site"
        )
        self._container_sid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "container_site"
        )

        blk_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
        self._blk_qpos_adr = m.jnt_qposadr[blk_jid]
        self._blk_qvel_adr = m.jnt_dofadr[blk_jid]

        self._block_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "block"
        )
        # The container body has a freejoint? In the existing scene it has no
        # joint (it's static-ish but pushable in the original env). Just track
        # its body id for jacobian-free reads.
        self._container_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "container"
        )

        # Calibrate a fixed (gripperframe-local) offset to the grasp center.
        # The grasp center is approximated by the inner face of the FIXED
        # finger geom, biased slightly toward the moving jaw so the block sits
        # centered between jaws when closed. This is computed once at startup
        # using a forward pass; the moving jaw site is not used at runtime.
        self._grasp_local_offset = self._calibrate_grasp_offset()

    def _calibrate_grasp_offset(self) -> np.ndarray:
        """One-time calibration: compute a CONSTANT offset (in gripperframe
        local coords) from the gripperframe origin to the grasp center.

        Uses geom positions, not site positions, so we never need ee_gripper /
        ee_wrist at runtime. The fixed-finger inner geom is rigidly attached
        to the flange, so its local offset is constant.
        """
        m = self.model
        d = self.data
        try:
            gid_fixed = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_GEOM, "gripper_finger_collision_inner"
            )
        except Exception:
            gid_fixed = -1
        # Forward pass to populate geom_xpos / site_xmat.
        mujoco.mj_forward(m, d)
        gf_pos = d.site_xpos[self._ee_sid].copy()
        gf_mat = d.site_xmat[self._ee_sid].reshape(3, 3).copy()
        if gid_fixed < 0:
            # Fallback: zero offset (gripperframe == grasp point).
            return np.zeros(3, dtype=np.float64)
        fixed_world = d.geom_xpos[gid_fixed].copy()
        fixed_local = gf_mat.T @ (fixed_world - gf_pos)
        # Bias by a few mm along local +Z so the IK aims at the BETWEEN-jaws
        # midline rather than at the fixed-finger surface itself.
        fixed_local[2] += 0.008
        return fixed_local.astype(np.float64)

    # ------------------------------------------------------------------
    # Helpers (used by the oracle as well)
    # ------------------------------------------------------------------
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self._jnt_qposadr].copy()

    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self._jnt_dofadr].copy()

    def ee_pos(self) -> np.ndarray:
        """End-effector position == gripperframe (flange-fixed) site.

        We deliberately use the rigidly-mounted gripperframe site rather than
        averaging the two finger sites (ee_gripper / ee_wrist). The SO101
        gripper has a single moving jaw, so a finger-midpoint EE would shift
        whenever the gripper opens or closes, which destabilises IK and makes
        success / distance metrics dependent on the gripper DoF. The
        flange-fixed gripperframe gives a clean, configuration-independent
        anchor for the inverse-kinematics chain.
        """
        return self.data.site_xpos[self._ee_sid].copy()

    def gripperframe_pos(self) -> np.ndarray:
        """Alias for ee_pos -- the flange-fixed gripperframe site position."""
        return self.data.site_xpos[self._ee_sid].copy()

    def grasp_offset_world(self) -> np.ndarray:
        """Rotated offset from gripperframe to the calibrated grasp center,
        expressed in world coords. Useful for the scripted oracle to bias
        its target slightly without taking a hard dependency on the moving
        ee_gripper / ee_wrist sites at runtime."""
        R = self.data.site_xmat[self._ee_sid].reshape(3, 3)
        return R @ self._grasp_local_offset

    def block_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._block_sid].copy()

    def block_quat(self) -> np.ndarray:
        q = self.data.qpos[self._blk_qpos_adr + 3 : self._blk_qpos_adr + 7].copy()
        return q.astype(np.float64)

    def container_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._container_sid].copy()

    # Normalize / unnormalize joint position targets (action <-> ctrl).
    def normalize_joint(self, q: np.ndarray) -> np.ndarray:
        return 2.0 * (q - self._jnt_low) / (self._jnt_high - self._jnt_low) - 1.0

    def unnormalize_action(self, a: np.ndarray) -> np.ndarray:
        return self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset_model(self) -> np.ndarray:
        self._step_count = 0
        self._episode_count += 1

        self.data.qvel[:] = 0
        self.data.qpos[self._jnt_qposadr] = INIT_QPOS

        if self._random_block_pos:
            bx = float(self.np_random.uniform(-0.50, -0.30))
            by = float(self.np_random.uniform(0.00, 0.15))
        else:
            bx, by = -0.40, 0.05
        bz = TABLE_Z + 0.025
        self.data.qpos[self._blk_qpos_adr : self._blk_qpos_adr + 3] = [bx, by, bz]
        self.data.qpos[self._blk_qpos_adr + 3 : self._blk_qpos_adr + 7] = [1, 0, 0, 0]

        if self._random_container_pos:
            cx = float(self.np_random.uniform(-0.45, -0.30))
            cy = float(self.np_random.uniform(-0.20, -0.05))
            self.model.body_pos[self._container_body_id] = [cx, cy, TABLE_Z]

        self._prev_action = np.zeros(self.ACT_DIM, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = self.unnormalize_action(action.astype(np.float64))
        self.do_simulation(ctrl, self.frame_skip)
        self._step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        terminated = bool(info["is_success"])
        truncated = self._step_count >= self._max_episode_steps

        if terminated:
            self._success_count += 1

        info["step"] = self._step_count
        info["success_rate"] = self._success_count / max(1, self._episode_count)
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation / reward
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.joint_pos(),
                self.joint_vel(),
                self.ee_pos(),
                self.block_pos(),
                self.block_quat(),
                self.container_pos(),
            ]
        ).astype(np.float32)

    def _gripper_touching_block(self) -> bool:
        try:
            g1 = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM,
                "gripper_finger_collision_inner",
            )
            g2 = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM,
                "moving_jaw_finger_collision_inner",
            )
            blk = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom"
            )
        except Exception:
            return False
        touch_g, touch_j = False, False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if blk in pair and g1 in pair:
                touch_g = True
            if blk in pair and g2 in pair:
                touch_j = True
        return touch_g and touch_j

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, Any]]:
        ee = self.ee_pos()
        blk = self.block_pos()
        tgt = self.container_pos()

        d_ee_blk = float(np.linalg.norm(blk - ee))
        d_blk_tgt_xy = float(np.linalg.norm(blk[:2] - tgt[:2]))
        block_h = float(blk[2] - TABLE_Z)

        in_container = (
            d_blk_tgt_xy < SUCCESS_XY_TOL
            and -0.005 < block_h < SUCCESS_Z_MAX
        )
        grasping = self._gripper_touching_block()
        released = not grasping and d_ee_blk > 0.04

        is_success = bool(in_container and released)

        # Dense shaping kept simple — BC doesn't need this, but it's useful for
        # eval and any RL fine-tune.
        reward = (
            -d_ee_blk
            - 0.5 * d_blk_tgt_xy
            + 1.0 * float(grasping)
            + 5.0 * float(in_container)
            + 50.0 * float(is_success)
            - 0.01 * float(np.sum(np.square(action - self._prev_action)))
        )
        info = dict(
            d_ee_blk=d_ee_blk,
            d_blk_tgt_xy=d_blk_tgt_xy,
            block_h=block_h,
            is_grasping=bool(grasping),
            is_in_container=bool(in_container),
            is_released=bool(released),
            is_success=is_success,
        )
        return float(reward), info

    @property
    def success_rate(self) -> float:
        return self._success_count / max(1, self._episode_count)
