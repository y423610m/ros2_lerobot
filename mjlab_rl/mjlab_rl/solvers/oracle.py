"""
Scripted oracle for SO-101 block pick-and-place.

Design rationale
================
The SO-101 has a 1-DoF asymmetric gripper (one fixed finger, one moving jaw)
and ±3.35 N·m position actuators with kp=17.8. Three things kill placement
reliability if not handled carefully:

  1. The asymmetric jaw lets the block slip out at low contact force, so
     any high-acceleration carry motion (rad/s² across the wrist) flings the
     block. Carry trajectory must be RATE-LIMITED in joint space.

  2. IK is solved against the rigid `gripperframe` site (flange-fixed) per
     user instruction. We expose `ee_pos() == gripperframe`. The gripperframe
     sits ~3 cm ABOVE the actual grasp center between the fingertips when
     the gripper is closed; the geometry is constant in gripperframe-local
     coords. We precompute that local offset once at startup and use it to
     pick IK targets that put the *grasp center* on the block.

  3. The container interior tolerance is 30 mm, but the EE drops ~12 mm
     from the commanded XY position when the arm is extended and motors
     saturate. We therefore aim TRANSFER and DESCEND_PLACE at the
     *container center XY*, hold long enough for the EE to settle in the
     sag equilibrium, then release the block straight down into the cup.

Phase machine
=============
APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSFER -> DESCEND_PLACE ->
RELEASE -> SETTLE -> RETREAT -> HOME -> DONE.

`SETTLE` is a no-IK wait phase that lets the block come to rest inside the
container before any retreat motion drags it out.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import mujoco
import numpy as np

from mjlab_rl.envs.block_picking_env import (
    BlockPickingEnv,
    TABLE_Z,
    CONTAINER_RIM_HEIGHT,
    INIT_QPOS,
)


class Phase(IntEnum):
    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    TRANSFER = 4
    DESCEND_PLACE = 5
    RELEASE = 6
    SETTLE = 7
    RETREAT = 8
    HOME = 9
    DONE = 10


# ---- geometry constants ----
# Heights are absolute world Z (TABLE_Z = 0.825, CONTAINER_RIM_HEIGHT = 0.046).
APPROACH_CLEARANCE_Z = TABLE_Z + 0.085     # block-center reference + 6cm
GRASP_Z              = TABLE_Z + 0.025     # block-center height (block half=0.015 + 0.01 margin)
LIFT_Z               = TABLE_Z + 0.090     # well above table top
TRANSFER_Z           = TABLE_Z + 0.090     # same as LIFT - no z motion in transit
PLACE_Z              = TABLE_Z + 0.060     # ~4cm above container rim
RELEASE_LOWER_Z      = TABLE_Z + 0.075     # let arm sag here before opening

# Gripper normalized commands (action space is normalized in [-1, 1]).
GRIPPER_WIDE_OPEN    =  0.80   # APPROACH and after-release
GRIPPER_PRE_GRASP    =  0.15   # snug-open right before close
GRIPPER_CLOSED       = -0.92   # close to limit: q~-0.10 (block dim 30mm, finger compliance squeezes)
GRIPPER_RELEASE      =  0.50   # open enough to let block drop

# Tolerances.
XY_TOL_APPROACH = 0.020
XY_TOL_DESCEND  = 0.010
XY_TOL_PLACE    = 0.020
Z_TOL           = 0.010

# Per-phase max joint target delta per step. Empirically the asymmetric
# jaw cannot hold the block at arm Cartesian speeds above ~0.3 m/s, which
# is ~0.05 rad/step on the longest moment-arm joint at 50 Hz control.
ARM_CLIP_FAST = 0.20    # APPROACH / DESCEND / HOME -- arm is free
ARM_CLIP_HOLD = 0.020    # LIFT / TRANSFER / DESCEND_PLACE -- holding block
GRIP_CLIP     = 0.08

# Phase timeouts (in env steps).
TIMEOUT = {
    Phase.APPROACH:      120,
    Phase.DESCEND:        80,
    Phase.GRASP:          60,
    Phase.LIFT:          200,
    Phase.TRANSFER:      400,
    Phase.DESCEND_PLACE: 200,
    Phase.RELEASE:        30,
    Phase.SETTLE:         60,
    Phase.RETREAT:        80,
    Phase.HOME:          150,
}


class OracleSolver:
    """Pose-based state machine + DLS IK on the flange `gripperframe` site."""

    def __init__(
        self,
        env: BlockPickingEnv,
        ik_step_limit: int = 32,
        ik_pos_step: float = 0.05,
        ik_damping: float = 0.08,
        posture_weight: float = 0.05,
        wrist_roll_posture_weight: float = 0.50,
    ) -> None:
        self.env = env
        self.model = env.model
        self._scratch = mujoco.MjData(self.model)
        self._ik_step_limit = ik_step_limit
        self._ik_pos_step = ik_pos_step
        self._ik_damping = ik_damping
        self._posture_weight = posture_weight
        self._wrist_roll_posture_weight = wrist_roll_posture_weight

        # gripperframe is the IK target site (rigid on the flange).
        self._ee_sid = env._ee_sid

        # Two IK chains: wrist_roll is free during APPROACH/GRASP (to orient
        # the jaw), then frozen for everything else so the held block does
        # not rotate in the asymmetric jaw.
        self._arm_idx_with_roll = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        self._arm_idx_no_roll   = np.array([0, 1, 2, 3], dtype=np.int64)
        self._dof_idx_with_roll = env._jnt_dofadr[self._arm_idx_with_roll]
        self._dof_idx_no_roll   = env._jnt_dofadr[self._arm_idx_no_roll]

        # Compute the offset (in gripperframe-local coords) from the
        # gripperframe origin to the *grasp center* (midpoint of the two
        # inner fingertip geoms when the gripper is closed). The grasp
        # center is what we want to align with the block, NOT the
        # gripperframe origin.
        self._grasp_local = self._compute_grasp_local_offset()

        # Runtime state.
        self.phase: Phase = Phase.APPROACH
        self._phase_step: int = 0
        self._settled_steps: int = 0
        self._last_target: Optional[np.ndarray] = None
        self._grasp_block_xy: Optional[np.ndarray] = None
        self._ee_to_block_xy: Optional[np.ndarray] = None  # offset (ee_xy - blk_xy) post-lift
        self._place_lock_q: Optional[np.ndarray] = None
        self._prev_ee: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _compute_grasp_local_offset(self) -> np.ndarray:
        """Local-frame offset from gripperframe origin to the grasp center.

        The grasp center is the midpoint of the two inner fingertip geoms
        when the gripper is fully closed. The fingers are rigidly attached
        to the flange so this offset is constant in gripperframe-local
        coordinates regardless of arm pose.
        """
        m = self.model
        d = self._scratch
        d.qpos[:] = self.env.data.qpos
        d.qvel[:] = 0
        # Fully close the gripper to find the grasp center geometry.
        d.qpos[self.env._jnt_qposadr[5]] = self.env._jnt_low[5]
        mujoco.mj_forward(m, d)
        gf = d.site_xpos[self._ee_sid].copy()
        R  = d.site_xmat[self._ee_sid].reshape(3, 3).copy()
        g1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gripper_finger_collision_inner")
        g2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_finger_collision_inner")
        mid = 0.5 * (d.geom_xpos[g1].copy() + d.geom_xpos[g2].copy())
        return R.T @ (mid - gf)   # local-frame offset

    def _gripperframe_target_for_grasp(self, grasp_world: np.ndarray) -> np.ndarray:
        """Inverse of: grasp_world = gf + R @ grasp_local.

        We want IK to find joints such that the grasp center lands on
        `grasp_world`. Since R depends on the joint pose, we approximate
        with the current orientation -- this is iterated implicitly by
        the closed-loop oracle.
        """
        R = self.env.data.site_xmat[self._ee_sid].reshape(3, 3)
        return grasp_world - R @ self._grasp_local

    # ------------------------------------------------------------------
    def reset(self, obs: np.ndarray) -> None:  # noqa: ARG002
        self.phase = Phase.APPROACH
        self._phase_step = 0
        self._settled_steps = 0
        self._last_target = None
        self._grasp_block_xy = None
        self._ee_to_block_xy = None
        self._place_lock_q = None
        self._prev_ee = None

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> np.ndarray:  # noqa: ARG002
        env = self.env
        cur_q = env.joint_pos().astype(np.float64)
        ee = env.ee_pos()
        blk = env.block_pos()
        tgt = env.container_pos()

        if self._last_target is None:
            self._last_target = cur_q.copy()

        if self.phase in (Phase.HOME, Phase.DONE):
            return self._home_action(cur_q)

        grasp_target_world = None  # If set, IK targets this point for the GRASP CENTER.
        ee_target_world    = None  # If set, IK targets this point for the gripperframe directly.
        gripper_cmd  = GRIPPER_WIDE_OPEN
        advance      = False
        skip_ik      = False
        arm_clip     = ARM_CLIP_FAST
        use_roll     = True

        if self.phase == Phase.APPROACH:
            grasp_target_world = np.array([blk[0], blk[1], APPROACH_CLEARANCE_Z])
            gripper_cmd = GRIPPER_WIDE_OPEN
            if (
                np.linalg.norm(ee[:2] - blk[:2]) < XY_TOL_APPROACH
                and abs(ee[2] - APPROACH_CLEARANCE_Z) < Z_TOL + 0.02
            ):
                advance = True

        elif self.phase == Phase.DESCEND:
            grasp_target_world = np.array([blk[0], blk[1], GRASP_Z])
            gripper_cmd = GRIPPER_PRE_GRASP
            if (
                np.linalg.norm(ee[:2] - blk[:2]) < XY_TOL_DESCEND
                and abs(ee[2] - (GRASP_Z + 0.005)) < Z_TOL + 0.02
            ):
                advance = True
                self._grasp_block_xy = blk[:2].copy()

        elif self.phase == Phase.GRASP:
            if self._grasp_block_xy is None:
                self._grasp_block_xy = blk[:2].copy()
            grasp_target_world = np.array([
                self._grasp_block_xy[0], self._grasp_block_xy[1], GRASP_Z
            ])
            gripper_cmd = GRIPPER_CLOSED
            # Wait for sustained contact (both inner fingers touching block)
            # for several consecutive steps so the jaw has time to actually
            # clamp before we lift.
            if env._gripper_touching_block():
                self._settled_steps += 1
            else:
                self._settled_steps = 0
            if self._settled_steps >= 8 and self._phase_step >= 12:
                advance = True
                self._settled_steps = 0
                self._ee_to_block_xy = (ee[:2] - blk[:2]).copy()
                # Re-anchor: block may have shifted during jaw close.
                self._grasp_block_xy = blk[:2].copy()

        elif self.phase == Phase.LIFT:
            anchor = self._grasp_block_xy if self._grasp_block_xy is not None else blk[:2]
            grasp_target_world = np.array([anchor[0], anchor[1], LIFT_Z])
            gripper_cmd = GRIPPER_CLOSED
            arm_clip = ARM_CLIP_HOLD
            use_roll = False
            lifted_ok = (
                ee[2] > LIFT_Z - 0.020
                and blk[2] > TABLE_Z + 0.040
                and env._gripper_touching_block()
            )
            if lifted_ok:
                self._settled_steps += 1
            else:
                self._settled_steps = 0
            if self._settled_steps >= 6:
                advance = True
                self._settled_steps = 0
                # Recapture offset AFTER lift, when block has settled in jaw.
                self._ee_to_block_xy = (ee[:2] - blk[:2]).copy()

        elif self.phase == Phase.TRANSFER:
            # Target the GRASP CENTER over the container center.
            # We want block above container_xy, so grasp center should be
            # above (tgt_xy - <horizontal block-in-jaw offset>). The block
            # is held basically at the grasp center, so just target tgt_xy.
            grasp_target_world = np.array([tgt[0], tgt[1], TRANSFER_Z])
            gripper_cmd = GRIPPER_CLOSED
            arm_clip = ARM_CLIP_HOLD
            use_roll = False
            # Settle once the BLOCK is above the container (we care about
            # block xy, not gripperframe xy -- they may differ by ~1 cm).
            blk_xy_err = float(np.linalg.norm(blk[:2] - tgt[:2]))
            held = env._gripper_touching_block()
            within = blk_xy_err < 0.012 and abs(ee[2] - TRANSFER_Z) < 0.025 and held
            self._settled_steps = self._settled_steps + 1 if within else 0
            if self._settled_steps >= 6:
                advance = True
                self._settled_steps = 0

        elif self.phase == Phase.DESCEND_PLACE:
            grasp_target_world = np.array([tgt[0], tgt[1], PLACE_Z])
            gripper_cmd = GRIPPER_CLOSED
            arm_clip = ARM_CLIP_HOLD
            use_roll = False
            # Track EE motion; advance once the arm settles (motor saturation
            # equilibrium reached) OR once the block is at container height.
            cur_ee = ee.copy()
            if self._prev_ee is None:
                self._prev_ee = cur_ee
            d_ee = float(np.linalg.norm(cur_ee - self._prev_ee))
            self._prev_ee = cur_ee
            block_in_cup_z = blk[2] < TABLE_Z + CONTAINER_RIM_HEIGHT + 0.005
            ee_settled = d_ee < 0.0010
            if ee_settled or block_in_cup_z:
                self._settled_steps += 1
            else:
                self._settled_steps = 0
            if self._settled_steps >= 10 and self._phase_step >= 25:
                advance = True
                self._settled_steps = 0
                self._place_lock_q = cur_q.copy()
                self._prev_ee = None

        elif self.phase == Phase.RELEASE:
            # Hold joints exactly; open the gripper. No IK during release so
            # the arm does not drift the moment the load on the gripper changes.
            target_q = (self._place_lock_q if self._place_lock_q is not None else cur_q).copy()
            target_q[5] = self._unnormalize_gripper(GRIPPER_RELEASE)
            target_q = self._step_toward_split(self._last_target, target_q, ARM_CLIP_HOLD)
            self._last_target = target_q
            if (self._phase_step >= 10 and not env._gripper_touching_block()):
                self._advance(Phase.SETTLE)
            else:
                self._phase_step += 1
                if self._phase_step > TIMEOUT[Phase.RELEASE]:
                    self._advance(Phase.SETTLE)
            return env.normalize_joint(target_q).astype(np.float32)

        elif self.phase == Phase.SETTLE:
            # Wait for the block to come to rest in the container before
            # any retreat motion drags it out. Hold the post-release joints.
            target_q = (self._place_lock_q if self._place_lock_q is not None else cur_q).copy()
            target_q[5] = self._unnormalize_gripper(GRIPPER_RELEASE)
            target_q = self._step_toward_split(self._last_target, target_q, ARM_CLIP_HOLD)
            self._last_target = target_q
            blk_vel = env.data.qvel[env._blk_qvel_adr:env._blk_qvel_adr + 3]
            if float(np.linalg.norm(blk_vel)) < 0.005:
                self._settled_steps += 1
            else:
                self._settled_steps = 0
            if self._settled_steps >= 10 or self._phase_step > TIMEOUT[Phase.SETTLE]:
                self._advance(Phase.RETREAT)
                self._settled_steps = 0
            else:
                self._phase_step += 1
            return env.normalize_joint(target_q).astype(np.float32)

        elif self.phase == Phase.RETREAT:
            # Lift straight up from the release spot so we don't graze the
            # block on the way out.
            if self._place_lock_q is not None:
                # Just move up to the LIFT_Z plane above release point.
                # We do this via IK targeting (tgt_xy, RELEASE_LOWER_Z + 0.08).
                ee_target_world = np.array([tgt[0], tgt[1], LIFT_Z + 0.02])
            else:
                ee_target_world = np.array([ee[0], ee[1], LIFT_Z])
            gripper_cmd = GRIPPER_WIDE_OPEN
            arm_clip = ARM_CLIP_HOLD
            use_roll = False
            if ee[2] > LIFT_Z - 0.020 or self._phase_step >= TIMEOUT[Phase.RETREAT]:
                advance = True

        else:
            advance = True

        # Generic advance / timeout.
        if advance:
            self._advance(Phase(int(self.phase) + 1))
        else:
            self._phase_step += 1
            if self._phase_step > TIMEOUT.get(self.phase, 100):
                self._advance(Phase(int(self.phase) + 1))

        # IK
        if use_roll:
            arm_idx, dof_idx = self._arm_idx_with_roll, self._dof_idx_with_roll
        else:
            arm_idx, dof_idx = self._arm_idx_no_roll, self._dof_idx_no_roll

        if skip_ik:
            target_q = cur_q.copy()
        else:
            if grasp_target_world is not None:
                gf_target = self._gripperframe_target_for_grasp(grasp_target_world)
            else:
                gf_target = ee_target_world
            target_q = self._ik(cur_q, gf_target, arm_idx, dof_idx)
        target_q[5] = self._unnormalize_gripper(gripper_cmd)

        target_q = self._step_toward_split(self._last_target, target_q, arm_clip)
        self._last_target = target_q
        return env.normalize_joint(target_q).astype(np.float32)

    # ------------------------------------------------------------------
    def _home_action(self, cur_q: np.ndarray) -> np.ndarray:
        env = self.env
        target_q = INIT_QPOS.copy()
        target_q[5] = self._unnormalize_gripper(GRIPPER_WIDE_OPEN)
        target_q = self._step_toward_split(self._last_target, target_q, ARM_CLIP_FAST)
        self._last_target = target_q
        if self.phase == Phase.HOME:
            if np.max(np.abs(cur_q[:5] - INIT_QPOS[:5])) < 0.1:
                self._advance(Phase.DONE)
            else:
                self._phase_step += 1
                if self._phase_step > TIMEOUT[Phase.HOME]:
                    self._advance(Phase.DONE)
        return env.normalize_joint(target_q).astype(np.float32)

    # ------------------------------------------------------------------
    def _advance(self, nxt: Phase) -> None:
        self.phase = nxt
        self._phase_step = 0

    def _step_toward_split(self, cur: np.ndarray, target: np.ndarray, arm_clip: float) -> np.ndarray:
        out = cur.copy()
        out[:5] = cur[:5] + np.clip(target[:5] - cur[:5], -arm_clip, arm_clip)
        out[5]  = cur[5]  + float(np.clip(target[5] - cur[5], -GRIP_CLIP, GRIP_CLIP))
        return out

    def _unnormalize_gripper(self, gripper_norm: float) -> float:
        low = self.env._jnt_low[5]
        high = self.env._jnt_high[5]
        return low + 0.5 * (gripper_norm + 1.0) * (high - low)

    # ------------------------------------------------------------------
    def _ik(
        self,
        q0: np.ndarray,
        gf_target: np.ndarray,
        arm_idx: np.ndarray,
        dof_idx: np.ndarray,
    ) -> np.ndarray:
        """DLS position IK on gripperframe with null-space posture pull."""
        s = self._scratch
        s.qpos[:] = self.env.data.qpos
        s.qvel[:] = 0
        s.qpos[self.env._jnt_qposadr] = q0

        target = gf_target.astype(np.float64).copy()
        q_ref = INIT_QPOS.astype(np.float64)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        damping = self._ik_damping
        n_arm = len(arm_idx)
        I_arm = np.eye(n_arm)
        for _ in range(self._ik_step_limit):
            mujoco.mj_forward(self.model, s)
            ee = s.site_xpos[self._ee_sid].copy()
            err = target - ee
            n = np.linalg.norm(err)
            err_clipped = err if n <= self._ik_pos_step else err * (self._ik_pos_step / n)
            mujoco.mj_jacSite(self.model, s, jac_pos, jac_rot, self._ee_sid)
            J = jac_pos[:, dof_idx]
            JJt = J @ J.T + damping ** 2 * np.eye(3)
            J_pinv = J.T @ np.linalg.solve(JJt, np.eye(3))
            dq_task = J_pinv @ err_clipped

            cur_arm = np.array(
                [s.qpos[self.env._jnt_qposadr[ji]] for ji in arm_idx], dtype=np.float64
            )
            ref_arm = q_ref[arm_idx]
            pw = np.full(len(arm_idx), float(self._posture_weight))
            for k, ji in enumerate(arm_idx):
                if int(ji) == 4:  # wrist_roll is fully redundant for pos-only IK
                    pw[k] = self._wrist_roll_posture_weight
            dq_post = pw * (ref_arm - cur_arm)
            N = I_arm - J_pinv @ J
            dq_arm = dq_task + N @ dq_post

            for k, ji in enumerate(arm_idx):
                addr = self.env._jnt_qposadr[ji]
                s.qpos[addr] += dq_arm[k]

        q_out = self.env.joint_pos().astype(np.float64).copy()
        for k, ji in enumerate(arm_idx):
            q_out[ji] = s.qpos[self.env._jnt_qposadr[ji]]
        return np.clip(q_out, self.env._jnt_low, self.env._jnt_high)
