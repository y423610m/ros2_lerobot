#!/usr/bin/env python3
"""Runs a TorchScript-exported mjlab vision policy on the real SO-101 arm.

Subscribes:
    /follower/joint_states  (sensor_msgs/JointState, radians, names in any order)
    /wrist_cam/image_raw    (sensor_msgs/Image, rgb8, native resolution)
    /top_cam/image_raw      (sensor_msgs/Image, rgb8, native resolution)

Publishes:
    /follower/joint_command (sensor_msgs/JointState, radians,
                             names in SO101_JOINT_NAMES order)

The control chain replicates exactly what the sim env applies at training
time — any drift here puts the policy out of distribution on the first step:

    raw_action  = actor(state_obs, image_obs)            # 6-dim
    target      = home_pos + ACTION_SCALE * raw_action   # use_default_offset=True in sim
    target      = clamp(target,                          # mirrors RateLimitedJointPositionAction
                        present − MAX_RELATIVE_TARGET,
                        present + MAX_RELATIVE_TARGET)
    publish target

Export the checkpoint to TorchScript first via mjlab_rl's
``scripts/export_to_jit.py``; this node consumes the resulting .jit file.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


# ---------------------------------------------------------------------------
# Constants — MUST match training. Source of truth is
# ``mjlab_rl/mjlab_rl/assets/so101.py`` and ``mjlab_rl/tasks/block_picking.py``.
# Duplicated here so this node doesn't import the simulator stack.
# ---------------------------------------------------------------------------

SO101_JOINT_NAMES: tuple[str, ...] = (
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
)

HOME_JOINT_POS: dict[str, float] = {
    'shoulder_pan': 0.0,
    'shoulder_lift': -1.2,
    'elbow_flex': 0.0,
    'wrist_flex': 1.5,
    'wrist_roll': -1.5,
    'gripper': 0.0,
}

ACTION_SCALE: dict[str, float] = {
    'shoulder_pan': 0.5,
    'shoulder_lift': 0.5,
    'elbow_flex': 0.5,
    'wrist_flex': 0.5,
    'wrist_roll': 0.5,
    'gripper': 0.8,
}

MAX_RELATIVE_TARGET: dict[str, float] = {
    'shoulder_pan': 0.1,
    'shoulder_lift': 0.1,
    'elbow_flex': 0.1,
    'wrist_flex': 0.1,
    'wrist_roll': 0.1,
    'gripper': 0.3,
}

CAMERA_HW: tuple[int, int] = (64, 64)  # H, W expected by the actor's CNN


def _by_joint_order(d: dict[str, float]) -> np.ndarray:
    return np.array([d[n] for n in SO101_JOINT_NAMES], dtype=np.float32)


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__('policy_node')

        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('device', 'auto')  # 'auto' | 'cpu' | 'cuda'
        self.declare_parameter('joint_states_topic', 'joint_states')
        self.declare_parameter('joint_command_topic', 'joint_command')
        self.declare_parameter('wrist_cam_topic', '/wrist_cam/image_raw')
        self.declare_parameter('top_cam_topic', '/top_cam/image_raw')
        # TEMP/diagnostic: when true, replace the policy output with an all-zero
        # action vector and run it through the normal control equation
        # (target = home + scale*0, then the present ± MAX_RELATIVE_TARGET
        # clamp). This is exactly what the policy commands when it outputs
        # zeros, so it exercises the real action/unit path without the network.
        # No checkpoint or cameras needed.
        self.declare_parameter('zero_action', False)

        self._zero_action = bool(self.get_parameter('zero_action').value)
        control_hz = float(self.get_parameter('control_hz').value)
        device_str = str(self.get_parameter('device').value)
        if device_str == 'auto':
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device_str)

        checkpoint_path = str(self.get_parameter('checkpoint_path').value)
        if not checkpoint_path:
            if not self._zero_action:
                raise RuntimeError('checkpoint_path parameter is required')
            self.get_logger().warn(
                'ZERO_ACTION mode with no checkpoint: commanding zeros only '
                '(no policy output to print)'
            )
            self._policy = None
        else:
            self.get_logger().info(f'loading policy from {checkpoint_path} on {self._device}')
            self._policy = torch.jit.load(checkpoint_path, map_location=self._device)
            self._policy.eval()
            if self._zero_action:
                self.get_logger().warn(
                    'ZERO_ACTION mode: commanding zeros (target = home), but '
                    'still running the policy to print its output'
                )

        # Cached constants as numpy arrays in the canonical joint order.
        self._home = _by_joint_order(HOME_JOINT_POS)
        self._scale = _by_joint_order(ACTION_SCALE)
        self._max_rel = _by_joint_order(MAX_RELATIVE_TARGET)

        # Caches populated by subscriber callbacks; consumed by the timer.
        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_wrist: Optional[np.ndarray] = None
        self._latest_top: Optional[np.ndarray] = None
        self._last_action = np.zeros(len(SO101_JOINT_NAMES), dtype=np.float32)

        self._bridge = CvBridge()

        joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        joint_command_topic = str(self.get_parameter('joint_command_topic').value)
        wrist_topic = str(self.get_parameter('wrist_cam_topic').value)
        top_topic = str(self.get_parameter('top_cam_topic').value)

        self.create_subscription(JointState, joint_states_topic, self._on_joint_states, 10)
        self.create_subscription(Image, wrist_topic, self._on_wrist_image, 10)
        self.create_subscription(Image, top_topic, self._on_top_image, 10)
        self._cmd_pub = self.create_publisher(JointState, joint_command_topic, 10)

        self._timer = self.create_timer(1.0 / control_hz, self._on_timer)

        self.get_logger().info(
            f'policy_node ready  control_hz={control_hz}  '
            f'sub: {joint_states_topic} {wrist_topic} {top_topic}  '
            f'pub: {joint_command_topic}'
        )

    # ---- Subscriber callbacks (cache only — no inference here) ------------

    def _on_joint_states(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        missing = [n for n in SO101_JOINT_NAMES if n not in positions]
        if missing:
            self.get_logger().warn(f'joint_states missing: {missing}', throttle_duration_sec=2.0)
            return
        self._latest_joint_pos = np.array(
            [positions[n] for n in SO101_JOINT_NAMES], dtype=np.float32
        )

    def _on_wrist_image(self, msg: Image) -> None:
        self._latest_wrist = self._preprocess_image(msg)

    def _on_top_image(self, msg: Image) -> None:
        self._latest_top = self._preprocess_image(msg)

    def _preprocess_image(self, msg: Image) -> np.ndarray:
        # Camera publisher publishes rgb8; cv_bridge gives HxWx3 uint8.
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h, w = CAMERA_HW
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        # HWC uint8 -> CHW float32 in [0, 1].
        return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

    # ---- Control loop -----------------------------------------------------

    def _run_policy(self, joint_pos: np.ndarray) -> Optional[np.ndarray]:
        """Forward the policy on the current obs; returns the (6,) raw action,
        or None if no policy is loaded / camera frames haven't arrived yet."""
        if self._policy is None or self._latest_wrist is None or self._latest_top is None:
            return None
        joint_pos_rel = joint_pos - self._home  # mirrors mdp_obs.joint_pos_rel
        state_obs = np.concatenate([joint_pos_rel, self._last_action], axis=0)
        cam_obs = np.concatenate([self._latest_wrist, self._latest_top], axis=0)
        state_t = torch.from_numpy(state_obs).to(self._device).unsqueeze(0)        # (1, 12)
        cam_t = torch.from_numpy(cam_obs).to(self._device).unsqueeze(0)            # (1, 6, 64, 64)
        with torch.inference_mode():
            return self._policy(state_t, [cam_t]).squeeze(0).cpu().numpy()         # (6,)

    def _on_timer(self) -> None:
        # TEMP/diagnostic zero-action mode: command an all-zero action (target =
        # home), but still run the policy (if loaded + cameras ready) so its
        # output can be logged for comparison — without ever sending it.
        if self._zero_action:
            if self._latest_joint_pos is None:
                return
            joint_pos = self._latest_joint_pos
            policy_out = self._run_policy(joint_pos)  # logged only, not commanded
            raw = np.zeros(len(SO101_JOINT_NAMES), dtype=np.float32)
            target = self._home + self._scale * raw
            target = np.clip(target, joint_pos - self._max_rel, joint_pos + self._max_rel)
            self._publish_target(target)
            self._log_joints(joint_pos, target, raw, policy_out)
            self._last_action = raw
            return

        if (
            self._latest_joint_pos is None
            or self._latest_wrist is None
            or self._latest_top is None
        ):
            return  # warm-up: wait for first observations

        joint_pos = self._latest_joint_pos
        raw = self._run_policy(joint_pos)

        target = self._home + self._scale * raw
        lo = joint_pos - self._max_rel
        hi = joint_pos + self._max_rel
        target = np.clip(target, lo, hi)

        self._publish_target(target)
        self._log_joints(joint_pos, target, raw)
        self._last_action = raw.astype(np.float32)

    def _log_joints(
        self,
        present: np.ndarray,
        target: np.ndarray,
        raw: np.ndarray,
        policy_out: Optional[np.ndarray] = None,
    ) -> None:
        """Throttled (1 Hz) dump of present joint pos, commanded target, and the
        raw action — in canonical joint order — for comparing w/ and w/o
        zero_action. Present/target are radians; deg shown for the motor side.
        In zero_action mode ``policy_out`` is the (uncommanded) policy action;
        we also show where it *would* have driven the arm (home + scale*out)."""
        def fmt(arr: np.ndarray) -> str:
            return '  '.join(f'{n}={v:+.3f}' for n, v in zip(SO101_JOINT_NAMES, arr))

        lines = [
            'present[rad] ' + fmt(present),
            '    present[deg]        ' + fmt(present * 180.0 / np.pi),
            '    target [rad]        ' + fmt(target),
            '    target [deg]        ' + fmt(target * 180.0 / np.pi),
            # raw_action is the unitless normalized policy output; scale*raw is
            # the resulting joint-angle offset (rad) added on top of home.
            '    raw_action[unitless] ' + fmt(raw),
            '    scale*raw  [rad]    ' + fmt(self._scale * raw),
        ]
        if policy_out is not None:
            would_be = self._home + self._scale * policy_out
            lines.append('    policy_out[unitless] ' + fmt(policy_out))
            lines.append('    policy scale*out[rad] ' + fmt(self._scale * policy_out))
            lines.append('    policy_tgt[deg]      ' + fmt(would_be * 180.0 / np.pi))
        self.get_logger().info('\n'.join(lines), throttle_duration_sec=1.0)

    def _publish_target(self, target: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(SO101_JOINT_NAMES)
        msg.position = target.astype(float).tolist()
        self._cmd_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
