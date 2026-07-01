# Deploying a trained mjlab policy on the real SO-101

## What this package contains

- `camera_publisher_node` — `cv2.VideoCapture` → `sensor_msgs/Image` (rgb8). One
  instance per camera. Configurable device, topic, resolution, fps.
- `policy_node` — subscribes joint_states + 2 image topics, loads a TorchScript
  `.jit` policy, runs forward at `control_hz`, applies the same
  `home + scale × raw_action → rate-limit clamp` chain the sim env uses, and
  publishes `joint_command` in `SO101_JOINT_NAMES` order.

Supporting pieces in sibling packages:

- `src/lerobot_robots_bringup/launch/so101_policy.launch.py` — one-shot bring-up
  for arm + 2 cameras + policy.
- `mjlab_rl/scripts/export_to_jit.py` — converts a `model_*.pt` checkpoint to
  `policy.jit`. Decouples deployment from the simulator stack.
- `mjlab_rl/mjlab_rl/assets/so101.py::SO101_MAX_RELATIVE_TARGET` — exported as a
  public constant so training and deployment can't drift.

## What's verified so far

- `pixi run colcon build --packages-select lerobot_robots_inference` builds
  clean.
- Both `camera_publisher_node` and `policy_node` import in the pixi env
  (rclpy, torch 2.10, cv_bridge, cv2, numpy all present).
- `ros2 pkg executables lerobot_robots_inference` shows both entry points.
- Vision env config (`Mjlab-SO101-Block-Picking-Rgb`) still loads after the
  constants refactor.

## What still needs you before first run

1. **Export a real checkpoint.** No trained `.pt` was tested end-to-end through
   `export_to_jit.py`. Run:
   ```bash
   cd mjlab_rl
   uv run python scripts/export_to_jit.py Mjlab-SO101-Block-Picking-Rgb \
       --checkpoint-file logs/rsl_rl/so101_block_picking_vision/<run>/model_<N>.pt
   ```
   The script writes `policy.jit` next to the checkpoint. Confirm
   `policy_node` actually loads it with no shape mismatch.

2. **Pick camera device indices.**
   ```bash
   v4l2-ctl --list-devices
   ```
   Pass them into the launch file (`wrist_device:=N top_device:=M`). Default
   wrist=0, top=1.

3. **Mount the wrist cam on `camera_fixture`.** The sim wrist cam wraps the
   `hand_eye` MJCF camera at the fixture pose; physical mounting must match or
   the visual sim-to-real gap will dominate. See
   `mjlab_rl/mjlab_rl/tasks/block_picking_vision.py:85-89`.

4. **Line up the top cam.** Sim has it at `(-0.34, -0.02, 1.30)` relative to
   the table body, rotated 90° about world z, looking straight down. Use the
   rendered `top_cam.png` as the alignment target. See
   `block_picking_vision.py:95-105`.

5. **Pick a `control_hz`.** Sim runs the policy at 50 Hz
   (`timestep=0.005 × decimation=4`). The existing
   `src/lerobot_robots_robots/lerobot_robots_robots/lerobot_robot_node.py:30`
   publishes joint_states at 10 Hz (`timer_period = 0.1`). For the policy to
   see fresh observations every tick, bump that to match the policy rate IF the
   SOFollower can read the Dynamixel bus that fast. Measure first:
   ```bash
   ros2 topic hz /follower/joint_states
   ```
   Treat `control_hz` as a tunable. Don't run policy faster than joint_states
   refreshes or it'll fire on stale observations.

## How to actually run

```bash
# 0. Build + source
pixi run colcon build
source install/setup.bash

# 1. Verify the policy loads in isolation (no hardware)
ros2 run lerobot_robots_inference policy_node --ros-args \
    -p checkpoint_path:=$(realpath mjlab_rl/logs/rsl_rl/so101_block_picking_vision/<run>/policy.jit) \
    -p control_hz:=50.0

# 2. End-to-end on hardware
ros2 launch lerobot_robots_bringup so101_policy.launch.py \
    checkpoint:=$(realpath mjlab_rl/logs/rsl_rl/so101_block_picking_vision/<run>/policy.jit) \
    wrist_device:=0 top_device:=1 control_hz:=50.0
```

## Things to watch during the first hardware run

- `rqt_image_view /wrist_cam/image_raw` and `/top_cam/image_raw` — frames
  flowing, white balance/exposure reasonable.
- `ros2 topic hz /follower/joint_command` — matches `control_hz`.
- `ros2 topic hz /follower/joint_states` — what the policy actually sees;
  policy is fed stale data if this is slower than `control_hz`.
- `ros2 topic echo --once /follower/joint_command` — values in radians, in
  `SO101_JOINT_NAMES` order, within `±MAX_RELATIVE_TARGET` of the latest
  `joint_states`.

## Failure-mode debug order (most-likely first)

1. **Camera alignment.** If the policy moves but in completely wrong
   directions, the wrist or top view doesn't match what was rendered in sim.
   Compare against `mjlab_rl/wrist_cam.png` and `mjlab_rl/top_cam.png`.
2. **Control rate / observation latency.** Bump or drop `control_hz` and
   `joint_states` rate; mismatched rates degrade behavior.
3. **Lighting.** Domain randomization covers a range; ambient outside that
   range degrades the visual policy. Try matching sim lighting roughly.
4. **Joint home calibration.** Verify the real arm's joint zeros match the
   sim's `HOME_JOINT_POS` (`shoulder_lift: -1.2`, `wrist_flex: 1.5`,
   `wrist_roll: -1.5`). If servo calibration is off, the policy's home-pose
   reward becomes punishing on real.
