# Notes for Claude

## Commit messages
Keep them simple — short one-line subjects, no body unless something is
genuinely surprising. Match the existing log style (`improve utility`,
`fix container rim height`, `claude explores rewards v6`).

## Sim-to-real debugging (SO-101 vision policy) — findings as of 2026-06-10

Deploying the mjlab vision policy on the real arm via `pixi run run-so101-policy`
(launch: `src/lerobot_robots_bringup/launch/so101_policy.launch.py`,
node: `src/lerobot_robots_inference/.../policy_node.py`). The real arm moved to
an obviously wrong pose. What we established:

- **The deployed checkpoint is GOOD in sim.** `model_19999.pt` from
  `mjlab_rl/logs/rsl_rl/so101_block_picking_vision/2026-06-03_20-21-22/`
  (the source of the deployed `.jit`) reaches the training-peak reward in a
  headless rollout (~0.283 reward/step × 400-step episode ≈ 113/ep; tensorboard
  `Train/mean_reward`≈73.7, peak 113; `Episode_Reward/success`≈9.67). So the bad
  real-world pose is **not** a bad policy / retraining problem.
- **The command path and units are correct.** In `zero_action` mode (commands a
  zero action = home pose) the arm settles at home: present vs target match
  within ~1°. So the rad↔deg conversion in `lerobot_robot_node` (topics are
  radians, motor bus is degrees; config has `use_degrees: true`), joint ordering,
  and the rate-limit clamp are all fine. The earlier wrong pose is **not** a unit
  bug.
- **Large policy action magnitudes are normal, not a bug.** `clip_actions=None`
  in the PPO cfg; motion is bounded by the rate-limiter (`SO101_MAX_RELATIVE_TARGET`,
  ±0.1 rad/step). The policy legitimately saturates its raw outputs (gripper pegged
  at +300s) and still succeeds in sim. So large `policy_out` on the real robot is
  not itself evidence of a problem — the policy is sensitive to action *direction*,
  which comes from observations.
- **Remaining suspect: the sim-to-real observation gap.** Most likely the camera
  obs (device mapping `wrist_device=2`/`top_device=0`, framing/FOV, color, the
  90° top-cam rotation) and/or the home/calibration offset between the real arm's
  zero and sim `HOME_JOINT_POS`.

Action term detail worth knowing: `RateLimitedJointPositionAction`
(`mjlab_rl/mjlab_rl/envs/actions.py`) sends `target = home + scale*raw −
encoder_bias`. `encoder_bias` is **0** for the block-picking task (only the
velocity/tracking tasks enable the DR event that sets it), so zero action == home.

Diagnostic aids currently in the tree (TEMP):
- `policy_node.py` `zero_action` param: commands zeros (= home) while still
  running the policy on live cameras and logging its output (1 Hz: present /
  target / raw_action / scale*raw / policy_out / policy_tgt). `pixi run
  run-so101-policy` currently passes `zero_action:=true` — remove it to command
  the real policy.
- `mjlab_rl/scripts/_check_actions.py`: headless rollout that reports action
  magnitude + reward/return for a checkpoint in sim.

Also note: the launch's default `arm_config` was fixed to resolve from
`lerobot_robots_robots` (was wrongly pointing at `lerobot_robots_bringup`).

Next steps not yet done: (a) verify the exported `.jit` reproduces the
checkpoint on one sim obs (rule out a broken export); (b) dump the exact 64×64
images the node feeds the policy and compare to sim `wrist_cam.png`/`top_cam.png`.
