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
- **The `.jit` export is faithful.** It reproduces the checkpoint exactly
  (max|Δ|=0) on sim observations; obs layout confirmed `actor:(1,12)`,
  `camera:(1,6,64,64)`. So the artifact and obs layout are not the problem.
- **ROOT CAUSE: the wrist/top cameras were swapped (now FIXED).** Dumping the
  64×64 frames the node feeds vs the sim renders showed the `top` feed pointed at
  the ceiling and the `wrist` feed showed the bench. At the home pose the
  wrist-mounted cam points up, the overhead cam looks down — so the correct
  mapping is `wrist_device=0`, `top_device=2` (was 2/0). Fixed in the
  `run-so101-policy` task and the launch default. With OOD images the (good)
  policy steered confidently wrong; the swap explains the bad pose.

Still to confirm: the wrist-cam *viewpoint* matches sim `hand_eye` once the arm
is in a task pose (only checked at home, where it points at the ceiling).

Action term detail worth knowing: `RateLimitedJointPositionAction`
(`mjlab_rl/mjlab_rl/envs/actions.py`) sends `target = home + scale*raw −
encoder_bias`. `encoder_bias` is **0** for the block-picking task (only the
velocity/tracking tasks enable the DR event that sets it), so zero action == home.

Fixes kept in the tree:
- Camera swap: `wrist_device=0`, `top_device=2`.
- `arm_config` launch default now resolves from `lerobot_robots_robots` (was
  wrongly `lerobot_robots_bringup`).
- `policy_node.py` homes the arm (rate-limited) to `HOME_JOINT_POS` on startup
  before policy control, mirroring the sim reset (`home_on_start`, default true).

Temporary diagnostics removed from `policy_node.py` after use (`zero_action`,
`dump_images`, joint/image logging). Standalone diagnostic scripts
`mjlab_rl/scripts/_check_actions.py` (action magnitude + reward in sim) and
`_check_jit.py` (.jit-vs-checkpoint equivalence) may still be present — throwaway.
