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

## Action chunking + chunk-size curriculum (branch `20260701_actionChuck`)

Action chunking: the actor emits `CHUNK_SIZE` future actions per inference, executed
open-loop (`ChunkedVecEnvWrapper`, `mjlab_rl/envs/chunked_env.py`; env var `CHUNK_SIZE`,
default 1 = the original closed-loop path). Interior chunk steps skip `sim.sense`+obs
(render only the boundary frame); the `.jit` metadata carries `chunk_size` and the ROS
node streams the chunk at 50 Hz. Deploy horizon = chunk=50 (1 s).

Chunk-size **curriculum** (grows the open-loop horizon with the DR level so the grasp is
learned near-closed-loop, then made robust — cheap early iters too, since collection
scales ~linearly with chunk): `dr0=1 → dr1=10 → dr2=25 → dr3=50` (see the pixi comment
block above the vision tasks for exact commands). Because chunk changes the actor's
output-head size, resume can't be a strict load — the vision `runner_cls` is
`WarmStartOnPolicyRunner` (`mjlab_rl/envs/warmstart_runner.py`): on a head-shape change it
transfers the CNN backbone + trunk + critic verbatim and **prefix-extends** the head/std
(the head is action-major, so the first `old_chunk*6` rows are actions `0..old_chunk-1` —
copied; the tail is fresh-init), resets the optimizer, and starts at iteration 0.
Same-chunk resume (and export) sees no mismatch and delegates to the base loader.

## Chunk policy via distillation (preferred over from-scratch chunk RL)

From-scratch chunk RL (incl. the chunk-size curriculum) failed to even *reach* the object —
a blind 1 s open-loop commit under sparse reward is an exploration dead-end. If a GOOD
**closed-loop** (chunk=1) policy exists, distill it instead: `scripts/distill_chunk.py`
(pixi `distill-mjlab-vision`, `TEACHER=<closed-loop model_*.pt>`). Behavior cloning (ACT
recipe): roll the teacher out in sim (a perfectly queryable expert), form non-overlapping
`(boundary_obs → next-50 teacher raw actions)` windows aligned to episode reset (= the
student's deploy inference points), regress the chunk=50 student head to them with smooth-L1.
Reuses: `ChunkedVecEnvWrapper` (sizes the 300-dim student head), `WarmStartOnPolicyRunner.load`
(warm-starts the student = teacher backbone/trunk/**obs_normalizer** + head prefix), and a
bare 6-dim `SpatialSoftmaxCNNModel` as the frozen teacher (no 2nd env/VRAM). Output is a
standard ckpt → `export_to_jit.py` stamps `chunk_size=50` unchanged → ROS node unchanged.
Start on `-DR0` to confirm it reaches/grasps, then re-distill on `-DR3` for the deploy ckpt.
Teacher raw actions are large (gripper ~300) → use L1/smooth-L1, never MSE. Known limit:
open-loop can't reproduce the teacher's sub-second feedback, so BC may plateau below teacher
success — `--dagger` (student-driven rollout, teacher relabels visited states) is the fix.

Findings + knobs from the chunk=50/chunk=10 distill runs (2026-07):
- **chunk=50 hits an information floor** (student fits action 0 well, actions ~25-49 are
  unpredictable from the boundary frame; eval success ~25% of teacher). **chunk=10 works**:
  BC ≈ 0.47 vs teacher 1.95 `Episode_Reward/success`; + DAgger deployed on the real arm at
  5 Hz inference (node reads `chunk_size` from the .jit; rebuild colcon after node changes!).
- Pixi knobs: `RESUME=<distill model_*.pt>` continues a run (weights + Adam state, iteration
  numbering; `ITER` = additional updates), `DAGGER=1` toggles `--dagger`.
- Eval is a fixed `eval_steps=1200` **control-step** horizon (≥3 episodes). A chunk-count
  horizon shorter than one episode logs only early-failure episodes → success reads ~0
  (bit us at chunk=10). `ChunkedVecEnvWrapper` also merges interior sub-step `extras["log"]`
  so mid-chunk episode ends aren't dropped.
- Distill run dirs are date-first: `<teacher_run>/<stamp>_distill_chunk<N>/`.
- `scripts/play.py` streams a chunk one action per viewer tick (deploy semantics, smooth
  viewer); `ChunkedVecEnvWrapper.step` passes a base-dim (6) action through as one control
  step for this.
