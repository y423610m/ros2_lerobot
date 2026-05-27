# mjlab_rl

SO‑101 arm pick‑and‑place (block → container), with a scripted oracle that
generates teacher trajectories used to train an MLP policy by Behavior
Cloning, and a second script that distills that policy into a new (smaller)
student.

The MuJoCo scene file (`mujoco_models/block_picking.xml`) is reused from
`mujoco_rl/`; everything else is built from scratch in this workspace.

## What's in here

| What | Where | Notes |
|---|---|---|
| **Env**            | `mjlab_rl/envs/block_picking_env.py` | `BlockPickingEnv`, Gymnasium-compatible. 25-dim state obs (joints, ee, block, container), 6-dim normalized joint-target action. |
| **Oracle solver**  | `mjlab_rl/solvers/oracle.py`         | `OracleSolver`: closed-loop pose-based state machine (APPROACH→DESCEND→GRASP→LIFT→TRANSFER→DESCEND_PLACE→RELEASE→RETREAT→HOME). Damped least-squares IK with **null-space posture regularization** (strong on wrist_roll) computed on a scratch `MjData`. |
| **Solver test**    | `scripts/test_solver.py`             | Rolls out the oracle in the env, prints success rate / phase / placement metrics, optional `--video out.mp4` and `--render`. |
| **BC + DAgger**    | `scripts/train_bc.py`                | (1) Collects N successful oracle demos, (2) MSE-trains the MLP, (3) optional `--dagger-rounds N` DAgger phase where the student rolls out and the *oracle* relabels each visited state. Saves the best checkpoint by env success. |
| **Policy distill** | `scripts/distill.py`                 | Loads a trained policy as **teacher**, distills it into a new (typically smaller) student with both offline teacher rollouts and DAgger-style student rollouts relabelled by the teacher. |
| **Eval**           | `scripts/eval_policy.py`             | Replay a saved checkpoint in the env for a few episodes, report success / return, write video. |

## Quick start

```bash
cd mjlab_rl
uv sync

# 1) Sanity-check the env + oracle.
uv run python -m scripts.test_solver --episodes 10

# 2) BC training: collect 100 successful demos and train, then 3 DAgger rounds.
uv run python -m scripts.train_bc \
    --num-demos 100 --epochs 40 \
    --eval-every 10 --eval-episodes 10 \
    --dagger-rounds 3 --dagger-episodes 20 --dagger-epochs 10 \
    --hidden 256 256

# 3) Distill BC into a smaller (64x64) student.
uv run python -m scripts.distill \
    --teacher-ckpt checkpoints/bc.pt \
    --student-hidden 64 64 \
    --offline-demos 50 --dagger-rounds 3 --rollouts-per-round 20

# 4) Eval any saved checkpoint and record a video.
uv run python -m scripts.eval_policy --ckpt checkpoints/student.pt --episodes 10 --video /tmp/student.mp4
```

## End-effector site choice (gripperframe)

The SO101 gripper has **1 DoF actuating 2 fingers**, but only one finger
moves — the other is fixed to the flange. Sites that live on the moving jaw
(`ee_gripper`) or on the wrist body (`ee_wrist`) therefore shift in world
space whenever the gripper opens or closes, which:

1. Makes the IK target a moving function of the gripper DoF, so the IK loop
   "chases its own tail" whenever the jaw moves.
2. Breaks distance-to-block and success metrics, because they implicitly
   depend on the gripper opening rather than the arm pose.

We therefore expose `gripperframe` (a site that the SO101 URDF rigidly
attaches to the flange) as the EE for both the env (`env.ee_pos()`) and the
oracle's damped least-squares IK target. The flange-fixed gripperframe pose
is a function of arm joints 1–5 only — i.e., the gripper DoF (joint 6) does
not enter the IK Jacobian for position, which is exactly the desired
behaviour for a 1-DoF asymmetric gripper.

For convenience the env also computes a constant **grasp-local offset** at
construction time, derived from the fixed-finger collision geom expressed in
gripperframe local coords (see `BlockPickingEnv._calibrate_grasp_offset`).
This is exposed via `env.grasp_offset_world()` and lets downstream code
recover the world-frame grasp center as `ee_pos + grasp_offset_world()`
without ever touching the moving `ee_gripper` / `ee_wrist` sites at runtime.

## Notes on the oracle

The oracle uses pos-only damped least-squares IK on the EE site. Pos-only IK
on a 6-DoF arm has multiple solutions; left unregularized the IK tends to
spin wrist_roll into bizarre poses that drop the held block. The IK adds a
*null-space* posture term:

```
dq = J⁺ err  +  (I − J⁺J) · diag(w) · (q_init − q)
```

with `w[wrist_roll]` much larger than the other joints, so the redundant DoFs
are biased back toward the initial pose without sacrificing task accuracy.

Empirically the oracle reaches ~20–30 % task success across random block /
container placements. Failures are dominated by (a) the gripper jaw sweeping
the block during the open→close transition and (b) the block bouncing off
the container rim on release. Both are addressable with further geometric
tuning, orientation-aware IK, or a learned residual on top of the oracle —
all out of scope here.

This oracle quality is *good enough* to drive BC + DAgger; the student can
exceed the oracle by avoiding the oracle's failure modes once the dataset
sees a wider state distribution.

## Directory layout

```
mjlab_rl/
├── mjlab_rl/
│   ├── envs/block_picking_env.py
│   ├── solvers/oracle.py
│   └── policies/mlp.py
├── scripts/
│   ├── test_solver.py
│   ├── train_bc.py
│   ├── distill.py
│   └── eval_policy.py
├── mujoco_models/
│   └── block_picking.xml      # the only file reused from mujoco_rl/; everything else is from scratch
├── checkpoints/               # saved policies (.pt)
├── logs/
├── pyproject.toml
└── uv.lock
```
