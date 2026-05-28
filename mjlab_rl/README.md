# mjlab_rl

GPU-batched RL task built on [mjlab](https://github.com/mujocolab/mjlab): SO-101
arm picks a block and drops it into a container. The whole stack is the
manager-based env (observations / rewards / events / terminations) plus a PPO
runner config that plugs into mjlab's bundled `rsl-rl` trainer.

> This is a rewrite of the earlier plain-MuJoCo prototype. The old gym-style
> env, scripted oracle, behavior-cloning and distillation scripts have been
> removed; mjlab provides a much faster path (batched MuJoCo Warp + PPO from
> demos-not-needed) to the same end goal.

See `plan.md` for the original design notes.

## What's in here

| What | Where |
|---|---|
| SO-101 EntityCfg (wraps the URDF/MJCF in `src/lerobot_robots_description`) | `mjlab_rl/assets/so101.py` |
| Block + container MjSpec helpers                                         | `mjlab_rl/assets/scene_objects.py` |
| Task-specific MDP terms (reach / lift / place / success)                 | `mjlab_rl/envs/mdp.py` |
| Env + PPO config and task registration                                   | `mjlab_rl/tasks/block_picking.py` |
| Train wrapper (delegates to `mjlab.scripts.train`)                       | `scripts/train.py` |
| Play / eval wrapper                                                      | `scripts/play.py` |
| CPU smoke test (env construction + a few steps)                          | `scripts/smoke_test.py` |

The task registers itself as **`Mjlab-SO101-Block-Picking`** at import time.

## Setup

```bash
cd mjlab_rl
uv sync
```

`uv sync` will pull `mjlab` (with `mujoco-warp`, `rsl-rl-lib`, `torch`, `warp-lang`).
GPU training requires a CUDA-capable Warp install; mjlab also runs on CPU
(slow, single-env) which is what `scripts/smoke_test.py` uses.

## Quick start

```bash
# 1) Confirm the env builds and steps on CPU.
uv run python scripts/smoke_test.py

# 2) List the registered tasks (you should see Mjlab-SO101-Block-Picking).
uv run python -m mjlab.scripts.list_envs

# 3) Train on GPU with 4096 parallel envs.
uv run python scripts/train.py Mjlab-SO101-Block-Picking \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 5000

# 4) Play a trained checkpoint in the viewer.
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --checkpoint-file logs/rsl_rl/so101_block_picking/<run>/model_*.pt
```

## Notes on the env

* **Robot:** SO-101 arm via `XmlActuatorCfg` — the existing `<position>`
  actuators (kp=17.8 from the `sts3215` class) are reused as-is, so action
  scaling and joint-space PD gains match the real robot.
* **Action:** 6-D `JointPositionAction` with `use_default_offset=True`; each
  joint gets its own scale (see `SO101_ACTION_SCALE` in
  `mjlab_rl/assets/so101.py`). The PPO output is added on top of the home
  pose so the policy stays in a sane operating region.
* **Observation groups:**
  * `actor` (24-d, noisy): joint pos/vel + ee→block + block→container + last action
  * `critic` (31-d, clean): same as actor plus the block's world pose (privileged)
* **Reward shape:**
  | term            | weight |
  |-----------------|-------:|
  | reach (gaussian)|   1.0  |
  | lift (binary)   |   2.0  |
  | place (gaussian)|   3.0  |
  | success bonus   |  50.0  |
  | action_rate_l2  |  -0.01 |
  | joint_pos_limits|  -5.0  |
* **End-effector reference:** the flange-fixed `gripperframe` site. This is
  rigidly attached to the wrist and does *not* shift when the single moving
  jaw opens/closes, so the policy's ee-to-block vector stays a clean function
  of arm joints 1–5.
* **Randomization:** on every reset, the block is placed in
  `(x ∈ [0.20, 0.35], y ∈ [-0.10, 0.10])` and the container in
  `(x ∈ [-0.35, -0.20], y ∈ [-0.15, 0.05])`. Robot joints get a small
  ±0.02 rad perturbation.
* **Termination:** time-out after 8 s (= 400 env steps at 50 Hz control),
  early termination if the block falls below `z = -0.05`.

## Directory layout

```
mjlab_rl/
├── mjlab_rl/
│   ├── __init__.py            # registers the task on import
│   ├── assets/
│   │   ├── so101.py
│   │   └── scene_objects.py
│   ├── envs/
│   │   └── mdp.py             # task-specific reward / obs / termination terms
│   └── tasks/
│       └── block_picking.py   # env_cfg, ppo_cfg, register_mjlab_task(...)
├── scripts/
│   ├── train.py
│   ├── play.py
│   └── smoke_test.py
├── plan.md
├── pyproject.toml
└── uv.lock
```
