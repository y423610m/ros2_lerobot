# block-picking-rl

Reinforcement learning environment for training a **SO-101 robot arm** (5-DOF arm + 1 gripper)
to pick a block from a table and place it on a target zone.

Built with **MuJoCo 3**, **Gymnasium**, **Stable-Baselines3 (SAC)**, and **uv**.

---

## Project structure

```
block_picking_rl/
├── mujoco_models/
│   └── block_picking.xml      # MuJoCo scene: SO-101 arm, table, block, target
├── envs/
│   ├── __init__.py
│   └── block_picking_env.py   # Gymnasium environment (MujocoEnv subclass)
├── scripts/
│   └── test_env.py            # Smoke test with random actions
├── train_sac.py               # SAC training + evaluation entry point
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install dependencies
uv sync
```

---

## Usage

### Smoke test

Verify the environment loads and steps without error:

```bash
uv run python scripts/test_env.py

# With MuJoCo viewer
uv run python scripts/test_env.py --render --episodes 1
```

### Train (SAC)

```bash
# Full run (2M steps, 4 parallel envs)
uv run python train_sac.py

# Quick sanity check
uv run python train_sac.py --timesteps 100000

# Visualize one env during training (slower)
uv run python train_sac.py --render

# Monitor with TensorBoard
tensorboard --logdir logs/
```

### Evaluate a checkpoint

```bash
uv run python train_sac.py --eval --checkpoint checkpoints/best_model
```

---

## Environment spec

### Robot

SO-101: 5 revolute joints + 1 parallel gripper.

| Actuator   | Joint       | Axis | Range (rad)    |
|------------|-------------|------|----------------|
| act_j1     | base yaw    | Z    | ±π             |
| act_j2     | shoulder    | Y    | ±1.57          |
| act_j3     | elbow       | Y    | ±2.0           |
| act_j4     | wrist pitch | Y    | ±1.57          |
| act_j5     | wrist roll  | X    | ±π             |
| gripper    | fingers     | —    | open / close   |

### Observation (30-dim, `float32`)

| Field           | Dim | Description                           |
|-----------------|-----|---------------------------------------|
| `joint_pos`     | 5   | Joint angles [rad]                    |
| `joint_vel`     | 5   | Joint velocities [rad/s]              |
| `ee_pos`        | 3   | End-effector position [m]             |
| `ee_quat`       | 4   | End-effector orientation (quaternion) |
| `block_pos`     | 3   | Block position [m]                    |
| `block_vel`     | 3   | Block linear velocity [m/s]           |
| `ee_to_block`   | 3   | Vector from EE to block [m]           |
| `block_to_tgt`  | 3   | Vector from block to target [m]       |
| `gripper_open`  | 1   | Normalized aperture [0=closed, 1=open]|

### Action (6-dim, `float32`, `[-1, 1]`)

| Index | Description                                  |
|-------|----------------------------------------------|
| 0–4   | Torque commands for joint1..joint5           |
| 5     | Gripper: −1 = open, +1 = close              |

### Dense reward

| Component       | Weight | Condition                          |
|-----------------|--------|------------------------------------|
| `reach`         | −1.0×d | EE–block distance (always)         |
| `grasp`         | +3.0   | EE near block and gripper closed   |
| `lift`          | +5.0   | Block raised > 5 cm above table    |
| `transport`     | −4.0×d | Block–target distance (when lifted)|
| `place`         | +10.0  | Block within 4 cm of target        |
| `success`       | +50.0  | Episode success bonus              |
| `alive`         | +0.1   | Every step                         |
| `ctrl_penalty`  | −0.01  | Per unit of squared joint torque   |

Success = block within 4 cm of target AND resting on surface.

---

## Connecting to LeRobot

After training, deploy the policy to the real SO-101:

```python
from stable_baselines3 import SAC

model = SAC.load("checkpoints/best_model")

while True:
    obs = build_obs_from_robot(robot)   # match the 30-dim obs spec above
    action, _ = model.predict(obs, deterministic=True)
    robot.send_action(action)           # map back to joint torques + gripper
```

> **Sim-to-real gap**: consider adding domain randomization (friction, mass,
> sensor noise) to `BlockPickingEnv` before real-robot deployment.

---

## Next steps

- **HER** – use `HerReplayBuffer` with a `GoalEnv` wrapper for sparse-reward training
- **TD-MPC2** – model-based RL for better sample efficiency (LeRobot integration)
- **HiL-SERL** – bootstrap from human demonstrations then fine-tune with RL
- **Domain randomization** – randomize block mass, friction, table height for robust transfer
