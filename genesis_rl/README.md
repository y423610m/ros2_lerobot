# Genesis RL for SO101 Pick and Place

Standalone training project for SO101 robot arm using Genesis simulator and RSL-RL.

## Overview

This project trains an SO101 robot arm to perform pick-and-place tasks using:
- **Genesis**: Physics simulation engine (pure, no genesis-forge)
- **RSL-RL**: PPO implementation for training

## Folder Structure

```
genesis_rl/
├── tasks/
│   ├── __init__.py
│   └── so101_pick_place.py         # Pick-and-place environment
├── config/
│   ├── env_config.yaml              # Environment settings
│   └── ppo_config.yaml             # PPO hyperparameters
├── scripts/
│   ├── train.py                    # Training entry point
│   ├── eval.py                     # Evaluation script
│   └── export_policy.py            # Export for deployment
├── models/                         # Output: trained models
├── logs/                           # TensorBoard logs
├── pyproject.toml                  # Dependencies
└── uv.lock                         # Lock file
```

## Quick Start

### 1. Install Dependencies

**Note**: This project uses `uv` for dependency management.

```bash
cd genesis_rl

# Dependencies are already in pyproject.toml, just sync:
uv sync
```

### 2. Test Environment
```bash
uv run python -c "from tasks import SO101PickPlaceEnv; print('Import OK')"
```

### 3. Train
```bash
# Small test (few iterations)
uv run python scripts/train.py --num-envs 256 --num-iterations 100

# Full training (GPU recommended)
uv run python scripts/train.py --num-envs 4096 --num-iterations 3000
```

### 4. Evaluate
```bash
uv run python scripts/eval.py --num-episodes 3000 --model models/so101_pick_place_final.pt 
```

### 5. Export for ROS2 Deployment
```bash
uv run python scripts/export_policy.py --model models/so101_pick_place_final.pt --format onnx
```

## Configuration

### Environment Config (`config/env_config.yaml`)
- `num_envs`: Parallel environments (4096 for GPU training)
- `episode_length`: Max steps per episode
- `table_height`, `object_size`: Task parameters

### PPO Config (`config/ppo_config.yaml`)
- `learning_rate`: 3.0e-4
- `num_learning_epochs`: 5
- `gamma`: 0.99

## Observation Space (28 dim)
- Joint positions (6) - 6th is gripper
- Joint velocities (6) - 6th is gripper
- End-effector position (3)
- End-effector orientation quaternion (4)
- Object position (3)
- Object relative position to EE (3)
- Target position (3)

## Action Space (6 dim)
- Joint position targets (6): [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
  (6th DOF is the gripper joint)

## Reward Structure
- `reach_object`: Distance to object (weight 2.0)
- `grasp_object`: Successful grasp (weight 5.0)
- `reach_target`: Distance to target (weight 5.0)
- `place_object`: Successful placement (weight 10.0)
- `action_smooth`: Action smoothness (weight -0.1)
- `success`: Task completion (weight 20.0)

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

## Next Steps

After training, deploy to ROS2 using the `lerobot_robots_rl` package (to be created in `src/lerobot_robots_rl/`).