# RL Training with Genesis - TODO

## Goal
Train SO101 robot arm for pick-and-place task using Genesis simulator, then deploy trained model via ROS2.

## Architecture
- **Training**: Standalone Python project (`genesis_rl/`) using pure Genesis + RSL-RL  
  - Uses only `genesis-world` (no genesis-forge dependency)
  - Trained models exported to ONNX for ROS2 deployment
- **Deployment**: ROS2 package (`lerobot_robots_rl/`) for running inference

---

## Status

### ✅ Completed
- [x] `genesis_rl/` project structure created (tasks/, config/, scripts/)
- [x] `tasks/so101_pick_place.py` - Pure Genesis SO101 environment (28-dim obs, 6-dim action including gripper)
- [x] `config/env_config.yaml` - Environment settings
- [x] `config/ppo_config.yaml` - PPO hyperparameters
- [x] `scripts/train.py` - RSL-RL training entry point
- [x] `scripts/eval.py` - Evaluation script
- [x] `scripts/export_policy.py` - ONNX/TorchScript export
- [x] `pyproject.toml` - Dependencies (genesis-world, rsl-rl-lib, torch, tensorboard, numpy, yaml, gymnasium)
- [x] `README.md` - Updated for uv
- [x] Rewrote for pure Genesis (no genesis-forge dependency)

### 🔜 In Progress
- [ ] Run training test
- [ ] Verify environment creation

### 📋 Pending
- [ ] Run full training
- [ ] Export trained model to ONNX
- [ ] Create `src/lerobot_robots_rl/` ROS2 package
- [ ] Implement `rl_inference_node.py`
- [ ] Create ROS2 launch file
- [ ] Test with real/simulated robot

---

## ✅ Correct Setup Procedure (uv)

### 1. Install dependencies
```bash
cd /home/y423610m/ros2_lerobot/genesis_rl
uv sync
```

### 2. Test environment
```bash
uv run python -c "from tasks import SO101PickPlaceEnv; print('Import OK')"
```

### 3. Test environment (small scale)
```bash
uv run python scripts/train.py --num-envs 64 --num-iterations 10
```

### 4. Full training (GPU recommended)
```bash
uv run python scripts/train.py --num-envs 4096 --num-iterations 3000
```

### 5. Export trained model for ROS2 deployment
```bash
uv run python scripts/export_policy.py --model models/so101_pick_place_final.pt --format onnx
```

### Monitor training
```bash
uv run tensorboard --logdir logs/tensorboard
```

---

## Key Configuration

**Observation (28 dim):** joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) + object_pos(3) + object_rel_pos(3) + target_pos(3)
(6th joint is gripper, so no separate gripper_pos observation)

**Action (6 dim):** joint_targets[6] = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] (6th DOF is gripper)

**Episodic Reward:**
- reach_object: 2.0 × (-distance)
- grasp_object: 5.0 × success
- reach_target: 5.0 × (-distance)
- place_object: 10.0 × success
- action_smooth: -0.1 × |Δaction|
- success: 20.0 × done

---

## Next Steps (Immediate)
1. Complete environment creation test
2. Run training
3. Export policy to ONNX
4. Create `src/lerobot_robots_rl/` ROS2 package