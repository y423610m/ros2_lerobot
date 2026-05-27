# Robot Arm Policy Training Plan
## Two-Stage Learning: Imitation + Reinforcement Learning

**Project Goal:** Train a policy for a SO101 robotic arm to pick a cube and place it in a container using visual observations from 2 cameras.

**Approach:** 
1. Stage 1 - Behavioral Cloning (BC) from teleoperation demonstrations
2. Stage 2 - Reinforcement Learning (RL) fine-tuning with task rewards

---

## 1. Overview & Architecture

### Problem Statement
- **Robot:** SO101 arm (6-DOF or similar articulated arm)
- **Sensors:** 2 RGB cameras for visual observation
- **Task:** Pick small cube → Place in container
- **Environment:** MuJoCo physics simulation

### Why This Approach Works
- **Stage 1 (BC):** Learns reasonable robot movements quickly from human demonstrations
- **Stage 2 (RL):** Optimizes beyond human performance, learns task-specific strategies
- **Benefit:** Avoids early random exploration, faster convergence, stable training

### Expected Outcomes
- A trained visual-motor policy that generalizes to:
  - Different cube positions
  - Different container positions
  - Potential sim-to-real transfer

---

## 2. Tools & Dependencies

### Primary Framework: **mjlab**
- Lightweight, GPU-accelerated robot learning framework
- **Physics Backend:** MuJoCo Warp (DeepMind + NVIDIA, 2025)
- **Manager API:** Composable building blocks (observations, rewards, events)
- **Performance:** 70-100x speedup for manipulation tasks on GPU
- **Management:** uv, python3.12

### Installation
```bash
# Recommended: Use uv for latest version
uv pip install mjlab

# Or standard pip
pip install mjlab

# GPU support requires:
# - NVIDIA CUDA toolkit
# - mujoco-warp compatible GPU (RTX, A100, etc.)
```

### Additional Dependencies
```
PyTorch (for policy networks)
RSL-RL (mjlab's RL trainer, included)
OpenCV (image preprocessing)
NumPy, PyYAML (configuration)
```

---

## 3. Stage 1: Behavioral Cloning (Imitation Learning)

### 3.1 Data Collection - Teleoperating the Robot

**Setup:**
- Create virtual robot in MuJoCo matching your real arm kinematics
- Implement teleop interface (keyboard, controller, or kinesthetic teaching)
- Record expert demonstrations

**Data Specification:**
```
Each demonstration trajectory contains:
├── Timesteps: 50-200 steps per trajectory
├── Observations:
│   ├── camera_1: RGB image (640x480 or similar)
│   ├── camera_2: RGB image (640x480)
│   └── state: [joint_positions, joint_velocities, gripper_state]
├── Actions:
│   ├── joint_targets: 6D arm joint commands
│   └── gripper_command: binary/continuous grasp signal
└── Metadata: trajectory_id, episode_success, duration

Collect: 50-100 successful pick-and-place trajectories
```

**Implementation in mjlab:**
```python
# Pseudo-structure (adapt to mjlab's ManagerBasedEnv)
class TeleopEnvironment:
    def __init__(self, env_cfg):
        self.env = mjlab.ManagerBasedEnv(cfg=env_cfg)
        self.replay_buffer = []
    
    def record_teleop_trajectory(self):
        """Collect one human demonstration"""
        obs = self.env.reset()
        trajectory = {"obs": [], "actions": [], "rewards": []}
        
        for t in range(max_steps):
            # Get teleop command (keyboard, VR, etc.)
            action = self.get_teleop_command()
            obs, reward, done, info = self.env.step(action)
            
            trajectory["obs"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            
            if done:
                break
        
        self.replay_buffer.append(trajectory)
        return trajectory
    
    def save_demonstrations(self, path):
        """Save all collected data"""
        torch.save(self.replay_buffer, path)
```

### 3.2 Behavioral Cloning Training

**Policy Network Architecture:**
```
Input: [camera_1_image, camera_2_image, joint_state]
        ↓
    CNN encoders (separate for each camera)
        ↓
    Concatenate features
        ↓
    MLP: 256 → 256 → 128 hidden units
        ↓
Output: [joint_targets (6D), gripper_command (1D)]

Loss: MSE(predicted_actions, expert_actions)
```

**Training Configuration:**
```yaml
# bc_config.yaml
behavioral_cloning:
  learning_rate: 1e-3
  batch_size: 32
  num_epochs: 100
  validation_split: 0.2
  early_stopping_patience: 10
  
  network:
    cnn_channels: [32, 64, 128]
    mlp_hidden: [256, 256, 128]
    dropout: 0.1
  
  data_augmentation:
    random_crop: true
    color_jitter: true
    gaussian_noise_images: 0.01
```

**Expected Metrics:**
- Action MSE: < 0.1 (tuned for your action scale)
- Success Rate (validation): > 80% picking and placing
- Training time: 1-3 hours on single GPU

**Checkpointing:**
```python
# Save best BC policy
torch.save({
    'policy_state_dict': bc_policy.state_dict(),
    'config': bc_config,
    'metrics': {'val_mse': 0.08, 'success_rate': 0.85}
}, 'checkpoints/bc_best.pth')
```

---

## 4. Stage 2: Reinforcement Learning Fine-tuning

### 4.1 Task Definition & Reward Function

**Reward Structure:**
```python
def compute_reward(state, action):
    """
    Reward composition for pick-and-place task
    """
    
    # 1. Reaching reward (encourage approach to cube)
    cube_pos = state['cube_position']
    gripper_pos = state['end_effector_position']
    reach_dist = torch.norm(gripper_pos - cube_pos)
    reach_reward = -reach_dist  # Negative distance (reward proximity)
    
    # 2. Grasp reward (encourage closing gripper near cube)
    if reach_dist < 0.05:  # Within grasp threshold
        grasp_reward = -reach_dist * 10 + state['gripper_force'] * 0.1
    else:
        grasp_reward = 0
    
    # 3. Lift reward (encourage lifting cube)
    if state['cube_grasped']:
        cube_height_gain = state['cube_height'] - initial_cube_height
        lift_reward = cube_height_gain * 5
    else:
        lift_reward = 0
    
    # 4. Place reward (encourage moving to container)
    if state['cube_grasped']:
        container_pos = state['container_position']
        place_dist = torch.norm(gripper_pos - container_pos)
        place_reward = -place_dist * 2
    else:
        place_reward = 0
    
    # 5. Success bonus (task completion)
    success_bonus = 0
    if state['cube_in_container']:
        success_bonus = 10.0
    
    # Total reward
    total = (reach_reward + 
             grasp_reward + 
             lift_reward + 
             place_reward + 
             success_bonus)
    
    return total
```

### 4.2 RL Training Configuration

**Algorithm:** PPO (Proximal Policy Optimization)
- Stable, sample-efficient
- Supported natively by mjlab (via RSL-RL)

**Training Hyperparameters:**
```yaml
# rl_config.yaml
reinforcement_learning:
  algorithm: PPO
  
  # Training
  num_envs: 4096  # Parallel environments on GPU
  num_steps_per_episode: 200
  total_training_steps: 10_000_000  # ~100k episodes
  
  # PPO specifics
  learning_rate: 5e-4
  num_epochs: 4
  batch_size: 256
  clip_range: 0.2
  entropy_coeff: 0.01
  value_loss_coeff: 1.0
  
  # Initialization
  init_from_bc: true
  bc_checkpoint_path: 'checkpoints/bc_best.pth'
  
  # Exploration
  action_std_init: 0.3
  action_std_final: 0.1
  action_std_decay_steps: 5_000_000
  
  # Domain randomization (for sim-to-real)
  randomize_cube_position: true
  randomize_cube_properties: true
  randomize_camera_pose: true
  randomize_lighting: true
```

**Training Loop:**
```python
class RLTrainer:
    def __init__(self, env, policy, rl_config):
        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.Adam(
            policy.parameters(), 
            lr=rl_config.learning_rate
        )
        self.ppo_loss = PPOLoss(clip_range=rl_config.clip_range)
    
    def train(self):
        """Main RL training loop"""
        obs = self.env.reset()
        
        for step in range(total_steps):
            # Collect rollout
            with torch.no_grad():
                actions, log_probs, values = self.policy(obs)
            
            obs_next, rewards, dones, info = self.env.step(actions)
            
            # Compute advantages
            advantages = self.compute_gae(rewards, values)
            
            # PPO update
            for epoch in range(num_epochs):
                loss = self.ppo_loss(
                    actions, log_probs, advantages, values
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Logging & checkpointing
            if step % log_interval == 0:
                success_rate = info['success_rate']
                avg_reward = info['episode_reward'].mean()
                print(f"Step {step}: Success={success_rate:.2%}, Reward={avg_reward:.2f}")
            
            if step % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoints/rl_step_{step}.pth")
            
            obs = obs_next
```

### 4.3 Curriculum & Skill Progression

**Optional: Curriculum Learning**
```yaml
curriculum:
  enabled: true
  phases:
    # Phase 1: Learn reaching without grasping
    - steps: 2_000_000
      mask_grasp_reward: true
      mask_place_reward: true
      randomization_level: 0.2
    
    # Phase 2: Add grasping
    - steps: 3_000_000
      mask_grasp_reward: false
      mask_place_reward: true
      randomization_level: 0.5
    
    # Phase 3: Full task with high randomization
    - steps: 5_000_000
      mask_grasp_reward: false
      mask_place_reward: false
      randomization_level: 1.0
```

---

## 5. Technical Implementation Details

### 5.1 MJCF Model Structure

**Your SO101 Arm Model:**
```xml
<!-- arm.xml -->
<mujoco model="so3_arm">
  <option gravity="0 0 -9.81"/>
  
  <default>
    <joint damping="1"/>
    <geom friction="1 0.1 0.1"/>
  </default>
  
  <asset>
    <!-- Your arm meshes -->
    <mesh name="base" file="meshes/base.stl"/>
    <mesh name="link1" file="meshes/link1.stl"/>
    <!-- ... -->
  </asset>
  
  <worldbody>
    <!-- Base -->
    <body name="base" pos="0 0 0">
      <geom mesh="base" type="mesh"/>
      <joint name="joint_1" type="hinge" axis="0 0 1"/>
      
      <!-- Link 1 -->
      <body name="link_1" pos="0 0 0.1">
        <geom mesh="link1" type="mesh"/>
        <joint name="joint_2" type="hinge" axis="0 1 0"/>
        <!-- ... more joints ... -->
      </body>
    </body>
    
    <!-- End effector / gripper -->
    <body name="gripper" pos="...">
      <joint name="gripper_joint" type="slide"/>
      <!-- Finger geometries -->
    </body>
    
    <!-- Cube (object to manipulate) -->
    <body name="cube" pos="0.3 0 0.05">
      <inertial mass="0.05" diaginv="1 1 1"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"/>
      <freejoint name="cube_free"/>
    </body>
    
    <!-- Container -->
    <body name="container" pos="0.3 0.2 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0 0 1 0.3"/>
    </body>
  </worldbody>
  
  <actuator>
    <!-- Joint actuators (motors) -->
    <motor name="motor_1" joint="joint_1" ctrlrange="-1 1"/>
    <motor name="motor_2" joint="joint_2" ctrlrange="-1 1"/>
    <!-- ... -->
    <motor name="gripper_motor" joint="gripper_joint" ctrlrange="0 1"/>
  </actuator>
</mujoco>
```

### 5.2 Observation & Action Space

**Observation Dictionary:**
```python
observation = {
    'camera_1': torch.FloatTensor(B, 3, 480, 640),  # RGB image
    'camera_2': torch.FloatTensor(B, 3, 480, 640),  # RGB image
    'joint_pos': torch.FloatTensor(B, 6),           # Arm joint angles
    'joint_vel': torch.FloatTensor(B, 6),           # Joint velocities
    'gripper_state': torch.FloatTensor(B, 2),       # [position, force]
    'ee_pos': torch.FloatTensor(B, 3),              # End-effector position
    'ee_rot': torch.FloatTensor(B, 9),              # End-effector rotation (flattened)
    'cube_pos': torch.FloatTensor(B, 3),            # Cube position
    'cube_rot': torch.FloatTensor(B, 9),            # Cube rotation
    'container_pos': torch.FloatTensor(B, 3),       # Container position
}

# Batch size B = number of parallel environments (e.g., 4096)
```

**Action Space:**
```python
action = {
    'joint_targets': torch.FloatTensor(B, 6),  # [-1, 1] normalized
    'gripper_cmd': torch.FloatTensor(B, 1),    # [0, 1] (0=open, 1=closed)
}

# Total action dim = 7 (6 arm + 1 gripper)
```

### 5.3 Camera Configuration

**Camera Setup in MJCF:**
```xml
<camera name="camera_1" mode="renderoffscreen" pos="0.4 -0.4 0.3" xyaxes="1 0 0 0 1 1">
  <frame pixelwidth="640" pixelheight="480"/>
</camera>

<camera name="camera_2" mode="renderoffscreen" pos="0.4 0.4 0.3" xyaxes="1 0 0 0 1 1">
  <frame pixelwidth="640" pixelheight="480"/>
</camera>
```

**Rendering in mjlab:**
```python
# mjlab automatically handles multi-camera rendering
obs = env.reset()
rgb_1 = obs['camera_1']  # Already rendered
rgb_2 = obs['camera_2']
```

---

## 6. Implementation Checklist

### Pre-Training
- [ ] Create MJCF model for SO101 arm with accurate kinematics
- [ ] Validate physics in MuJoCo (check stability, contact dynamics)
- [ ] Implement teleop interface (keyboard/gamepad/VR)
- [ ] Create data collection pipeline
- [ ] Design camera placement for good task visibility
- [ ] Test observation/action spaces

### Stage 1: Behavioral Cloning
- [ ] Collect 50-100 teleop demonstrations
- [ ] Implement BC network (CNN + MLP)
- [ ] Train BC policy (target: >80% success rate)
- [ ] Save BC checkpoint
- [ ] Log BC training metrics (loss, success rate)

### Stage 2: RL Fine-tuning
- [ ] Define reward function (test individual components)
- [ ] Create mjlab environment configuration
- [ ] Initialize RL policy from BC checkpoint
- [ ] Run RL training on GPU (4+ parallel envs)
- [ ] Monitor training metrics (episode reward, success rate)
- [ ] Save checkpoints every 500k steps
- [ ] Evaluate policy on held-out test scenarios

### Post-Training
- [ ] Generate evaluation videos
- [ ] Test generalization (different cube/container positions)
- [ ] Prepare policy for sim-to-real transfer (if applicable)
- [ ] Document final metrics and hyperparameters

---

## 7. Expected Results & Timeline

### Benchmark Milestones
```
Behavioral Cloning (Stage 1):
├── Training time: 1-3 hours (single GPU)
├── Success rate: 80-90%
└── Policy size: ~10-50 MB

RL Fine-tuning (Stage 2):
├── Training time: 8-24 hours (GPU, 4096 envs)
├── Success rate: 95-99%
├── Final reward: >8.0 (per episode)
└── Convergence: ~5-10M steps
```

### Generalization Tests
- **Cube position variance:** ±10cm in x-y
- **Container position variance:** ±10cm in x-y
- **Object property variance:** ±20% mass/friction
- **Camera perturbations:** ±5° pose noise

**Expected success on generalization:** >90%

---

## 8. Debugging & Troubleshooting

### Common Issues

**BC Training Not Converging:**
- Check observation preprocessing (normalize images to [0,1])
- Verify action space scaling matches teleop commands
- Increase network capacity or training duration

**RL Reward Plateau:**
- Adjust reward weights (reach vs. grasp vs. place)
- Verify cube is actually being grasped (check grasp_reward condition)
- Enable curriculum learning

**Poor Generalization:**
- Increase domain randomization levels
- Collect more diverse demonstrations
- Add noise to camera observations during training

**GPU Memory Issues:**
- Reduce `num_envs` (try 512 or 1024)
- Reduce image resolution (e.g., 320x240)
- Use gradient accumulation

---

## 9. File Structure & Organization

```
robot_arm_project/
├── README.md
├── configs/
│   ├── arm_model.yaml          # mjlab env config
│   ├── bc_config.yaml          # BC training config
│   └── rl_config.yaml          # RL training config
├── mjcf/
│   ├── arm.xml                 # Your SO101 arm model
│   ├── meshes/                 # Mesh files
│   └── materials.xml           # Material definitions
├── scripts/
│   ├── 01_collect_teleop.py    # Data collection
│   ├── 02_train_bc.py          # BC training
│   ├── 03_train_rl.py          # RL fine-tuning
│   ├── 04_evaluate.py          # Policy evaluation
│   └── utils.py                # Helper functions
├── checkpoints/
│   ├── bc_best.pth             # Best BC policy
│   ├── rl_step_5000000.pth     # Best RL policy
│   └── metrics.json            # Training metrics
├── data/
│   └── demonstrations/         # Teleop data
└── outputs/
    ├── videos/                 # Evaluation videos
    └── results.json            # Final metrics
```

---

## 10. References & Resources

### Key Papers
- **Behavioral Cloning:** Pomerleau, D. A. (1989). "ALVINN: An autonomous land vehicle in a neural network"
- **PPO:** Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
- **Motion Imitation + RL:** Peng et al. (2021). "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills"

### Official Documentation
- **mjlab:** https://mujocolab.github.io/mjlab/
- **MuJoCo:** https://mujoco.org/
- **MuJoCo Warp:** https://github.com/google-deepmind/mujoco_warp
- **RSL-RL:** https://rsl-rl.readthedocs.io/

### Related Projects
- MuJoCo Playground: Quick prototyping without heavy frameworks
- Isaac Lab: More comprehensive but heavier (Isaac Sim required)
- DeepMimic: Motion imitation using phase-functionalized neural networks

---

## 11. Next Steps

1. **Immediate (Week 1):**
   - Set up mjlab environment
   - Create MJCF model for your arm
   - Implement teleop interface

2. **Short-term (Weeks 2-3):**
   - Collect demonstrations (50-100 trajectories)
   - Train BC policy
   - Achieve >80% success rate on validation

3. **Medium-term (Weeks 4-6):**
   - Configure reward function
   - Run RL training (10M+ steps)
   - Achieve >95% success on test scenarios

4. **Long-term (Weeks 7+):**
   - Generalization testing
   - Sim-to-real preparation (if applicable)
   - Documentation & paper writing

---

**Document Version:** 1.0  
**Last Updated:** 2026-05-27  
**Author Notes:** This plan assumes a single high-end GPU (RTX 4090, A100, etc.). Adjust `num_envs` and training duration for different hardware. Questions? Refer to mjlab documentation or community forums.
