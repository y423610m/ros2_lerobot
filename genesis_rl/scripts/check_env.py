#!/usr/bin/env python3
"""
Environment Inspection Script for SO101 Pick and Place

Inspect Genesis RL environment state:
- Robot model and joint states
- Object and target poses
- Configuration values

Usage:
    cd genesis_rl && uv run python scripts/check_env.py
    cd genesis_rl && uv run python scripts/check_env.py --seed 123
"""
import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import SO101PickPlaceEnv


def print_separator(title: str):
    """Print formatted section separator."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def quaternion_to_euler(q: np.ndarray) -> tuple:
    """Convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    # Roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    # Yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    parser = argparse.ArgumentParser(description="Inspect SO101 Pick and Place environment")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible reset"
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel envs (default: 1)"
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        default=False,
        help="Show viewer (default: False)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps to run for visualization (default: 100)",
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"\n{'#' * 60}")
    print(f" GENESIS RL ENVIRONMENT INSPECTOR")
    print(f" Seed: {args.seed} | Num Envs: {args.num_envs} | Show Viewer: {args.show_viewer}")
    print(f"{'#' * 60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create environment
    print("\nInitializing environment...")
    env = SO101PickPlaceEnv(
        num_envs=args.num_envs,
        env_spacing=2.0,
        episode_length=200,
        device=device,
        headless=not args.show_viewer,  # headless is opposite of show_viewer
        show_viewer=args.show_viewer,
    )

    # =========================================================================
    # 1. CONFIGURATION VALUES (from class, hardcoded)
    # =========================================================================
    print_separator("CONFIGURATION VALUES")
    print("\n[Scene]")
    print(f"  num_envs:           {env.num_envs}")
    print(f"  episode_length:     {env.episode_length}")

    print("\n[Robot]")
    print(f"  URDF path:          {env.robot_path}")
    print(f"  num_joints:         {env.num_joints}")
    print(f"  joint_names:        {env.joint_names}")

    print("\n[Task Parameters]")
    print(f"  table_height:       {env.table_height} m")
    print(f"  object_size:        {env.object_size} m (cube)")
    print(f"  success_threshold:  {env.success_threshold} m")
    print(f"  grasp_threshold:    {env.grasp_threshold} m")

    print("\n[Space Dimensions]")
    print(f"  obs_dim:            {env.obs_dim}")
    print(f"  action_dim:         {env.action_dim}")

    # =========================================================================
    # 2. ROBOT JOINT STATES
    # =========================================================================
    print_separator("ROBOT JOINT STATES")

    # Reset to get initial state
    env.reset()

    joint_pos = env.robot_entity.get_dofs_position(env.joint_indices)
    joint_vel = env.robot_entity.get_dofs_velocity(env.joint_indices)

    print("\n[Joint Positions & Velocities]")
    for i, name in enumerate(env.joint_names):
        pos = joint_pos[0, i].item()
        vel = joint_vel[0, i].item()
        print(f"  {name:15s}: pos = {pos:8.4f} rad, vel = {vel:8.4f} rad/s")

    # =========================================================================
    # 3. END-EFFECTOR POSE
    # =========================================================================
    print_separator("END-EFFECTOR POSE")

    # Try direct link access (try gripper first, then gripper_link)
    ee_pos_from_link = None
    ee_quat_from_link = None
    try:
        ee_link = env.robot_entity.get_link("gripper")
        ee_pos_from_link = ee_link.get_pos()
        ee_quat_from_link = ee_link.get_quat()
        print("\n[Direct Link Access: gripper]")
        print(f"  Position:  ({ee_pos_from_link[0, 0]:.4f}, {ee_pos_from_link[0, 1]:.4f}, {ee_pos_from_link[0, 2]:.4f}) m")
        print(f"  Quaternion: ({ee_quat_from_link[0, 0]:.4f}, {ee_quat_from_link[0, 1]:.4f}, {ee_quat_from_link[0, 2]:.4f}, {ee_quat_from_link[0, 3]:.4f})")
        # Euler angles
        q = ee_quat_from_link[0].cpu().numpy()
        roll, pitch, yaw = quaternion_to_euler(q)
        print(f"  Euler:     (roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}) rad")
    except Exception as e:
        try:
            ee_link = env.robot_entity.get_link("gripper_link")
            ee_pos_from_link = ee_link.get_pos()
            ee_quat_from_link = ee_link.get_quat()
            print("\n[Direct Link Access: gripper_link]")
            print(f"  Position:  ({ee_pos_from_link[0, 0]:.4f}, {ee_pos_from_link[0, 1]:.4f}, {ee_pos_from_link[0, 2]:.4f}) m")
            print(f"  Quaternion: ({ee_quat_from_link[0, 0]:.4f}, {ee_quat_from_link[0, 1]:.4f}, {ee_quat_from_link[0, 2]:.4f}, {ee_quat_from_link[0, 3]:.4f})")
            # Euler angles
            q = ee_quat_from_link[0].cpu().numpy()
            roll, pitch, yaw = quaternion_to_euler(q)
            print(f"  Euler:     (roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}) rad")
        except Exception as e2:
            print(f"\n  Note: Neither gripper nor gripper_link found ({e2})")

    # Via observations
    obs = env._compute_observations()
    print("\n[Via Observations]")
    # Debug tensor shapes
    print(f"  ee_pos shape: {obs['ee_pos'].shape}")
    print(f"  ee_quat shape: {obs['ee_quat'].shape}")
    # Handle different possible shapes
    ee_pos_tensor = obs["ee_pos"]
    ee_quat_tensor = obs["ee_quat"]
    if len(ee_pos_tensor.shape) == 2:
        ee_pos_obs = ee_pos_tensor[0]  # [1, N] -> [N]
    else:
        ee_pos_obs = ee_pos_tensor     # [N] - use as-is
    if len(ee_quat_tensor.shape) == 2:
        ee_quat_obs = ee_quat_tensor[0]  # [1, 4] -> [4]
    else:
        ee_quat_obs = ee_quat_tensor     # [4] - use as-is
    
    # Print EE position (the ee_pos from _compute_observations seems to only return gripper joint pos)
    # We'll use the direct link access for full position, but show what obs provides
    if len(ee_pos_obs) >= 3:
        print(f"  ee_pos (from obs):  ({ee_pos_obs[0]:.4f}, {ee_pos_obs[1]:.4f}, {ee_pos_obs[2]:.4f}) m")
    elif len(ee_pos_obs) == 1:
        print(f"  ee_pos (from obs):  ({ee_pos_obs[0]:.4f}, 0.0000, 0.0000) m (NOTE: obs ee_pos only provides gripper joint position)")
    else:
        print(f"  ee_pos (from obs):  (0.0000, 0.0000, 0.0000) m (invalid)")
        
    # Print EE quaternion
    if len(ee_quat_obs) >= 4:
        print(f"  ee_quat (from obs): ({ee_quat_obs[0]:.4f}, {ee_quat_obs[1]:.4f}, {ee_quat_obs[2]:.4f}, {ee_quat_obs[3]:.4f})")
        # Euler angles
        if len(ee_quat_obs) >= 4:
            q = ee_quat_obs.cpu().numpy()
            roll, pitch, yaw = quaternion_to_euler(q)
            print(f"  Euler (from obs):     (roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}) rad")
    else:
        print(f"  ee_quat (from obs): (1.0000, 0.0000, 0.0000, 0.0000) (invalid)")

    # =========================================================================
    # 4. OBJECT AND TARGET POSITIONS
    # =========================================================================
    print_separator("OBJECT AND TARGET POSITIONS")

    obj_pos = env.object.get_pos()
    obj_quat = env.object.get_quat()
    tgt_pos = env.target.get_pos()
    tgt_quat = env.target.get_quat()

    print("\n[Object (Cube)]")
    print(f"  Position:  ({obj_pos[0, 0]:.4f}, {obj_pos[0, 1]:.4f}, {obj_pos[0, 2]:.4f}) m")
    print(f"  Quaternion: ({obj_quat[0, 0]:.4f}, {obj_quat[0, 1]:.4f}, {obj_quat[0, 2]:.4f}, {obj_quat[0, 3]:.4f})")
    print(f"  Size:      {env.object_size} m (cube, volume: {env.object_size**3:.6f} m³)")
    print(f"  On table:  z = table_height + size/2 = {env.table_height} + {env.object_size/2} = {env.table_height + env.object_size/2} m")

    print("\n[Target (Cube)]")
    print(f"  Position:  ({tgt_pos[0, 0]:.4f}, {tgt_pos[0, 1]:.4f}, {tgt_pos[0, 2]:.4f}) m")
    print(f"  Quaternion: ({tgt_quat[0, 0]:.4f}, {tgt_quat[0, 1]:.4f}, {tgt_quat[0, 2]:.4f}, {tgt_quat[0, 3]:.4f})")
    print(f"  Size:      {env.object_size * 1.5} m (1.5x object size)")

    # Distance from object to target
    dist = torch.norm(tgt_pos - obj_pos, dim=-1)
    print(f"\n[Distance: Object to Target: {dist[0].item():.4f} m]")

    # =========================================================================
    # 5. FULL OBSERVATION DICTIONARY
    # =========================================================================
    print_separator("OBSERVATION SPACE (28 dim)")

    print("\n[All Observations - env_id=0]")
    obs_keys = [
        ("joint_pos", 6),      # 6
        ("joint_vel", 6),      # 6
        ("ee_pos", 3),          # 3
        ("ee_quat", 4),         # 4
        ("object_pos", 3),      # 3
        ("object_rel_pos", 3),  # 3 (object_pos - ee_pos)
        ("target_pos", 3),     # 3
    ]
    for key, dim in obs_keys:
        vals = obs[key][0].cpu().numpy()
        formatted = ", ".join([f"{v:.4f}" for v in vals])
        print(f"  {key:17s} ({dim:2d}): [{formatted}]")

    # =========================================================================
    # 6. SIMULATION STEP DEMONSTRATION
    # =========================================================================
    print_separator("SIMULATION STEP DEMONSTRATION")

    # Record state before
    ee_before_raw = env.robot_entity.get_link("gripper").get_pos() if hasattr(env.robot_entity, 'get_link') and env.robot_entity.get_link("gripper") is not None else obs["ee_pos"]
    obj_before = env.object.get_pos()
    # Handle tensor shapes for EE pos
    ee_before = ee_before_raw[0] if len(ee_before_raw.shape) == 2 else ee_before_raw
    obj_before = obj_before[0] if len(obj_before.shape) == 2 else obj_before

    # Take step with zero action (hold position)
    zero_action = torch.zeros((args.num_envs, env.action_dim), device=device)
    obs, rewards, dones, info = env.step(zero_action)

    ee_after_raw = env.robot_entity.get_link("gripper").get_pos() if hasattr(env.robot_entity, 'get_link') and env.robot_entity.get_link("gripper") is not None else obs["ee_pos"]
    obj_after = env.object.get_pos()
    # Handle tensor shapes for EE pos
    ee_after = ee_after_raw[0] if len(ee_after_raw.shape) == 2 else ee_after_raw
    obj_after = obj_after[0] if len(obj_after.shape) == 2 else obj_after

    print(f"\n[Step Results - env_id=0]")
    print(f"  Reward:     {rewards[0].item():.4f}")
    print(f"  Done:       {dones[0].item()}")
    print(f"  Success:    {info['success'][0].item()}")
    # Print shapes for debugging
    print(f"  ee_before shape: {ee_before.shape}, ee_after shape: {ee_after.shape}")
    print(f"  obj_before shape: {obj_before.shape}, obj_after shape: {obj_after.shape}")
    print(f"  EE pos:     ({ee_before[0]:.4f} -> {ee_after[0]:.4f}, {ee_before[1]:.4f} -> {ee_after[1]:.4f}, {ee_before[2]:.4f} -> {ee_after[2]:.4f})")
    print(f"  Obj pos:    ({obj_before[0]:.4f} -> {obj_after[0]:.4f}, {obj_before[1]:.4f} -> {obj_after[1]:.4f}, {obj_before[2]:.4f} -> {obj_after[2]:.4f})")

    # =========================================================================
    # 7. RANDOMIZATION INFO
    # =========================================================================
    print_separator("RESET RANDOMIZATION (from reset())")

    print("\n[Object Randomization]")
    print("  x: rand() * 0.3 - 0.15  => [-0.15, 0.15] m")
    print("  y: rand() * 0.3 - 0.15  => [-0.15, 0.15] m")
    print("  z: table_height + object_size/2 (fixed)")

    print("\n[Target Randomization]")
    print("  x: rand() * 0.3 - 0.15  => [-0.15, 0.15] m")
    print("  y: rand() * 0.3 - 0.15 + 0.3  => [0.15, 0.45] m")
    print("  z: table_height + object_size/2 (fixed)")

    print_separator("INSPECTION COMPLETE")
    print("\nEnvironment state inspection finished successfully!")
    print(f"\nNext: Run 'uv run python scripts/train.py' to start RL training")
    
    # Run additional steps for visualization if show_viewer is True and steps > 0
    if args.show_viewer and args.steps > 0:
        print(f"\nRunning {args.steps} additional simulation steps for visualization...")
        print("Close the viewer window to exit.")
        try:
            # Run simulation with zero actions (hold position)
            zero_action = torch.zeros((args.num_envs, env.action_dim), device=device)
            for i in range(args.steps):
                obs, rewards, dones, info = env.step(zero_action)
                # Small delay to make visualization smoother
                import time
                time.sleep(0.01)
                if i % 20 == 0:
                    print(f"  Step {i}/{args.steps}")
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user.")
        except Exception as e:
            print(f"\nVisualization ended: {e}")


if __name__ == "__main__":
    main()