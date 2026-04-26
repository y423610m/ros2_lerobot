#!/usr/bin/env python3
"""Test if robot can complete pick-and-place task."""
import yaml
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import SO101PickPlaceEnv

def test_task_completion():
    """Test if task is solvable with scripted policy."""
    with open("config/env_config.yaml", "r") as f:
        env_cfg = yaml.safe_load(f)
    env_cfg["env"]["num_envs"] = 1
    
    env = SO101PickPlaceEnv(env_cfg=env_cfg, device='cuda', show_viewer=False)
    
    print("=== Testing Task Completion ===")
    
    for episode in range(3):
        obs = env.reset()
        done = False
        steps = 0
        
        print(f"\nEpisode {episode+1}:")
        print(f"  Sponge pos: {obs['object_pos'][0].cpu().numpy()}")
        print(f"  Container pos: {obs['target_pos'][0].cpu().numpy()}")
        
        # Phase 1: Reach sponge
        print("  Phase 1: Reaching sponge...")
        for i in range(50):
            # Simple proportional control toward sponge
            ee_pos = obs["ee_pos"][0].cpu().numpy()[:3]
            obj_pos = obs["object_pos"][0].cpu().numpy()
            direction = obj_pos - ee_pos
            
            action = torch.zeros(1, 6, device='cuda')
            action[0, 0] = float(direction[0]) * 5.0
            action[0, 1] = float(direction[1]) * 5. 0
            action[0, 2] = float(direction[2]) * 5. 0
            action[0, 5] = -0. 5  # Close gripper
            
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if done.item():
                print(f"    Episode ended at step {steps}!")
                break
        
        if done.item():
            continue
            
        # Phase 2: Try to grasp (close gripper)
        print("  Phase 2: Grasping...")
        for i in range(10):
            action = torch.zeros(1, 6, device='cuda')
            action[0, 5] = 0.5  # Open gripper
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if done.item():
                break
        
        print(f"  Episode {episode+1} ended after {steps} steps")
        print(f"  Final sponge pos: {obs['object_pos'][0].cpu().numpy()}")
        print(f"  Final container pos: {obs['target_pos'][0].cpu().numpy()}")
        print(f"  Success: {info['success'].item()}")

if __name__ == "__main__":
    test_task_completion()
