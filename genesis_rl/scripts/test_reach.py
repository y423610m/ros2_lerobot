#!/usr/bin/env python3
"""Test if robot can reach and grasp sponge, then place in container."""
import yaml
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import SO101PickPlaceEnv

def test_scripted_policy():
    """Simple scripted policy to test if task is solvable."""
    with open("config/env_config.yaml", "r") as f:
        env_cfg = yaml.safe_load(f)
    env_cfg["env"]["num_envs"] = 1
    
    env = SO101PickPlaceEnv(env_cfg=env_cfg, device='cuda', show_viewer=False)
    
    print("=== Testing Scripted Policy ===")
    
    for episode in range(5):
        obs = env.reset()
        done = False
        steps = 0
        phase = "reach"  # reach -> grasp -> place
        
        print(f"\nEpisode {episode+1}:")
        
        while not done and steps < 200:
            ee_pos = obs["ee_pos"][0].cpu().numpy()
            sponge_pos = obs["object_pos"][0].cpu().numpy()
            container_pos = obs["target_pos"][0].cpu().numpy()
            
            if phase == "reach":
                # Move toward sponge
                direction = sponge_pos - ee_pos[:3]
                action = torch.tensor([[direction[0]*5, direction[1]*5, direction[2]*5, 0, 0, -0.5]], device='cuda')
                
                dist = np.linalg.norm(direction)
                if dist < 0.05:
                    phase = "grasp"
                    print(f"  Step {steps}: Reached sponge! Dist={dist:.4f}")
            
            elif phase == "grasp":
                # Close gripper
                action = torch.tensor([[0, 0, 0, 0, 0, 0.5]], device='cuda')
                print(f"  Step {steps}: Grasping...")
                phase = "place"
            
            elif phase == "place":
                # Move toward container
                direction = container_pos - ee_pos[:3]
                action = torch.tensor([[direction[0]*5, direction[1]*5, direction[2]*5, 0, 0, 0.5]], device='cuda')
                
                dist = np.linalg.norm(container_pos[:2] - sponge_pos[:2])
                if dist < 0.08:  # Within container
                    print(f"  Step {steps}: Reached container! Dist={dist:.4f}")
                    print(f"  Sponge pos: {sponge_pos}")
                    print(f"  Container pos: {container_pos}")
                    break
            
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if info["success"].item():
                print(f"  SUCCESS at step {steps}!")
                break
        
        if steps >= 200:
            print(f"  FAILED - timeout after 200 steps")
            print(f"  Final sponge pos: {obs['object_pos'][0].cpu().numpy()}")
            print(f"  Final container pos: {obs['target_pos'][0].cpu().numpy()}")

if __name__ == "__main__":
    test_scripted_policy()
