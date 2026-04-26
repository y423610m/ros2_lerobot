#!/usr/bin/env python3
"""
Evaluation script for trained SO101 pick and place policy
"""
import os
import sys
import yaml
import argparse
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import SO101PickPlaceEnv
from rsl_rl.runners import OnPolicyRunner


def evaluate(model_path: str, num_episodes: int = 10, show_viewer: bool = True):
    """Evaluate a trained policy."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configs
    with open("config/env_config.yaml", "r") as f:
        env_cfg = yaml.safe_load(f)
    with open("config/ppo_config.yaml", "r") as f:
        ppo_cfg = yaml.safe_load(f)

    # Override num_envs to 1 for evaluation
    env_cfg["env"]["num_envs"] = 1

    # Create environment with visualization
    env = SO101PickPlaceEnv(
        env_cfg=env_cfg,
        device=device,
        show_viewer=show_viewer,
    )

    # import IPython
    # IPython.embed()

    # Load model
    experiment_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("logs", experiment_name)

    runner = OnPolicyRunner(
        env,
        train_cfg=ppo_cfg,
        log_dir=log_dir,
        device=device,
    )

    runner.load(model_path)

    # Run evaluation episodes
    successes = 0
    total_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        import IPython
        IPython.embed()

        while not done:
            with torch.no_grad():
                actions = runner.get_inference_policy()(obs)
                print(actions)
            obs, rewards, dones, info = env.step(actions)
            episode_reward += rewards.item()
            steps += 1

            if info["success"].item():
                successes += 1
                print(f"Episode {ep+1}: SUCCESS in {steps} steps!")
                break

            if dones.item():
                print(f"Episode {ep+1}: FAILED after {steps} steps")
                break

        total_rewards.append(episode_reward)

    print(f"\n=== Evaluation Results ===")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SO101 pick and place policy")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--show_viewer", action="store_true", default=True)
    args = parser.parse_args()

    evaluate(args.model, args.num_episodes, args.show_viewer)


if __name__ == "__main__":
    main()
