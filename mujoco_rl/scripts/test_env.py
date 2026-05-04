"""
Smoke test: run random actions for a few episodes to verify the env loads correctly.

Usage:
    uv run python scripts/test_env.py
    uv run python scripts/test_env.py --render --episodes 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.block_picking_env import BlockPickingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps",    type=int, default=200)
    args = parser.parse_args()

    env = BlockPickingEnv(
        render_mode="human" if args.render else None,
        max_episode_steps=args.steps,
        random_block_pos=True,
        reward_type="dense",
    )

    print(f"obs_space:    {env.observation_space}")
    print(f"action_space: {env.action_space}")
    print()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        t = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            t += 1

            if t % 50 == 0:
                print(
                    f"  ep={ep+1} t={t:3d}  "
                    f"r={reward:+5.2f}  "
                    f"ee_dist={info['ee_to_block']:.3f}  "
                    f"h={info['block_height']:.3f}  "
                    f"grasp={info['is_grasping']}"
                )

            if terminated or truncated:
                print(
                    f"  -> ep={ep+1} done  steps={t}  "
                    f"total_r={total_reward:.1f}  "
                    f"success={info['is_success']}\n"
                )
                break
    env.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
