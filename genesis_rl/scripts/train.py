"""
Training script for SO101 Pick and Place using RSL-RL
"""
import os
import sys
import yaml
import argparse
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import SO101PickPlaceEnv
from rsl_rl.runners import OnPolicyRunner


def main():
    parser = argparse.ArgumentParser(description="Train SO101 pick and place")
    parser.add_argument("--env-config", type=str, default="config/env_config.yaml")
    parser.add_argument("--ppo-config", type=str, default="config/ppo_config.yaml")
    parser.add_argument("--num-envs", type=int, default=None, help="Override num_envs")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override iterations")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    # Load configs
    with open(args.env_config, "r") as f:
        env_cfg = yaml.safe_load(f)
    with open(args.ppo_config, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    # Override config with command line args
    if args.num_envs is not None:
        env_cfg["env"]["num_envs"] = args.num_envs
    if args.num_iterations is not None:
        ppo_cfg["train"]["num_iterations"] = args.num_iterations

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    print("Creating environment...")
    env = SO101PickPlaceEnv(
        num_envs=env_cfg["env"]["num_envs"],
        env_spacing=env_cfg["env"]["env_spacing"],
        episode_length=env_cfg["env"]["episode_length"],
        device=device,
        headless=args.headless,
    )

    # Get observation and action dimensions
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Create runner
    experiment_name = args.experiment_name or f"so101_pick_place_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("logs", experiment_name)

    runner = OnPolicyRunner(
        env,
        train_cfg=ppo_cfg["train"],
        log_dir=log_dir,
        device=device,
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        runner.load(args.resume)

    # Train
    print(f"Starting training for {ppo_cfg['train']['num_iterations']} iterations...")
    runner.learn(
        num_learning_iterations=ppo_cfg["train"]["num_iterations"],
        init_at_random_ep_len=True,
    )

    # Save final model
    final_model_path = os.path.join("models", f"{experiment_name}_final.pt")
    runner.save(final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Export to ONNX
    if ppo_cfg["checkpoint"]["export_onnx"]:
        onnx_path = os.path.join("models", f"{experiment_name}_policy.onnx")
        runner.export_policy_onnx(onnx_path)
        print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
