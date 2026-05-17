"""
Asymmetric Actor-Critic SAC training with cameras.

Actor  : proprio (joint_pos + joint_vel, 12-dim) + overview image + hand_eye image
Critic : full ground-truth state (25-dim, privileged — training only)

Usage:
    uv run python train_sac_with_camera.py
    uv run python train_sac_with_camera.py --timesteps 1000000
    uv run python train_sac_with_camera.py --eval --checkpoint checkpoints/best_model
"""

from __future__ import annotations

import argparse
import time
import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.sac.policies import SACPolicy

from envs.block_picking_env import BlockPickingEnv, CAM_H, CAM_W

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

CONFIG: dict = {
    # Environment
    "n_envs":            8,      # fewer than train_sac.py — camera rendering is CPU-heavy
    "max_episode_steps": 500,
    "max_relative_action": 0.1,
    # SAC
    "learning_rate":     3e-4,
    "buffer_size":       1_000_000,
    "learning_starts":   10_000,
    "batch_size":        256,
    "tau":               0.005,
    "gamma":             0.99,
    "train_freq":        1,
    "gradient_steps":    1,
    "ent_coef":         "auto",
    "target_entropy":   "auto",
    # Network
    "net_arch":          [256, 256, 256],
    "cnn_features_dim":  256,    # output dim of NatureCNN per camera
    # Schedule
    "total_timesteps":   5_000_000,
    "eval_freq":         500_000,
    "n_eval_episodes":   10,
    "save_freq":         500_000,
    # Paths
    "log_dir":           "logs/",
    "checkpoint_dir":    "checkpoints/",
}

# ---------------------------------------------------------------------------
# Asymmetric feature extractors
# ---------------------------------------------------------------------------


class ActorExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the actor.
    Inputs: "proprio" (12-dim) + "overview" image + "hand_eye" image.
    Images are normalized to [0, 1] internally.
    """

    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 256):
        proprio_dim = observation_space["proprio"].shape[0]
        features_dim = proprio_dim + cnn_features_dim * 2
        super().__init__(observation_space, features_dim=features_dim)

        self._proprio_dim = proprio_dim

        # One NatureCNN per camera; input shape is (C, H, W) after channel-first conversion
        cam_space = spaces.Box(0, 255, shape=(3, CAM_H, CAM_W), dtype=np.uint8)
        self.cnn_overview = NatureCNN(cam_space, features_dim=cnn_features_dim)
        self.cnn_hand_eye = NatureCNN(cam_space, features_dim=cnn_features_dim)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        proprio = obs["proprio"]

        # obs images arrive as (B, H, W, C) uint8 from SB3 — convert to (B, C, H, W) float
        overview = obs["overview"].float().permute(0, 3, 1, 2) / 255.0
        hand_eye  = obs["hand_eye"].float().permute(0, 3, 1, 2) / 255.0

        feat_overview = self.cnn_overview(overview)
        feat_hand_eye = self.cnn_hand_eye(hand_eye)

        return torch.cat([proprio, feat_overview, feat_hand_eye], dim=1)


class CriticExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the critic.
    Input: "state" key only (full 25-dim ground-truth state — privileged).
    """

    def __init__(self, observation_space: spaces.Dict):
        state_dim = observation_space["state"].shape[0]
        super().__init__(observation_space, features_dim=state_dim)
        self._state_dim = state_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["state"]


# ---------------------------------------------------------------------------
# Asymmetric SAC policy
# ---------------------------------------------------------------------------


class AsymmetricSACPolicy(SACPolicy):
    """
    SAC policy where actor and critic use different observations:
      - actor  -> ActorExtractor  (proprio + cameras)
      - critic -> CriticExtractor (full ground-truth state)
    """

    def __init__(self, *args, cnn_features_dim: int = 256, **kwargs):
        self._cnn_features_dim = cnn_features_dim
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        if features_extractor is None:
            features_extractor = ActorExtractor(
                self.observation_space,
                cnn_features_dim=self._cnn_features_dim,
            ).to(self.device)
        return super().make_actor(features_extractor=features_extractor)

    def make_critic(self, features_extractor=None):
        if features_extractor is None:
            features_extractor = CriticExtractor(self.observation_space).to(self.device)
        return super().make_critic(features_extractor=features_extractor)


# ---------------------------------------------------------------------------
# Callbacks (identical to train_sac.py)
# ---------------------------------------------------------------------------


class LogCallback(BaseCallback):
    def __init__(self, log_interval: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self._interval = log_interval
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[float] = []
        self._successes:  list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])
            if "is_success" in info:
                self._successes.append(float(info["is_success"]))

        if self.n_calls % self._interval == 0 and self._ep_rewards:
            n = min(50, len(self._ep_rewards))
            print(
                f"[{self.num_timesteps:>8d}]  "
                f"reward={np.mean(self._ep_rewards[-n:]):+7.2f}  "
                f"len={np.mean(self._ep_lengths[-n:]):5.0f}  "
                f"success={np.mean(self._successes[-n:]) if self._successes else 0:.1%}"
            )
        return True


class TBCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._successes: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "is_success" in info:
                self._successes.append(float(info["is_success"]))
        for info in self.locals.get("infos", []):
            for key in info:
                if key.startswith("r_"):
                    self.logger.record(f"train/{key}", info[key])
        if len(self._successes) >= 100:
            self.logger.record("rollout/success_rate", np.mean(self._successes[-100:]))
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def _make_env(rank: int, seed: int, render_mode: str | None = None):
    def _init():
        env = BlockPickingEnv(
            render_mode=render_mode if rank == 0 else None,
            max_episode_steps=CONFIG["max_episode_steps"],
            random_block_pos=True,
            reward_type="dense",
            use_cameras=True,
            max_relative_action=CONFIG["max_relative_action"],
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(render: bool = False) -> SAC:
    taskid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_cam"
    CONFIG["checkpoint_dir"] += f"{taskid}/"
    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Envs: {CONFIG['n_envs']}  |  Steps: {CONFIG['total_timesteps']:,}")

    render_mode = "human" if render else None
    vec_env = SubprocVecEnv([
        _make_env(rank=i, seed=42, render_mode=render_mode)
        for i in range(CONFIG["n_envs"])
    ])
    # norm_obs=False: VecNormalize cannot handle mixed Dict obs (images + vectors).
    # Images are normalized to [0,1] inside ActorExtractor; state is used raw by critic.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: Monitor(BlockPickingEnv(
        max_episode_steps=CONFIG["max_episode_steps"],
        random_block_pos=True,
        reward_type="dense",
        use_cameras=True,
        max_relative_action=CONFIG["max_relative_action"],
    ))])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    model = SAC(
        policy=AsymmetricSACPolicy,
        env=vec_env,
        learning_rate=CONFIG["learning_rate"],
        buffer_size=CONFIG["buffer_size"],
        learning_starts=CONFIG["learning_starts"],
        batch_size=CONFIG["batch_size"],
        tau=CONFIG["tau"],
        gamma=CONFIG["gamma"],
        train_freq=CONFIG["train_freq"],
        gradient_steps=CONFIG["gradient_steps"],
        ent_coef=CONFIG["ent_coef"],
        target_entropy=CONFIG["target_entropy"],
        policy_kwargs=dict(
            net_arch=CONFIG["net_arch"],
            activation_fn=torch.nn.ReLU,
            cnn_features_dim=CONFIG["cnn_features_dim"],
        ),
        tensorboard_log=CONFIG["log_dir"],
        verbose=0,
        device=device,
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy parameters: {param_count:,}")

    callbacks = [
        LogCallback(log_interval=5_000),
        TBCallback(),
        CheckpointCallback(
            save_freq=CONFIG["save_freq"] // CONFIG["n_envs"],
            save_path=CONFIG["checkpoint_dir"],
            name_prefix="sac_cam",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=CONFIG["checkpoint_dir"],
            log_path=CONFIG["log_dir"],
            eval_freq=CONFIG["eval_freq"] // CONFIG["n_envs"],
            n_eval_episodes=CONFIG["n_eval_episodes"],
            deterministic=True,
            render=False,
        ),
    ]

    t0 = time.time()
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
        tb_log_name=taskid,
    )
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed / 3600:.2f}h")

    model.save(f"{CONFIG['checkpoint_dir']}/sac_cam_final")
    vec_env.save(f"{CONFIG['checkpoint_dir']}/sac_cam_vecnormalize_final.pkl")

    vec_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render",     action="store_true")
    parser.add_argument("--timesteps",  type=int, default=None)
    args = parser.parse_args()

    if args.timesteps:
        CONFIG["total_timesteps"] = args.timesteps

    train(render=args.render)


if __name__ == "__main__":
    main()
