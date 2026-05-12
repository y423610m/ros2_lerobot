"""
SAC training script for block picking.

Usage:
    uv run python train_sac.py                          # train with defaults
    uv run python train_sac.py --render                 # visualize during training
    uv run python train_sac.py --timesteps 10000000       # quick test run
    uv run python train_sac.py --eval --render --checkpoint checkpoints/best_model
    
    uv run python -m mujoco.viewer --mjcf block_picking.xml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import datetime

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.block_picking_env import BlockPickingEnv

import IPython

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

CONFIG: dict = {
    # Environment
    "n_envs":            32,
    "max_episode_steps": 200,
    # SAC
    "learning_rate":     3e-4,
    "buffer_size":       1_000_000,
    "learning_starts":   10_000,
    "batch_size":        2048,
    "tau":               0.005,
    "gamma":             0.99,
    "train_freq":        1,
    "gradient_steps":    1,
    "ent_coef":         "auto",
    "target_entropy":   "auto",
    # Network
    "net_arch":          [256, 256, 256],
    # Schedule
    "total_timesteps":   2_000_000,
    "eval_freq":         2_000,
    "n_eval_episodes":   10,
    "save_freq":         200_000,
    # Paths
    "log_dir":           "logs/",
    "checkpoint_dir":    "checkpoints/",
}

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class LogCallback(BaseCallback):
    """Print episode stats to stdout every ``log_interval`` steps."""

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
    """Log extra metrics to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._successes: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "is_success" in info:
                self._successes.append(float(info["is_success"]))
        for info in self.locals.get("infos", []):
            for key in info:
                if key.startswith('r_'):
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
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(render: bool = False) -> SAC:
    taskid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    CONFIG["checkpoint_dir"] += f"{taskid}/"
    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    device = "cuda"
    print(f"Device: {device}  |  Envs: {CONFIG['n_envs']}  |  Steps: {CONFIG['total_timesteps']:,}")

    render_mode = "human" if render else None
    vec_env = SubprocVecEnv([
        _make_env(rank=i, seed=42, render_mode=render_mode)
        for i in range(CONFIG["n_envs"])
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: Monitor(BlockPickingEnv(
        max_episode_steps=CONFIG["max_episode_steps"],
        random_block_pos=True,
        reward_type="dense",
    ))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    model = SAC(
        policy="MlpPolicy",
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
            name_prefix="sac",
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

    model.save(f"{CONFIG['checkpoint_dir']}/sac_final")
    vec_env.save(f"{CONFIG['checkpoint_dir']}/vec_normalize_final.pkl")

    vec_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(checkpoint: str, n_episodes: int = 20, render: bool = True) -> None:
    vec_normalize_path = Path(checkpoint).parent / "vec_normalize_final.pkl"

    CONFIG["max_episode_steps"] = 100000

    env = DummyVecEnv([lambda: Monitor(BlockPickingEnv(
        render_mode="human" if render else None,
        max_episode_steps=CONFIG["max_episode_steps"],
        random_block_pos=True,
    ))])

    if vec_normalize_path.exists():
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False    # freeze stats, do not update during eval
        env.norm_reward = False # evaluate on raw rewards
        print(f"Loaded VecNormalize stats from {vec_normalize_path}")
    else:
        print(f"Warning: {vec_normalize_path} not found, running without obs normalization")

    model = SAC.load(checkpoint, env=env)

    rewards, lengths, successes = [], [], []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        IPython.embed()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if render and ep_len%10==0:
                env.render()
            ep_reward += float(reward[0])
            ep_len += 1
            print(f"{action[0]=}")
            # print(f"{obs[0][19:19+6]=}")
        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(float(info[0].get("is_success", False)))
        print(f"  ep {ep+1:2d}: reward={ep_reward:+7.1f}  len={ep_len:3d}  success={bool(successes[-1])}")

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  mean reward:   {np.mean(rewards):+.2f} +/- {np.std(rewards):.2f}")
    print(f"  mean length:   {np.mean(lengths):.0f}")
    print(f"  success rate:  {np.mean(successes):.1%}")

    env.close()


def check(checkpoint: str, n_episodes: int = 20, render: bool = True) -> None:
    vec_normalize_path = Path(checkpoint).parent / "vec_normalize_final.pkl"

    env = DummyVecEnv([lambda: Monitor(BlockPickingEnv(
        render_mode="human" if render else None,
        max_episode_steps=CONFIG["max_episode_steps"],
        random_block_pos=True,
    ))])

    if vec_normalize_path.exists():
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False    # freeze stats, do not update during eval
        env.norm_reward = False # evaluate on raw rewards
        print(f"Loaded VecNormalize stats from {vec_normalize_path}")
    else:
        print(f"Warning: {vec_normalize_path} not found, running without obs normalization")

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        # IPython.embed()
        action = np.array([[0, 0, 0, 0, 0, 0]], np.float32)
        cnt = 0
        e = env.envs[0].env
        while 1:
            action[0][5] = np.sin(cnt * np.pi / 180)
            obs, reward, done, info = env.step(action)
            if render:
                env.render()
            ep_reward += float(reward[0])
            ep_len += 1
            # print(f"{action[0]=}")
            d = e.data
            ee_gripper = d.site_xpos[e._ee_gripper_sid].copy()
            ee_wrist = d.site_xpos[e._ee_wrist_sid].copy()
            print(f"{obs[0][12:12+3]=}")
            print(f"{ee_gripper=}")
            print(f"{ee_wrist=}")
            cnt += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--check",        action="store_true")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model")
    parser.add_argument("--n-eval-eps",  type=int, default=3)
    parser.add_argument("--timesteps",   type=int, default=None)
    args = parser.parse_args()

    if args.timesteps:
        CONFIG["total_timesteps"] = args.timesteps

    if args.eval:
        evaluate(args.checkpoint, n_episodes=args.n_eval_eps, render=args.render)
    if args.check:
        check(args.checkpoint, n_episodes=args.n_eval_eps, render=args.render)
    else:
        train(render=args.render)


if __name__ == "__main__":
    main()
