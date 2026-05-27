"""Evaluate a saved policy checkpoint in the env (success rate + video).

Example:
    uv run python -m scripts.eval_policy --ckpt checkpoints/bc.pt --episodes 20
    uv run python -m scripts.eval_policy --ckpt checkpoints/student.pt --episodes 10 --video out.mp4
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from mjlab_rl.envs import BlockPickingEnv
from mjlab_rl.policies import MLPPolicy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-random-block", action="store_true")
    p.add_argument("--no-random-container", action="store_true")
    args = p.parse_args()

    policy = MLPPolicy.load(args.ckpt)
    policy.eval()

    render_mode = "rgb_array" if args.video else ("human" if args.render else None)
    env = BlockPickingEnv(
        render_mode=render_mode,
        max_episode_steps=args.max_steps,
        random_block_pos=not args.no_random_block,
        random_container_pos=not args.no_random_container,
    )

    succ = 0
    rets: list[float] = []
    frames: list[np.ndarray] = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_r = 0.0
        for _ in range(args.max_steps):
            a = policy.act_numpy(obs)
            obs, r, term, trunc, info = env.step(a)
            ep_r += float(r)
            if args.video and ep == 0:
                frames.append(env.render())
            if term or trunc:
                break
        ok = bool(info.get("is_success", False))
        succ += int(ok)
        rets.append(ep_r)
        print(
            f"[ep {ep:3d}] success={ok} return={ep_r:8.2f} "
            f"d_blk_tgt_xy={info['d_blk_tgt_xy']:.3f} block_h={info['block_h']:.3f}"
        )

    print("=" * 60)
    print(f"{args.ckpt}: success {succ}/{args.episodes} = {succ/max(1,args.episodes):.1%}")
    if rets:
        print(f"mean return : {statistics.mean(rets):.2f}")

    if args.video and frames:
        try:
            import imageio.v2 as imageio
            out = Path(args.video)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(out, frames, fps=int(env.metadata["render_fps"]))
            print(f"Video -> {out}")
        except Exception as e:  # pragma: no cover
            print(f"Video write failed: {e}")


if __name__ == "__main__":
    main()
