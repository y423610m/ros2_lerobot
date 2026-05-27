"""Roll out the scripted oracle in the env to sanity-check it.

Example:
    uv run python -m scripts.test_solver --episodes 5 --render --video out.mp4
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from mjlab_rl.envs import BlockPickingEnv
from mjlab_rl.solvers import OracleSolver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=str, default=None,
                   help="If set, write an .mp4 of the first episode.")
    p.add_argument("--render", action="store_true",
                   help="Open a live mujoco viewer window.")
    p.add_argument("--no-random-block", action="store_true")
    p.add_argument("--no-random-container", action="store_true")
    args = p.parse_args()

    render_mode = "rgb_array" if args.video else ("human" if args.render else None)

    env = BlockPickingEnv(
        render_mode=render_mode,
        max_episode_steps=args.max_steps,
        random_block_pos=not args.no_random_block,
        random_container_pos=not args.no_random_container,
    )
    solver = OracleSolver(env)

    successes = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    frames: list[np.ndarray] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        solver.reset(obs)
        ep_ret = 0.0
        steps = 0
        for t in range(args.max_steps):
            a = solver.act(obs)
            obs, r, term, trunc, info = env.step(a)
            ep_ret += float(r)
            steps += 1
            if args.video and ep == 0:
                frames.append(env.render())
            if term or trunc:
                break
        ok = bool(info.get("is_success", False))
        successes += int(ok)
        ep_returns.append(ep_ret)
        ep_lengths.append(steps)
        print(
            f"[ep {ep:3d}] steps={steps:4d} return={ep_ret:8.2f} "
            f"phase={solver.phase.name:14s} success={ok} "
            f"block->tgt_xy={info['d_blk_tgt_xy']:.3f} block_h={info['block_h']:.3f}"
        )

    sr = successes / max(1, args.episodes)
    print("=" * 64)
    print(f"Oracle success rate: {sr:.1%} ({successes}/{args.episodes})")
    if ep_returns:
        print(f"Mean return : {statistics.mean(ep_returns):.2f}")
        print(f"Mean length : {statistics.mean(ep_lengths):.1f}")

    if args.video and frames:
        try:
            import imageio.v2 as imageio
            out = Path(args.video)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(out, frames, fps=int(env.metadata["render_fps"]))
            print(f"Saved video -> {out}")
        except Exception as e:  # pragma: no cover
            print(f"Video write failed: {e}")


if __name__ == "__main__":
    main()
