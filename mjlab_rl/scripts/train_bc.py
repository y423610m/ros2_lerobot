"""Train an MLP policy via Behavior Cloning on oracle-generated trajectories.

Pipeline:
    1. Roll out the OracleSolver in BlockPickingEnv to collect (obs, act).
    2. Keep only *successful* episodes.
    3. MSE-regress the policy to oracle actions.
    4. Periodically evaluate the policy in the env and report success rate.

Example:
    uv run python -m scripts.train_bc --num-demos 200 --epochs 30 \
        --hidden 256 256 --eval-every 5 --eval-episodes 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mjlab_rl.envs import BlockPickingEnv
from mjlab_rl.policies import MLPPolicy
from mjlab_rl.solvers import OracleSolver


# ---------------------------------------------------------------------------
def collect_demos(
    num_demos: int,
    max_steps: int,
    seed: int,
    keep_only_success: bool = True,
):
    env = BlockPickingEnv(max_episode_steps=max_steps)
    solver = OracleSolver(env)

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    successes = 0
    attempts = 0
    pbar = tqdm(total=num_demos, desc="oracle demos", unit="ep")
    while successes < num_demos and attempts < num_demos * 5:
        obs, _ = env.reset(seed=seed + attempts)
        solver.reset(obs)
        ep_obs, ep_act = [], []
        ok = False
        for _ in range(max_steps):
            a = solver.act(obs)
            ep_obs.append(obs.copy())
            ep_act.append(a.copy())
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc:
                ok = bool(info.get("is_success", False))
                break
        attempts += 1
        if (not keep_only_success) or ok:
            obs_buf.extend(ep_obs)
            act_buf.extend(ep_act)
            successes += 1
            pbar.update(1)
    pbar.close()
    if successes < num_demos:
        print(
            f"[warn] only collected {successes}/{num_demos} successful demos "
            f"after {attempts} attempts."
        )
    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.float32),
        successes,
        attempts,
    )



def dagger_rollout(policy: MLPPolicy, num_episodes: int, max_steps: int, seed: int):
    """Roll out the student policy and relabel each visited state with the oracle action."""
    env = BlockPickingEnv(max_episode_steps=max_steps)
    solver = OracleSolver(env)
    obs_buf, act_buf = [], []
    succ = 0
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        solver.reset(obs)
        for _ in range(max_steps):
            # Oracle action FROM the current env state (label).
            oracle_a = solver.act(obs)
            obs_buf.append(obs.copy())
            act_buf.append(oracle_a.copy())
            # Student action drives the env (rolls out on student's distribution).
            step_a = policy.act_numpy(obs)
            obs, _r, term, trunc, info = env.step(step_a)
            if term or trunc:
                break
        if bool(info.get("is_success", False)):
            succ += 1
    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.float32),
        succ / max(1, num_episodes),
    )

def evaluate(policy: MLPPolicy, episodes: int, max_steps: int, seed: int) -> dict:
    env = BlockPickingEnv(max_episode_steps=max_steps)
    succ = 0
    rets: list[float] = []
    policy.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        ep_r = 0.0
        for _ in range(max_steps):
            a = policy.act_numpy(obs)
            obs, r, term, trunc, info = env.step(a)
            ep_r += float(r)
            if term or trunc:
                break
        if bool(info.get("is_success", False)):
            succ += 1
        rets.append(ep_r)
    return dict(
        success_rate=succ / max(1, episodes),
        mean_return=float(np.mean(rets)) if rets else 0.0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-demos", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--ckpt", type=str, default="checkpoints/bc.pt")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--keep-failures",
        action="store_true",
        help="Also keep unsuccessful oracle rollouts in the dataset.",
    )
    p.add_argument("--dagger-rounds", type=int, default=0,
                   help="After offline BC, run N DAgger rounds (oracle relabels student-visited states).")
    p.add_argument("--dagger-episodes", type=int, default=20)
    p.add_argument("--dagger-epochs", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"== Collecting {args.num_demos} oracle demos ==")
    t0 = time.time()
    obs, act, n_ok, n_try = collect_demos(
        args.num_demos,
        args.max_steps,
        args.seed,
        keep_only_success=not args.keep_failures,
    )
    dt = time.time() - t0
    print(
        f"Collected {len(obs)} transitions from {n_ok} demos "
        f"(oracle success rate {n_ok/max(1,n_try):.1%}) in {dt:.1f}s."
    )
    if len(obs) == 0:
        raise SystemExit("No demonstrations collected -- aborting.")

    policy = MLPPolicy(
        obs_dim=obs.shape[1], act_dim=act.shape[1], hidden_sizes=args.hidden
    ).to(args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.as_tensor(obs), torch.as_tensor(act))
    n_train = int(0.9 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    best_sr = -1.0
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_loss = 0.0
        for ob, ac in train_loader:
            ob = ob.to(args.device, non_blocking=True)
            ac = ac.to(args.device, non_blocking=True)
            pred = policy(ob)
            loss = loss_fn(pred, ac)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * ob.size(0)
        train_loss /= max(1, n_train)

        policy.eval()
        with torch.no_grad():
            val_loss = 0.0
            for ob, ac in val_loader:
                ob = ob.to(args.device); ac = ac.to(args.device)
                val_loss += loss_fn(policy(ob), ac).item() * ob.size(0)
            val_loss /= max(1, n_val)

        msg = f"[epoch {epoch:3d}] train_mse={train_loss:.5f}  val_mse={val_loss:.5f}"
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate(
                policy.cpu(), args.eval_episodes, args.max_steps, args.seed
            )
            policy.to(args.device)
            msg += f"  eval_success={metrics['success_rate']:.1%}  eval_ret={metrics['mean_return']:.2f}"
            if metrics["success_rate"] >= best_sr:
                best_sr = metrics["success_rate"]
                policy.cpu().save(args.ckpt)
                policy.to(args.device)
                msg += "  [saved]"
        print(msg)

    # Final save (and final eval if not just done).
    policy.cpu().save(args.ckpt)
    policy.to(args.device)
    if args.eval_episodes > 0:
        final = evaluate(
            policy.cpu(), args.eval_episodes, args.max_steps, args.seed
        )
        print(
            f"== Final BC policy success rate: {final['success_rate']:.1%}, "
            f"mean return: {final['mean_return']:.2f} =="
        )
    # ---------------- Optional DAgger fine-tuning ----------------
    if args.dagger_rounds > 0:
        print(f"== DAgger fine-tuning: {args.dagger_rounds} rounds ==")
        obs_pool = obs
        act_pool = act
        for r in range(1, args.dagger_rounds + 1):
            policy.cpu().eval()
            obs_new, act_new, s_sr = dagger_rollout(
                policy, args.dagger_episodes, args.max_steps,
                args.seed + 10000 + r * 1000,
            )
            policy.to(args.device).train()
            print(
                f"  [round {r}] student rollout success={s_sr:.1%}, "
                f"new transitions={len(obs_new)}"
            )
            obs_pool = np.concatenate([obs_pool, obs_new], axis=0)
            act_pool = np.concatenate([act_pool, act_new], axis=0)

            ds = TensorDataset(torch.as_tensor(obs_pool), torch.as_tensor(act_pool))
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
            for ep in range(args.dagger_epochs):
                policy.train()
                for ob, ac in loader:
                    ob = ob.to(args.device); ac = ac.to(args.device)
                    loss = loss_fn(policy(ob), ac)
                    opt.zero_grad(); loss.backward(); opt.step()
            metrics = evaluate(
                policy.cpu(), args.eval_episodes, args.max_steps, args.seed
            )
            policy.to(args.device)
            print(                f"  [round {r}] post-fit eval success={metrics['success_rate']:.1%} "
                f"ret={metrics['mean_return']:.2f}  dataset={len(obs_pool)}"
            )
            if metrics["success_rate"] >= best_sr:
                best_sr = metrics["success_rate"]
                policy.cpu().save(args.ckpt)
                policy.to(args.device)
                print(f"  [saved best]")

    print(f"Saved {args.ckpt}")


if __name__ == "__main__":
    main()
