"""Distill a trained BC policy into a new (typically smaller) student policy.

Two data sources are supported and combined:

  * Offline dataset: roll out the *teacher* in the env, store (obs, teacher_act).
  * Online dataset:  during training, periodically have the *student* roll out,
    relabel its visited obs with the teacher (DAgger-style aggregation).

This way the student matches the teacher both on the teacher's
state-distribution and on its own.

Example:
    uv run python -m scripts.distill --teacher-ckpt checkpoints/bc.pt \
        --student-hidden 128 128 --offline-demos 50 --dagger-rounds 5 \
        --epochs-per-round 10 --student-ckpt checkpoints/student.pt
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


# ---------------------------------------------------------------------------
def rollout_with_teacher(
    teacher: MLPPolicy,
    episodes: int,
    max_steps: int,
    seed: int,
    actor: MLPPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Roll out `actor` (or `teacher` if None) and relabel each obs with teacher action."""
    env = BlockPickingEnv(max_episode_steps=max_steps)
    obs_list, act_list = [], []
    succ = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            obs_list.append(obs.copy())
            with torch.no_grad():
                t_act = teacher.act_numpy(obs)
            act_list.append(np.asarray(t_act, dtype=np.float32))
            step_act = t_act if actor is None else actor.act_numpy(obs)
            obs, _r, term, trunc, info = env.step(step_act)
            if term or trunc:
                break
        if bool(info.get("is_success", False)):
            succ += 1
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(act_list, dtype=np.float32),
        succ / max(1, episodes),
    )


def evaluate(policy: MLPPolicy, episodes: int, max_steps: int, seed: int) -> dict:
    env = BlockPickingEnv(max_episode_steps=max_steps)
    succ = 0
    rets: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 2000 + ep)
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
    return dict(success_rate=succ / max(1, episodes), mean_return=float(np.mean(rets)))


def fit_student(
    student: MLPPolicy,
    obs: np.ndarray,
    act: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> float:
    ds = TensorDataset(torch.as_tensor(obs), torch.as_tensor(act))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    student.train()
    last_loss = float("inf")
    for ep in range(epochs):
        run_loss = 0.0
        n = 0
        for ob, ac in loader:
            ob = ob.to(device); ac = ac.to(device)
            pred = student(ob)
            loss = loss_fn(pred, ac)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * ob.size(0); n += ob.size(0)
        last_loss = run_loss / max(1, n)
    return last_loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt", type=str, required=True)
    p.add_argument(
        "--student-hidden", type=int, nargs="+", default=[128, 128]
    )
    p.add_argument("--offline-demos", type=int, default=50)
    p.add_argument("--dagger-rounds", type=int, default=3)
    p.add_argument("--rollouts-per-round", type=int, default=20)
    p.add_argument("--epochs-per-round", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--student-ckpt", type=str, default="checkpoints/student.pt")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"== Loading teacher from {args.teacher_ckpt} ==")
    teacher = MLPPolicy.load(args.teacher_ckpt)
    teacher.eval()
    print(
        f"   teacher: obs_dim={teacher.obs_dim} act_dim={teacher.act_dim} "
        f"hidden={teacher.hidden_sizes}"
    )

    student = MLPPolicy(
        obs_dim=teacher.obs_dim,
        act_dim=teacher.act_dim,
        hidden_sizes=args.student_hidden,
    ).to(args.device)
    print(f"   student: hidden={tuple(args.student_hidden)}")

    # ---- 1) Offline phase: imitate teacher on teacher rollouts ------------
    print(f"== Offline distillation: {args.offline_demos} teacher rollouts ==")
    t0 = time.time()
    obs_buf, act_buf, t_sr = rollout_with_teacher(
        teacher, args.offline_demos, args.max_steps, args.seed, actor=None
    )
    print(
        f"   teacher success rate during collection: {t_sr:.1%} | "
        f"transitions: {len(obs_buf)} | {time.time()-t0:.1f}s"
    )

    if len(obs_buf) > 0:
        loss = fit_student(
            student, obs_buf, act_buf,
            args.epochs_per_round, args.batch_size, args.lr, args.device,
        )
        print(f"   offline fit loss: {loss:.5f}")

    # ---- 2) DAgger rounds: relabel student rollouts with teacher ---------
    for r in range(1, args.dagger_rounds + 1):
        print(f"== DAgger round {r}/{args.dagger_rounds} ==")
        student.eval().cpu()
        obs_new, act_new, s_sr = rollout_with_teacher(
            teacher, args.rollouts_per_round, args.max_steps,
            args.seed + 1000 + r, actor=student,
        )
        student.to(args.device)
        print(
            f"   student-driven rollouts: success={s_sr:.1%}, "
            f"transitions added: {len(obs_new)}"
        )
        if len(obs_new) > 0:
            obs_buf = np.concatenate([obs_buf, obs_new], axis=0)
            act_buf = np.concatenate([act_buf, act_new], axis=0)
        loss = fit_student(
            student, obs_buf, act_buf,
            args.epochs_per_round, args.batch_size, args.lr, args.device,
        )
        print(f"   refit loss: {loss:.5f}  (dataset size: {len(obs_buf)})")

    # ---- 3) Save + evaluate ----------------------------------------------
    Path(args.student_ckpt).parent.mkdir(parents=True, exist_ok=True)
    student.cpu().save(args.student_ckpt)
    student.to(args.device)
    print(f"== Saved student -> {args.student_ckpt} ==")
    if args.eval_episodes > 0:
        m_t = evaluate(teacher, args.eval_episodes, args.max_steps, args.seed)
        m_s = evaluate(student.cpu(), args.eval_episodes, args.max_steps, args.seed)
        print(
            f"Teacher  success={m_t['success_rate']:.1%}  return={m_t['mean_return']:.2f}"
        )
        print(
            f"Student  success={m_s['success_rate']:.1%}  return={m_s['mean_return']:.2f}"
        )


if __name__ == "__main__":
    main()
