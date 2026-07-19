"""Distill a closed-loop (chunk=1) vision policy into an action-chunking student.

From-scratch chunk RL couldn't even reach the object (blind 1 s open-loop is an
exploration nightmare). Instead we use the good closed-loop policy as a TEACHER and
train a chunk=N STUDENT (actor head = N*6) to *copy* it — the ACT recipe (action
chunking + behavior cloning). The teacher is a perfectly queryable expert in sim.

Pipeline (one process, one env):
  * env      : ChunkedVecEnvWrapper at CHUNK_SIZE=N (so the student actor head is N*6).
  * student  : built by the runner; warm-started from the teacher ckpt via
               WarmStartOnPolicyRunner.load (copies CNN backbone + trunk + obs-normalizer
               verbatim and prefix-seeds the head with the teacher's action-0).
  * teacher  : a bare 6-dim SpatialSoftmaxCNNModel loaded from the same ckpt, frozen.
  * collect  : drive the env ONE step at a time with the teacher (renders every step,
               unlike the chunk wrapper which skips interior frames). Non-overlapping
               windows aligned to episode reset: boundary obs at t=0,N,2N,... (exactly the
               student's deploy inference points) -> target = the next N teacher raw actions.
  * train    : regress student(boundary_obs) (N*6) to the N-action target with smooth-L1.
  * save     : runner.save() -> standard ckpt that export_to_jit.py consumes unchanged
               (it stamps chunk_size = num_actions // 6).

Optional DAgger (--dagger): drive the env with the STUDENT's chunk (open-loop) but label
each visited state with the teacher -> corrects open-loop covariate shift.

Progress is logged to TensorBoard (event files in the out-dir, same tree as the DR runs):
  tensorboard --logdir logs/rsl_rl/so101_block_picking_vision
Tags: train/loss, train/loss_step0 vs train/loss_steplast (first- vs last-sub-action fit —
shows late-chunk divergence), train/buffer, eval/reward_per_chunk, eval/success, eval/lift, ...

Example:
  uv run python scripts/distill_chunk.py \\
    --teacher-checkpoint logs/rsl_rl/so101_block_picking_vision/2026-06-29_22-45-05_dr3/model_81999.pt \\
    --task-id Mjlab-SO101-Block-Picking-Rgb-DR0
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class DistillConfig:
  teacher_checkpoint: str
  task_id: str = "Mjlab-SO101-Block-Picking-Rgb-DR0"
  chunk_size: int = 50
  num_envs: int = 256
  # Continue a previous distill run: a model_*.pt saved by THIS script (same chunk
  # size). Restores student weights + optimizer state and continues the iteration
  # numbering, so model files / TB steps carry on from where the run stopped.
  # `iters` then means ADDITIONAL updates to run.
  resume_checkpoint: str | None = None
  iters: int = 20000
  batch_size: int = 256
  lr: float = 3e-4
  loss: str = "smooth_l1"  # {smooth_l1, l1, mse}
  buffer_size: int = 20000
  chunk_weight_decay: float = 1.0  # <1 downweights late-in-chunk sub-steps
  collect_every: int = 1  # env collection steps per gradient update
  dagger: bool = False  # student-driven rollout + teacher relabel
  eval_every: int = 1000
  # Eval horizon in CONTROL steps (not chunks), so it covers the same sim time for
  # any chunk size. Must span >= a full episode (400 steps) or timeouts never occur
  # during eval and Episode_Reward/* is computed only on early-failure episodes
  # (success reads ~0 no matter how good the policy is). 1200 = 3 episode lengths.
  eval_steps: int = 1200
  save_every: int = 20000
  log_every: int = 50
  out_dir: str | None = None
  device: str = "cuda"
  seed: int = 0


def _build_teacher(model_cls, obs, obs_groups, actor_cfg, ckpt, device):
  """Bare 6-dim actor loaded from the closed-loop teacher ckpt (frozen)."""
  sig = set(inspect.signature(model_cls.__init__).parameters)
  kw = {k: v for k, v in dict(actor_cfg).items() if k in sig}
  teacher = model_cls(obs, obs_groups, "actor", 6, **kw).to(device)
  missing, unexpected = teacher.load_state_dict(ckpt["actor_state_dict"], strict=False)
  if missing or unexpected:
    print(f"[distill] teacher load: missing={list(missing)} unexpected={list(unexpected)}")
  teacher.eval()
  for p in teacher.parameters():
    p.requires_grad_(False)
  return teacher


def run(cfg: DistillConfig) -> None:
  # CHUNK_SIZE must be set before the wrapper reads it at construction.
  os.environ["CHUNK_SIZE"] = str(cfg.chunk_size)

  import mjlab_rl  # noqa: F401  (registers tasks)
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from tensordict import TensorDict

  from mjlab_rl.envs.chunked_env import ChunkedVecEnvWrapper

  torch.manual_seed(cfg.seed)
  device = cfg.device
  W, base = cfg.chunk_size, 6

  ckpt_path = Path(cfg.teacher_checkpoint).expanduser().resolve()
  if not ckpt_path.exists():
    raise FileNotFoundError(ckpt_path)
  # Default out-dir starts with the timestamp (like the DR run dirs) so runs sort
  # chronologically in ls/TensorBoard; the chunk size is the suffix.
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out_dir = (
    Path(cfg.out_dir).expanduser().resolve()
    if cfg.out_dir
    else ckpt_path.parent / f"{stamp}_distill_chunk{W}"
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  # TensorBoard: event files land in out_dir (logs/rsl_rl/.../<stamp>_distill_chunkN/),
  # so `tensorboard --logdir logs/rsl_rl/so101_block_picking_vision` shows this run
  # alongside the DR runs.
  writer = SummaryWriter(log_dir=str(out_dir))

  # --- env (training cfg: DR + resets active) ---
  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  agent_cfg = load_rl_cfg(cfg.task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = ChunkedVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  assert env.num_actions == W * base, f"expected head {W * base}, got {env.num_actions}"
  N = cfg.num_envs
  obs = env.get_observations()

  # --- student (warm-started from the teacher: backbone+trunk+norm + head prefix) ---
  runner = load_runner_cls(cfg.task_id)(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=False, map_location=device)
  student = runner.alg._raw_actor
  student.eval()  # no dropout/bn here; also freezes the copied obs_normalizer stats

  # --- teacher (bare 6-dim actor, frozen) ---
  ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
  obs_groups = asdict(agent_cfg)["obs_groups"]
  teacher = _build_teacher(
    SpatialSoftmaxCNNModel, obs, obs_groups, asdict(agent_cfg)["actor"], ckpt, device
  )
  print(f"[distill] student head={tuple(student.state_dict()['mlp.6.weight'].shape)} "
        f"teacher head=(6,128) | out_dir={out_dir}")

  optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)

  # --- resume: overwrite the teacher warm-start with a previous distill state ---
  start_it = 0
  if cfg.resume_checkpoint:
    rpath = Path(cfg.resume_checkpoint).expanduser().resolve()
    rck = torch.load(str(rpath), map_location=device, weights_only=False)
    student.load_state_dict(rck["actor_state_dict"], strict=True)  # same chunk -> exact
    if "optimizer_state_dict" in rck:
      optimizer.load_state_dict(rck["optimizer_state_dict"])  # keep Adam moments
    start_it = int(rck.get("iter", 0))
    print(f"[distill] resumed student+optimizer from {rpath} (iter {start_it}); "
          f"running {cfg.iters} more updates")

  def save_ckpt(path: str, it: int) -> None:
    # Standard rsl-rl checkpoint layout so export_to_jit.py (load_cfg={"actor":True})
    # and runner.load consume it unchanged. Avoids runner.save's logger/writer
    # dependency (only set up inside learn()).
    torch.save(
      {
        "actor_state_dict": student.state_dict(),
        "critic_state_dict": runner.alg._raw_critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": it,
        "infos": None,
      },
      path,
    )

  # --- replay buffer (CPU; camera stored uint8) ---
  cap = cfg.buffer_size
  rb_actor = torch.zeros(cap, obs["actor"].shape[1])
  rb_cam = torch.zeros(cap, *obs["camera"].shape[1:], dtype=torch.uint8)
  rb_tgt = torch.zeros(cap, W * base)
  rb_ptr, rb_n = 0, 0

  def push(a, c, t):
    nonlocal rb_ptr, rb_n
    n = a.shape[0]
    idx = (torch.arange(n) + rb_ptr) % cap
    rb_actor[idx] = a.cpu()
    rb_cam[idx] = c.cpu()
    rb_tgt[idx] = t.cpu()
    rb_ptr = int((rb_ptr + n) % cap)
    rb_n = min(rb_n + n, cap)

  # --- non-overlapping window state (device) ---
  win_actor0 = torch.zeros(N, obs["actor"].shape[1], device=device)
  win_cam0 = torch.zeros(N, *obs["camera"].shape[1:], dtype=torch.uint8, device=device)
  win_raw = torch.zeros(N, W, base, device=device)
  win_len = torch.zeros(N, dtype=torch.long, device=device)
  stu_chunk = torch.zeros(N, W, base, device=device)  # dagger only
  ar = torch.arange(N, device=device)

  def to_u8(cam):
    return (cam.clamp(0.0, 1.0) * 255).to(torch.uint8)

  def collect_step():
    nonlocal obs
    new_win = win_len == 0
    with torch.no_grad():
      teacher_raw = teacher(obs)  # (N,6) label
      if cfg.dagger and bool(new_win.any()):
        stu_chunk[new_win] = student(obs).view(N, W, base)[new_win]
    # capture boundary obs at window start
    if bool(new_win.any()):
      win_actor0[new_win] = obs["actor"][new_win]
      win_cam0[new_win] = to_u8(obs["camera"])[new_win]
    win_raw[ar, win_len.clamp(max=W - 1)] = teacher_raw
    win_len.add_(1)
    exec_act = stu_chunk[ar, (win_len - 1).clamp(max=W - 1)] if cfg.dagger else teacher_raw
    obs, _rew, dones, _extras = RslRlVecEnvWrapper.step(env, exec_act)
    # emit completed windows
    full = win_len == W
    if bool(full.any()):
      push(win_actor0[full], win_cam0[full], win_raw[full].reshape(-1, W * base))
      win_len[full] = 0
    # discard windows crossing an episode reset
    win_len[dones.bool()] = 0

  def sample(b):
    idx = torch.randint(0, rb_n, (b,))
    a = rb_actor[idx].to(device)
    c = (rb_cam[idx].float() / 255.0).to(device)
    t = rb_tgt[idx].to(device)
    return TensorDict({"actor": a, "camera": c}, batch_size=[b]), t

  loss_fn = {"smooth_l1": F.smooth_l1_loss, "l1": F.l1_loss, "mse": F.mse_loss}[cfg.loss]
  sub_w = None
  if cfg.chunk_weight_decay != 1.0:
    w = cfg.chunk_weight_decay ** torch.arange(W, device=device)
    sub_w = (w / w.mean()).repeat_interleave(base)  # (W*6,), mean-1 normalized

  def update():
    obs_b, tgt = sample(cfg.batch_size)
    pred = student(obs_b)  # (B, W*6) deterministic mean
    if sub_w is None:
      loss = loss_fn(pred, tgt)
    else:
      loss = (loss_fn(pred, tgt, reduction="none") * sub_w).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    # per-horizon fit: first vs last sub-action (shows late-chunk divergence).
    with torch.no_grad():
      p, t = pred.view(-1, W, base), tgt.view(-1, W, base)
      l0 = float(loss_fn(p[:, 0], t[:, 0]))
      llast = float(loss_fn(p[:, -1], t[:, -1]))
    return float(loss.detach()), l0, llast

  @torch.no_grad()
  def evaluate():
    o, _ = env.reset()
    total = torch.zeros(N, device=device)
    n_chunks = max(1, -(-cfg.eval_steps // W))  # ceil: fixed sim-time horizon
    acc: dict[str, list[float]] = {}
    for _ in range(n_chunks):
      o, rew, _dones, extras = env.step(student(o))  # chunked open-loop step
      total += rew
      log = extras.get("log", extras) if isinstance(extras, dict) else {}
      for k, v in (log or {}).items():
        if any(s in k.lower() for s in ("success", "lift", "reward")):
          try:
            acc.setdefault(k, []).append(float(v))
          except (TypeError, ValueError):
            pass
    env.reset()  # leave collection state clean
    win_len.zero_()
    logs = {k: sum(vs) / len(vs) for k, vs in acc.items()}  # mean, not last-write
    return float(total.mean()) / n_chunks, logs

  end_it = start_it + cfg.iters
  print(f"[distill] mode={'DAgger' if cfg.dagger else 'BC'} task={cfg.task_id} "
        f"chunk={W} num_envs={N} device={device} iters {start_it}->{end_it}")
  for it in range(start_it + 1, end_it + 1):
    for _ in range(cfg.collect_every):
      collect_step()
    if rb_n < cfg.batch_size:
      continue
    loss, l0, llast = update()
    if it % cfg.log_every == 0:
      writer.add_scalar("train/loss", loss, it)
      writer.add_scalar("train/loss_step0", l0, it)
      writer.add_scalar("train/loss_steplast", llast, it)
      writer.add_scalar("train/buffer", rb_n, it)
      print(f"[distill] it {it}/{end_it}  loss {loss:.4f}  buffer {rb_n}/{cap}")
    if it % cfg.eval_every == 0:
      rew_per_chunk, logs = evaluate()
      writer.add_scalar("eval/reward_per_chunk", rew_per_chunk, it)
      for k, v in logs.items():
        writer.add_scalar(f"eval/{k.split('/')[-1]}", v, it)
      writer.flush()
      extra = "  ".join(f"{k}={v:.3f}" for k, v in sorted(logs.items()))
      print(f"[distill][eval] it {it}  open-loop reward/chunk {rew_per_chunk:.3f}  {extra}")
    if it % cfg.save_every == 0:
      save_ckpt(str(out_dir / f"model_{it}.pt"), it)
      print(f"[distill] saved {out_dir / f'model_{it}.pt'}")

  save_ckpt(str(out_dir / f"model_{end_it}.pt"), end_it)
  writer.close()
  print(f"[distill] done. final -> {out_dir / f'model_{end_it}.pt'}")
  print(f"[distill] tensorboard --logdir {out_dir.parent}")


def main() -> None:
  run(tyro.cli(DistillConfig))


if __name__ == "__main__":
  main()
