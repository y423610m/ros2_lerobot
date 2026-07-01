"""Convert a trained mjlab checkpoint (model_*.pt) into a TorchScript .jit file
that can be loaded outside the mjlab simulator (e.g. from a ROS 2 node) via
``torch.jit.load(path)``.

The exported module embeds the obs normalizer and exposes a ``forward(obs_1d,
obs_2d_list)`` whose shapes match the actor's observation groups at training
time. For the SO-101 vision policy that's:
    obs_1d : (B, 12)            # actor group: joint_pos_rel + last_action
    obs_2d : [(B, 6, 64, 64)]   # camera group: wrist_rgb cat top_rgb / 255

Example:
    uv run python scripts/export_to_jit.py Mjlab-SO101-Block-Picking-Rgb \\
        --checkpoint-file logs/rsl_rl/so101_block_picking_vision/<run>/model_5000.pt
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_rl.envs.chunked_env import ChunkedVecEnvWrapper

# Name of the action term to read the control contract off of (see the
# ``actions`` dict in mjlab_rl/tasks/block_picking.py).
ACTION_TERM = "joint_pos"


@dataclass(frozen=True)
class ExportConfig:
  checkpoint_file: str
  out_dir: str | None = None
  filename: str = "policy.jit"
  device: str = "cpu"


def _per_joint(value, n: int) -> list[float]:
  """Normalize a scale/offset/max-rel (float or (num_envs, n) tensor) to a
  per-joint list of length ``n``."""
  if isinstance(value, torch.Tensor):
    return [float(x) for x in value[0].detach().cpu().tolist()]
  return [float(value)] * n


def _build_metadata(base_env: ManagerBasedRlEnv, chunk_size: int = 1) -> dict:
  """The deployment contract, read off the live action term so it exactly
  matches training: target = target_ref + action_scale * action, then clamped to
  present ± max_relative_target. Embedded in the .jit so the deployment node
  doesn't have to hardcode (and hand-sync) any of it.

  ``chunk_size`` is the number of action steps the actor emits per inference
  (1 = closed-loop). The deploy node reshapes the (1, chunk_size*n) output into a
  chunk of ``chunk_size`` n-dim actions and streams them one per control step."""
  act = base_env.action_manager.get_term(ACTION_TERM)
  names = list(act._target_names)
  n = len(names)
  return {
    "joint_names": names,
    "target_ref": _per_joint(act._offset, n),  # = home / default_joint_pos
    "action_scale": _per_joint(act._scale, n),
    "max_relative_target": _per_joint(act._max_rel, n),
    "control_dt": float(base_env.step_dt),  # 0.02 s -> 50 Hz
    "camera_hw": [64, 64],
    "chunk_size": int(chunk_size),
  }


def run(task_id: str, cfg: ExportConfig) -> None:
  ckpt = Path(cfg.checkpoint_file).expanduser().resolve()
  if not ckpt.exists():
    raise FileNotFoundError(ckpt)
  out_dir = Path(cfg.out_dir).expanduser().resolve() if cfg.out_dir else ckpt.parent
  out_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}
  agent_cfg = load_rl_cfg(task_id)

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  # ChunkedVecEnvWrapper reads CHUNK_SIZE from the env: with CHUNK_SIZE>1 the actor
  # head is sized to chunk_size*n so a chunked checkpoint loads. Run export with the
  # SAME CHUNK_SIZE used for training.
  env = ChunkedVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=cfg.device)

  runner.export_policy_to_jit(str(out_dir), filename=cfg.filename)
  jit_path = out_dir / cfg.filename
  print(f"[export_to_jit] wrote {jit_path}")

  # Embed the deployment contract inside the .jit (single file). The base
  # rsl-rl export saves without _extra_files, so re-load and re-save with it.
  chunk_size = env.num_actions // base_env.action_manager.total_action_dim
  metadata = _build_metadata(base_env, chunk_size=chunk_size)
  module = torch.jit.load(str(jit_path), map_location="cpu")
  module.save(str(jit_path), _extra_files={"metadata.json": json.dumps(metadata, indent=2)})
  print(f"[export_to_jit] embedded metadata: {metadata}")


def main() -> None:
  task_id = tyro.cli(str, args=None, default=None, prog="export_to_jit", description="Task id")
  raise SystemExit("Use the parsed entrypoint below — main() is for symmetry only.")


if __name__ == "__main__":
  import sys

  if len(sys.argv) < 2:
    sys.exit(
      "usage: export_to_jit.py <task-id> --checkpoint-file <path> "
      "[--out-dir <dir>] [--filename policy.jit] [--device cpu|cuda]"
    )
  task_id = sys.argv[1]
  cfg = tyro.cli(ExportConfig, args=sys.argv[2:])
  run(task_id, cfg)
