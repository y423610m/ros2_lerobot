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

from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

import mjlab_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass(frozen=True)
class ExportConfig:
  checkpoint_file: str
  out_dir: str | None = None
  filename: str = "policy.jit"
  device: str = "cpu"


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

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=cfg.device)

  runner.export_policy_to_jit(str(out_dir), filename=cfg.filename)
  print(f"[export_to_jit] wrote {out_dir / cfg.filename}")


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
