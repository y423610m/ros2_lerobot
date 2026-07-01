"""Render the two cameras of the vision task once and save them to disk.

The simplest way to "look through" the cameras without a viewer. Useful
for confirming that the top-down and wrist cameras are pointed at the
workspace before kicking off a long training run.

Usage:
    uv run python scripts/render_cameras.py
    # writes wrist_cam.png and top_cam.png to the current directory

Optional env vars:
    TASK=Mjlab-SO101-Block-Picking-Rgb   # default; change to test other tasks
    SETTLE=8                              # number of zero-action steps before
                                          # capturing (so block/container have
                                          # settled on the table)
    AUG=1                                 # also dump <cam>_aug.png: the frame
                                          # *after* camera_rgb_aug (blur + noise +
                                          # brightness + glare) — i.e. what the
                                          # policy actually sees. The raw render
                                          # never shows glare (it's obs-only) or
                                          # specular (warp ignores it); shadows
                                          # DO show in the raw render at dr2+.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import mjlab_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_rl.envs import mdp as task_mdp


def main() -> None:
  task_id = os.environ.get("TASK", "Mjlab-SO101-Block-Picking-Rgb")
  settle = int(os.environ.get("SETTLE", "8"))

  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()

  action = torch.zeros((1, env.action_manager.total_action_dim), device="cpu")
  for _ in range(settle):
    env.step(action)

  try:
    from PIL import Image
  except ImportError:
    print(
      "[render_cameras] Pillow not installed — printing per-camera stats only.\n"
      "Install with `uv add pillow` to dump PNGs."
    )
    Image = None

  cam_names = [n for n in env.scene.sensors if "cam" in n.lower()]
  if not cam_names:
    print("[render_cameras] No camera sensors found in scene.")
    return

  for name in cam_names:
    sensor = env.scene[name]
    rgb = sensor.data.rgb  # (B, H, W, 3) uint8
    if rgb is None:
      print(f"[render_cameras] {name}: no RGB data yet")
      continue
    img = rgb[0].cpu().numpy()
    print(
      f"{name}: shape={tuple(img.shape)} "
      f"mean={img.mean():.1f} std={img.std():.1f}"
    )
    if Image is not None:
      out = f"{name}.png"
      Image.fromarray(img).save(out)
      print(f"  -> wrote {out}")

    # Optionally dump the augmented frame the policy actually receives. Pull the
    # configured camera_rgb_aug params straight off the obs term so it matches
    # this task's DR level (glare/blur/noise/brightness all included).
    if os.environ.get("AUG", "0") not in ("0", "false", "no"):
      params = None
      for group_terms, group_names in zip(
        env.observation_manager._group_obs_term_cfgs.values(),
        env.observation_manager._group_obs_term_names.values(),
      ):
        for tcfg, tname in zip(group_terms, group_names):
          if getattr(tcfg.func, "__name__", "") == "camera_rgb_aug" and (
            tcfg.params.get("sensor_name") == name
          ):
            params = tcfg.params
      if params is None:
        print(f"  (no camera_rgb_aug term for {name}; skipping AUG)")
        continue
      aug = task_mdp.camera_rgb_aug(env, **params)  # (1, 3, H, W) float [0,1]
      aug_img = (aug[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
      if Image is not None:
        out = f"{name}_aug.png"
        Image.fromarray(aug_img).save(out)
        print(f"  -> wrote {out} (post-aug: blur+noise+brightness+glare)")


if __name__ == "__main__":
  main()
