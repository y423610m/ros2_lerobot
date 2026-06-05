"""Same as scripts/play.py, but continuously streams the viser camera's
pose (position, wxyz, lookat) to the terminal as you drag/zoom. Throttled
to roughly one print per 150 ms with a small-change deadband so the
output isn't a firehose.

Usage:
    pixi run play-mjlab-vision-zero-print-camera
    # or directly:
    uv run python scripts/play_print_camera.py Mjlab-SO101-Block-Picking-Rgb \\
        --viewer viser --agent zero --num-envs 1 --no-terminations True
"""

from __future__ import annotations

import math
import time

import numpy as np
import viser

import mjlab_rl  # noqa: F401  (registers tasks + patches mjviser camera default)
from mjlab.scripts.play import main


# Patch ViserServer.__init__ so every newly-created server gets a GUI
# button + on-connect hook that prints the camera state on demand.
_orig_init = viser.ViserServer.__init__


def _patched_init(self, *args, **kwargs):
  _orig_init(self, *args, **kwargs)

  # Continuously print the camera pose any time a client moves it.
  # Throttled so dragging doesn't flood the terminal — at most one print
  # per ~150 ms, and only when the camera actually changed enough to
  # matter (>5 mm position or >1° rotation).
  _last_print = {"t": 0.0, "pos": None, "wxyz": None}
  _THROTTLE_S = 0.15
  _POS_EPS = 5e-3
  _QUAT_EPS = math.cos(math.radians(0.5))  # |dot| > this ⇒ same orientation

  def _maybe_print(client: viser.ClientHandle):
    cam = client.camera
    pos = np.asarray(cam.position)
    wxyz = np.asarray(cam.wxyz)
    now = time.time()
    if (now - _last_print["t"]) < _THROTTLE_S:
      return
    if _last_print["pos"] is not None:
      dpos = float(np.linalg.norm(pos - _last_print["pos"]))
      dquat = float(abs(np.dot(wxyz, _last_print["wxyz"])))
      if dpos < _POS_EPS and dquat > _QUAT_EPS:
        return
    _last_print["t"] = now
    _last_print["pos"] = pos
    _last_print["wxyz"] = wxyz
    look_at = np.asarray(cam.look_at)
    print(
      f"[CAM] pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
      f"wxyz=({wxyz[0]:+.3f},{wxyz[1]:+.3f},{wxyz[2]:+.3f},{wxyz[3]:+.3f}) "
      f"lookat=({look_at[0]:+.3f},{look_at[1]:+.3f},{look_at[2]:+.3f})",
      flush=True,
    )

  @self.on_client_connect
  def _(client: viser.ClientHandle):
    print(f"[CAM] client {client.client_id} connected")

    @client.camera.on_update
    def _(_):
      _maybe_print(client)


viser.ViserServer.__init__ = _patched_init


if __name__ == "__main__":
  main()
