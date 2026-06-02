"""Monkey patches we apply to upstream viser/mjviser at import time.

These exist because mjviser hardcodes some defaults that override the
``ViewerConfig`` / ``stat.center`` settings we set on the env.
"""

from __future__ import annotations

from mjviser.scene import ViserMujocoScene

# By default ViserMujocoScene.__init__ does `camera_tracking_enabled = True`.
# When that flag is True, the viser camera forces its look_at to either
# the tracked body's origin or, when no body is tracked, the world
# origin (0, 0, 0). That makes our `spec.stat.center` override and
# ``ViewerConfig.lookat`` useless on the first frame.
#
# Patch the init so the default is OFF, restoring the documented
# "viser orbits around stat.center" behaviour. Users can still flip the
# "Track camera" checkbox on inside the viser GUI if they want body
# tracking back.
_orig_scene_init = ViserMujocoScene.__init__


def _patched_scene_init(self, *args, **kwargs):
  _orig_scene_init(self, *args, **kwargs)
  self.camera_tracking_enabled = False


ViserMujocoScene.__init__ = _patched_scene_init
