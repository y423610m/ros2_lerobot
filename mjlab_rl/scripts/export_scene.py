"""Export a compiled scene to a directory (scene.xml + meshes).

Wraps :mod:`mjlab.scripts.export_scene` but imports :mod:`mjlab_rl` first so
the task ID is known to the registry.
"""

from __future__ import annotations

import mjlab_rl  # noqa: F401
from mjlab.scripts.export_scene import main

if __name__ == "__main__":
  main()
