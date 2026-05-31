"""List registered mjlab tasks (including this project's task).

Wraps :mod:`mjlab.scripts.list_envs` but imports :mod:`mjlab_rl` first so our
``register_mjlab_task(...)`` call has run by the time the script enumerates
the registry.
"""

from __future__ import annotations

import mjlab_rl  # noqa: F401  (registers Mjlab-SO101-Block-Picking)
from mjlab.scripts.list_envs import main

if __name__ == "__main__":
  main()
