"""Train the SO-101 block-picking task with mjlab + RSL-RL.

Thin wrapper around ``mjlab.scripts.train`` that imports the project so the
task gets registered before tyro parses arguments.

Example:
  uv run python scripts/train.py --task Mjlab-SO101-Block-Picking \\
      --env.scene.num-envs 4096 --agent.max-iterations 5000
"""

from __future__ import annotations

import mjlab_rl  # noqa: F401  (registers the task)
from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
