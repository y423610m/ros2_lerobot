"""Roll out a trained SO-101 block-picking policy (or zero/random) in the viewer.

Example:
  uv run python scripts/play.py --task Mjlab-SO101-Block-Picking \\
      --checkpoint-file logs/rsl_rl/so101_block_picking/<run>/model_*.pt
"""

from __future__ import annotations

import mjlab_rl  # noqa: F401  (registers the task)
from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
