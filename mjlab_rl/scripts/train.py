"""Train the SO-101 block-picking task with mjlab + RSL-RL.

Thin wrapper around ``mjlab.scripts.train`` that imports the project so the
task gets registered before tyro parses arguments.

Example:
  uv run python scripts/train.py --task Mjlab-SO101-Block-Picking \\
      --env.scene.num-envs 4096 --agent.max-iterations 5000
"""

from __future__ import annotations

import mjlab_rl  # noqa: F401  (registers the task)
import mjlab.scripts.train as _train
from mjlab_rl.envs.chunked_env import ChunkedVecEnvWrapper

# Swap in the chunking-aware env wrapper. It behaves exactly like the stock
# RslRlVecEnvWrapper unless CHUNK_SIZE>1 is set in the environment, in which case
# the policy emits/executes CHUNK_SIZE-step action chunks. e.g.:
#   CHUNK_SIZE=50 uv run python scripts/train.py Mjlab-SO101-Block-Picking-Rgb-DR0 ...
_train.RslRlVecEnvWrapper = ChunkedVecEnvWrapper

if __name__ == "__main__":
  _train.main()
