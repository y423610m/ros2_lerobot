"""Roll out a trained SO-101 block-picking policy (or zero/random) in the viewer.

Example:
  uv run python scripts/play.py --task Mjlab-SO101-Block-Picking \\
      --checkpoint-file logs/rsl_rl/so101_block_picking/<run>/model_*.pt
"""

from __future__ import annotations

import mjlab_rl  # noqa: F401  (registers the task)
import mjlab.scripts.play as _play
from mjlab.scripts.play import main
from mjlab_rl.envs.chunked_env import ChunkedVecEnvWrapper

# Mirror scripts/train.py: swap in the chunking-aware wrapper so a chunked
# checkpoint (actor head = CHUNK_SIZE*6) loads and its chunk is executed
# open-loop per inference. With CHUNK_SIZE unset/1 it's the stock wrapper, so
# closed-loop checkpoints and the zero/random agents are unaffected. Play a
# chunked policy with e.g. CHUNK_SIZE=50 pixi run play-mjlab-vision-dr3 CKPT=...
_play.RslRlVecEnvWrapper = ChunkedVecEnvWrapper

if __name__ == "__main__":
  main()
