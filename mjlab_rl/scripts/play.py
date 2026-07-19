"""Roll out a trained SO-101 block-picking policy (or zero/random) in the viewer.

Example:
  uv run python scripts/play.py --task Mjlab-SO101-Block-Picking \\
      --checkpoint-file logs/rsl_rl/so101_block_picking/<run>/model_*.pt
"""

from __future__ import annotations

import torch

import mjlab_rl  # noqa: F401  (registers the task)
import mjlab.scripts.play as _play
from mjlab.scripts.play import main
from mjlab_rl.envs.chunked_env import ChunkedVecEnvWrapper

# Mirror scripts/train.py: swap in the chunking-aware wrapper so a chunked
# checkpoint (actor head = CHUNK_SIZE*6) loads. With CHUNK_SIZE unset/1 it's the
# stock wrapper, so closed-loop checkpoints and the zero/random agents are
# unaffected. Play a chunked policy with e.g.
#   CHUNK_SIZE=50 pixi run play-mjlab-vision-dr3 CKPT=...
_play.RslRlVecEnvWrapper = ChunkedVecEnvWrapper


class _ChunkStreamPolicy:
  """Stream a chunked policy one sub-action per viewer step (deploy semantics).

  The viewer treats one env.step() as one display tick. If the whole chunk
  executes inside a single step() call, all 50 control steps (1 s of sim) happen
  invisibly between frames -> the pose/RGB update ~1x per second. Instead, do
  what the ROS node does: infer once per chunk from the boundary obs, buffer the
  chunk, and emit ONE 6-dim action per step (ChunkedVecEnvWrapper passes a
  base-dim action straight through), so every control step is simulated AND
  rendered by the viewer.
  """

  def __init__(self, policy, chunk: int, base: int):
    self._policy = policy
    self._chunk = chunk
    self._base = base
    self._buf: torch.Tensor | None = None
    self._k = 0

  def __call__(self, obs) -> torch.Tensor:
    if self._buf is None or self._k >= self._chunk:
      flat = self._policy(obs)  # (N, chunk*base), inferred from boundary obs
      self._buf = flat.view(flat.shape[0], self._chunk, self._base)
      self._k = 0
    a = self._buf[:, self._k]
    self._k += 1
    return a


def _chunk_aware(viewer_cls):
  """Wrap a viewer class so chunked policies are streamed per-step."""

  class ChunkAwareViewer(viewer_cls):
    def __init__(self, env, policy, **kwargs):
      chunk = getattr(env, "_chunk", 1)
      if chunk > 1:
        policy = _ChunkStreamPolicy(policy, chunk, env._base_action_dim)
      super().__init__(env, policy, **kwargs)

  return ChunkAwareViewer


_play.ViserPlayViewer = _chunk_aware(_play.ViserPlayViewer)
_play.NativeMujocoViewer = _chunk_aware(_play.NativeMujocoViewer)

if __name__ == "__main__":
  main()
