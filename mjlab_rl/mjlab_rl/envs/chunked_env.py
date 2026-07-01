"""Action-chunking wrapper for the rsl-rl PPO training loop.

Standard closed-loop RL here is one 6-dim action per 50 Hz control step. With
*action chunking* the policy instead emits a chunk of ``CHUNK_SIZE`` future
actions at once, and the env executes them open-loop before the next inference —
i.e. one policy decision = one chunk = ``CHUNK_SIZE`` control steps (an
action-repeat-with-sequence / semi-MDP). The actor's output grows from ``6`` to
``CHUNK_SIZE * 6``; the chunk reward is the sum over the executed sub-steps.

This is a drop-in subclass of :class:`mjlab.rl.RslRlVecEnvWrapper`: with
``CHUNK_SIZE`` unset or ``1`` it behaves exactly like the base wrapper (no
chunking), so the same training script serves both closed-loop and chunked runs.
Enable via the env var, e.g. ``CHUNK_SIZE=50``.

Known approximations (documented, acceptable for a first implementation): mjlab
auto-resets an env the moment it terminates, so if an env finishes partway
through a chunk the remaining sub-actions land on the freshly-reset episode. We
mitigate by (a) not accumulating reward for an env after it is done in the chunk,
and (b) zeroing the remaining sub-actions of a done env (→ home-ward target) so
the bleed into the next episode is benign rather than random. Truncation
bootstrapping uses the chunk-boundary obs.
"""

from __future__ import annotations

import os

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper


def chunk_size_from_env() -> int:
  """Chunk length from ``CHUNK_SIZE`` (default 1 = no chunking)."""
  try:
    n = int(os.environ.get("CHUNK_SIZE", "1"))
  except ValueError:
    n = 1
  return max(1, n)


class ChunkedVecEnvWrapper(RslRlVecEnvWrapper):
  """RslRlVecEnvWrapper + optional action chunking (``CHUNK_SIZE`` env var)."""

  def __init__(self, env, clip_actions: float | None = None):
    super().__init__(env, clip_actions)
    self._chunk = chunk_size_from_env()
    self._base_action_dim = self.unwrapped.action_manager.total_action_dim
    if self._chunk > 1:
      # Actor head is sized from num_actions → grow it to CHUNK_SIZE * base.
      self.num_actions = self._base_action_dim * self._chunk
      self._modify_action_space()  # rebuild action_space with the new size

  def step(self, actions: torch.Tensor):
    if self._chunk <= 1:
      return super().step(actions)

    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

    b = actions.shape[0]
    sub = actions.view(b, self._chunk, self._base_action_dim)

    # Render only the chunk-boundary frame. The policy consumes obs once per
    # chunk, so rendering the 49 interior frames is wasted work — and for a
    # vision task the camera render (sim.sense) dominates rollout time. sim.sense
    # does BVH refit + camera render + raycast; rewards/terminations run earlier
    # off privileged state + contact sensors (updated in sim.step, NOT
    # sim.sense), so skipping it on interior steps leaves the RL signal
    # unchanged. Physics still runs all N steps.
    env = self.unwrapped
    real_sense = env.sim.sense
    real_compute = env.observation_manager.compute

    def _skip_sense(*a, **k):
      return None

    def _skip_compute(*a, **k):
      return env.obs_buf  # stale obs; discarded on interior steps anyway

    total_rew = torch.zeros(b, device=self.device)
    any_term = torch.zeros(b, dtype=torch.bool, device=self.device)
    any_trunc = torch.zeros(b, dtype=torch.bool, device=self.device)
    active = torch.ones(b, dtype=torch.bool, device=self.device)  # not-yet-done this chunk

    obs_dict: dict = {}
    extras: dict = {}
    try:
      for k in range(self._chunk):
        last = k == self._chunk - 1
        # interior steps: skip camera render + obs compute; render only the last.
        env.sim.sense = real_sense if last else _skip_sense
        env.observation_manager.compute = real_compute if last else _skip_compute
        a = sub[:, k]
        if not bool(active.all()):
          # freeze done envs to zero action (=> home-ward target) to keep the
          # bleed into their auto-reset episode benign.
          a = a.clone()
          a[~active] = 0.0
        obs_dict, rew, terminated, truncated, extras = self.env.step(a)
        total_rew = total_rew + rew * active.to(rew.dtype)
        any_term = any_term | (terminated & active)
        any_trunc = any_trunc | (truncated & active)
        active = active & ~(terminated | truncated)
    finally:
      env.sim.sense = real_sense
      env.observation_manager.compute = real_compute

    dones = (any_term | any_trunc).to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = any_trunc
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      total_rew,
      dones,
      extras,
    )
