"""Runner that warm-starts across a *change in chunk size* (action-space size).

The chunk-size curriculum (dr0=1 → dr1=10 → dr2=25 → dr3=50) resumes each stage
from the previous stage's checkpoint. Growing the chunk grows the actor's output
head (``mlp.<last>`` + ``distribution.std_param``, first-dim = ``chunk*6``), so a
plain ``strict`` load fails on the shape mismatch. Everything else — the CNN
backbone (``cnns.*``), the trunk (``mlp.0..N-1``), the input layer, the obs
normalizer, and the whole critic — is chunk-independent and transfers verbatim
(verified: only ``mlp.<last>.{weight,bias}`` and ``distribution.std_param`` change
shape across chunk sizes).

Because the head is *action-major* (``ChunkedVecEnvWrapper.step`` does
``actions.view(b, chunk, base)``), the first ``old_chunk*base`` output rows ARE
actions ``0..old_chunk-1``. So a growing chunk is a *prefix*: we copy the old
head rows into the new head's prefix and keep the fresh init for the extension.
The policy keeps its learned short-horizon behavior and only learns to extend the
tail; the expensive vision backbone is preserved.

Same-chunk resume (and the export path, which loads a chunk=N ckpt into a chunk=N
actor) hits no shape mismatch and delegates to the base ``load`` unchanged.
"""

from __future__ import annotations

import torch

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner


def _migrate_actor_state_dict(loaded_dict: dict) -> dict:
  """Apply the same key migrations the base loader does, so shape comparison and
  prefix-copy see current-format keys. Mirrors ``MjlabOnPolicyRunner.load``:
  legacy ``model_state_dict`` (actor.*→mlp.*, actor_obs_normalizer.*→
  obs_normalizer.*) and rsl-rl 4.x→5.x (``std``→``distribution.std_param``)."""
  if "model_state_dict" in loaded_dict:
    model_state_dict = loaded_dict.pop("model_state_dict")
    actor_state_dict: dict = {}
    critic_state_dict: dict = {}
    for key, value in model_state_dict.items():
      if key.startswith("actor."):
        actor_state_dict[key.replace("actor.", "mlp.")] = value
      elif key.startswith("actor_obs_normalizer."):
        actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = value
      elif key in ("std", "log_std"):
        actor_state_dict[key] = value
      if key.startswith("critic."):
        critic_state_dict[key.replace("critic.", "mlp.")] = value
      elif key.startswith("critic_obs_normalizer."):
        critic_state_dict[key.replace("critic_obs_normalizer.", "obs_normalizer.")] = value
    loaded_dict["actor_state_dict"] = actor_state_dict
    loaded_dict["critic_state_dict"] = critic_state_dict

  actor_sd = loaded_dict.get("actor_state_dict", {})
  if "std" in actor_sd:
    actor_sd["distribution.std_param"] = actor_sd.pop("std")
  if "log_std" in actor_sd:
    actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")
  return loaded_dict


class WarmStartOnPolicyRunner(ManipulationOnPolicyRunner):
  """ManipulationOnPolicyRunner + prefix-preserving warm-start on chunk change."""

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = _migrate_actor_state_dict(
      torch.load(path, map_location=map_location, weights_only=False)
    )
    ckpt_actor = loaded_dict.get("actor_state_dict", {})
    live = self.alg._raw_actor.state_dict()

    mismatched = {
      k: (tuple(v.shape), tuple(live[k].shape))
      for k, v in ckpt_actor.items()
      if k in live and live[k].shape != v.shape
    }
    if not mismatched:
      # Same chunk size (or export): nothing special to do.
      return super().load(path, load_cfg, strict, map_location)

    # --- prefix-preserving warm-start ---
    new_sd = dict(live)  # starts from the freshly-initialized new actor
    for k, v in ckpt_actor.items():
      if k not in live:
        continue
      if live[k].shape == v.shape:
        new_sd[k] = v  # backbone + trunk + input + normalizer: verbatim
      else:
        merged = live[k].clone()  # keep fresh init for the extension...
        n = v.shape[0]
        merged[:n] = v  # ...and copy the old head rows into the prefix
        new_sd[k] = merged
    self.alg._raw_actor.load_state_dict(new_sd, strict=True)

    # Critic is chunk-independent -> full transfer.
    critic_sd = loaded_dict.get("critic_state_dict", {})
    if critic_sd:
      self.alg._raw_critic.load_state_dict(critic_sd, strict=strict)

    # Optimizer state is sized to the old head -> drop it (fresh optimizer).
    # Iteration counter and common_step_counter are NOT restored: this is a new
    # curriculum stage, so training starts at iteration 0.
    kept = next(iter(mismatched.values()))[0][0]  # old head size = old_chunk * base
    print(
      f"[warm-start] chunk head change {mismatched}: transferred backbone+trunk+critic, "
      f"extended head prefix (kept first {kept} action-dims), reset optimizer + iteration=0"
    )
    return loaded_dict.get("infos", None)
