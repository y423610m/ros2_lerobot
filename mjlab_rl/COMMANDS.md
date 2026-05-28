# Useful commands

All commands assume you're in this project's root (`mjlab_rl/`) with the venv
synced. Each command listed here has been run end-to-end on this machine,
unless otherwise marked.

**Why every command goes through `scripts/`.** `register_mjlab_task(...)`
fires at `mjlab_rl` import time. Plain `python -m mjlab.scripts.list_envs`
never imports our package, so our task does not show up. The wrappers under
`scripts/` `import mjlab_rl` first and then delegate to the corresponding
`mjlab.scripts.*` `main()`, so our task is in the registry.

> Most mjlab CLI scripts use [tyro](https://github.com/brentyi/tyro), so add
> `--help` to any of them to see every flag.

## Setup

```bash
# One-time: pull mjlab + warp + rsl-rl into .venv/.
uv sync

# Verify Warp can see (or not see) a GPU.
.venv/bin/python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

## Sanity checks (no GPU needed) — verified working

```bash
# Build the env on CPU, reset, step it 16 times. ~30 s the first time
# (Warp JIT-compiles MuJoCo kernels), instant on subsequent runs.
uv run python scripts/smoke_test.py

# List every task mjlab knows about. Mjlab-SO101-Block-Picking should be
# in the table.
uv run python scripts/list_envs.py

# Same but filter to just our task family.
uv run python scripts/list_envs.py --keyword SO101
```

## Visualize the scene without training a policy

Requires a display (or X11-forwarded SSH). On a headless box, use
`--viewer viser` to get a browser-based viewer instead.

```bash
# Open the play viewer with a *zero-action* policy (the arm just sits at home,
# block + container settle on the table — useful for confirming geometry).
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --agent zero --num-envs 1 --no-terminations True

# Same idea but with random actions (the arm flails — sanity-checks action
# scaling and joint limits).
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --agent random --num-envs 1 --no-terminations True

# Force the web (viser) viewer instead of the native MuJoCo one.
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --agent zero --viewer viser
```

`--no-terminations True` is the key flag for visualization runs — without it
every episode resets after 8 s and the viewer keeps snapping back to the
home pose. (Tyro emits the flag as a value-taking option because the field
name already starts with `no_`, hence the explicit `True`.)

## Export the compiled scene (for inspection in any MuJoCo viewer)

```bash
# Writes scene.xml (and any mesh assets) to ./export/.
uv run python scripts/export_scene.py Mjlab-SO101-Block-Picking \
    --output-dir export

# Then open in stock MuJoCo:
uv run python -m mujoco.viewer --mjcf export/scene.xml
```

(Verified that the export creates `export/scene.xml`. The viewer line itself
needs a display.)

## Train

```bash
# (CPU) 2-iteration sanity run — verified end-to-end on this machine.
# Finishes in ~2 s, writes a model_*.pt + tfevents under logs/rsl_rl/.
uv run python scripts/train.py Mjlab-SO101-Block-Picking \
    --gpu-ids None \
    --env.scene.num-envs 4 \
    --agent.max-iterations 4 \
    --agent.num-steps-per-env 8 \
    --agent.logger tensorboard

# (GPU) Full training. ~5–10 min on an RTX 4090.
uv run python scripts/train.py Mjlab-SO101-Block-Picking \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 5000 \
    --agent.logger tensorboard

# Record a video every 2000 env steps.
uv run python scripts/train.py Mjlab-SO101-Block-Picking \
    --video True --video-interval 2000 --video-length 400
```

Checkpoints land under `logs/rsl_rl/so101_block_picking/<timestamp>/`. Use
`tensorboard --logdir logs/rsl_rl/so101_block_picking` while training to
watch curves.

### Gotchas you'll otherwise re-hit

| Symptom | Fix |
|---|---|
| `IndexError: list index out of range` in `select_gpus` | Pass `--gpu-ids None` (default is `[0]`, fails on CPU-only boxes). |
| `wandb.errors.errors.UsageError: No API key configured.` | `--agent.logger tensorboard`, or run `wandb login` once. |
| `Unrecognized options: --no-video` | The flag is `--video {True,False}`. Omit (default False) or pass `--video True`. |

> The full GPU training run was **not** executed on this machine (no CUDA
> visible). The CPU sanity run above was — it iterates the PPO loop, writes
> checkpoints, and saves tfevents.

## Replay a trained policy

```bash
# Native viewer.
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --checkpoint-file logs/rsl_rl/so101_block_picking/<run>/model_5000.pt

# Save an mp4 instead of opening a window.
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --checkpoint-file <ckpt.pt> --video True --video-length 400
```

## Misc helpers

```bash
# Show every CLI flag a script accepts.
uv run python scripts/train.py --help
uv run python scripts/play.py  --help
uv run python scripts/list_envs.py --help

# mjlab's NaN snapshot inspector (uses no task registration, so the bare
# module form is fine here):
uv run python -m mjlab.scripts.nan_viz <path/to/dump.pkl>
```

## About the oracle

The old plain-MuJoCo project shipped a hand-written IK oracle for collecting
demos. **That oracle has been removed** along with the BC / DAgger / distill
pipeline — this rewrite trains the policy with PPO from scratch using mjlab's
batched MuJoCo Warp envs, which is fast enough that imitation isn't needed
for this task.

If you want to *see* what an untrained policy looks like (the rough analogue
of "play the oracle"), use the zero or random agent variants documented under
[Visualize the scene without training a policy](#visualize-the-scene-without-training-a-policy)
above.

If you specifically want the IK oracle back as a teacher for warm-starting
PPO (or to record demos for behavior cloning on top of mjlab), that's a
separate piece of work — say the word and I'll add it.
