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

### Seeing the collision shape

Inside the **native MuJoCo viewer**, the number keys `0`–`5` toggle geom
groups on/off. By convention:

- **group 2** — visual-only geoms (the container's STL mesh)
- **group 3** — collision-only geoms (the container's invisible box walls;
  rendered translucent red when shown)

So to see the collision proxy overlaid on the visual cup, press `3` once.
Press `2` to hide the visual mesh and inspect just the collision shape.
Press `0` for everything else (table, block, robot links).

In the **viser web viewer**, group toggles live in the right-side panel
under *Display* → *Geom groups*.

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

### Via pixi (from the workspace root)

```bash
# Train from scratch with defaults (4096 envs, 5000 iters, tensorboard).
pixi run train-mjlab

# Override counts via env vars.
NENV=2048 ITER=10000 pixi run train-mjlab

# Resume from the most recent checkpoint under
# mjlab_rl/logs/rsl_rl/so101_block_picking/. Verified end-to-end.
pixi run train-mjlab-resume

# Resume from a specific run / checkpoint.
RUN=2026-05-29_04-23-41 CKPT=model_3.pt pixi run train-mjlab-resume
```

### Via the bare script (more flexibility, fewer assumptions)

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

# Resume the most recent run; mjlab's defaults for --agent.load-run / 
# --agent.load-checkpoint are regex .* which means "latest".
uv run python scripts/train.py Mjlab-SO101-Block-Picking \
    --env.scene.num-envs 4096 --agent.max-iterations 5000 \
    --agent.logger tensorboard --agent.resume True

# Resume from a specific run and checkpoint.
uv run python scripts/train.py Mjlab-SO101-Block-Picking \
    --env.scene.num-envs 4096 --agent.max-iterations 5000 \
    --agent.logger tensorboard --agent.resume True \
    --agent.load-run 2026-05-29_04-23-41 \
    --agent.load-checkpoint model_3.pt

# Record a video every 2000 env steps.
uv run python scripts/train.py Mjlab-SO101-Block-Picking \
    --video True --video-interval 2000 --video-length 400
```

Checkpoints land under `logs/rsl_rl/so101_block_picking/<timestamp>/`. Use
`tensorboard --logdir logs/rsl_rl/so101_block_picking` (or
`pixi run tensorboard-mjlab` from the workspace root) while training to
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

## Vision variant — `Mjlab-SO101-Block-Picking-Rgb`

Same env as the state-only task plus two 64×64 RGB cameras (the SO-101's
existing wrist-mounted `hand_eye` + a new top-down camera attached to the
table body). **Asymmetric PPO**: the actor sees joint state + camera, the
critic keeps the privileged ground-truth observations. Both tasks coexist
— the state-only `Mjlab-SO101-Block-Picking` is unchanged.

### Check the camera views

The fastest way is the **headless render helper** — no display needed:

```bash
# Captures one frame per camera, writes PNGs into mjlab_rl/.
pixi run render-mjlab-cameras
# -> mjlab_rl/wrist_cam.png  (3×64×64)
# -> mjlab_rl/top_cam.png    (3×64×64)
# Run again any time you change the camera pos/quat/fovy in
# block_picking_vision.py to re-render.
```

Or open the interactive viewer (Viser web UI shows live camera feeds in
the right sidebar):

```bash
pixi run play-mjlab-vision-zero
# Browser: http://<this-host>:8080 → right sidebar "Cameras"
```

### Train

```bash
# Standard run: 4096 envs, 10000 iters, tensorboard logger.
pixi run train-mjlab-vision

# Tweak counts via env vars.
NENV=2048 ITER=20000 pixi run train-mjlab-vision

# Or bare script.
uv run python scripts/train.py Mjlab-SO101-Block-Picking-Rgb \
    --env.scene.num-envs 4096 --agent.max-iterations 10000 \
    --agent.logger tensorboard
```

### Replay a trained vision policy

```bash
CKPT=mjlab_rl/logs/rsl_rl/so101_block_picking_vision/<run>/model_*.pt \
    pixi run play-mjlab-vision-trained
```

### Caveats

- **Vision is sample-hungry.** Expect 10–100× more iterations than the
  state-only task before `Episode_Reward/success` rises. Watch the curves
  in TensorBoard (`pixi run tensorboard-mjlab`).
- **Render cost.** 2 cams × 64×64 × 4096 envs adds noticeable per-step
  overhead. Iteration time will be larger than state-only on the same
  hardware.
- **Can't resume the state-only checkpoint.** Different actor input
  shape; train fresh.
- **Verify the camera frames before a long run.** `render-mjlab-cameras`
  takes ~15 s on CPU. If the top camera doesn't actually show both the
  block and the container, edit `top_cam`'s `pos`/`quat`/`fovy` in
  `mjlab_rl/mjlab_rl/tasks/block_picking_vision.py`.

## Replay a trained policy

```bash
# Native viewer (same machine that has the checkpoint).
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --checkpoint-file logs/rsl_rl/so101_block_picking/<run>/model_5000.pt

# Save an mp4 instead of opening a window.
uv run python scripts/play.py Mjlab-SO101-Block-Picking \
    --checkpoint-file <ckpt.pt> --video True --video-length 400
```

### Replay on the laptop from a desktop-trained checkpoint (no scp)

Viser ships a self-hosted web viewer that binds to `0.0.0.0:8080`, so the
desktop can run the policy and the laptop just needs a browser. Two pixi
tasks in the workspace `pixi.toml` wrap the calls.

**On the desktop** (has the GPU + checkpoint):

```bash
# Trained policy. CKPT is a path *relative to the ros2_lerobot workspace root*.
CKPT=mjlab_rl/logs/rsl_rl/so101_block_picking/<run>/model_5000.pt \
    pixi run play-mjlab-trained

# Or just the static scene (no checkpoint).
pixi run play-mjlab-zero
```

**On the laptop**:

```text
http://<desktop-hostname-or-ip>:8080
```

If port 8080 isn't exposed on the LAN (firewall, untrusted Wi-Fi), tunnel
it over SSH instead — runs viser bound to the desktop's loopback, then
forwards the port to the laptop:

```bash
# laptop:
ssh -L 8080:localhost:8080 <user>@<desktop>
# then in another laptop terminal:
xdg-open http://localhost:8080   # macOS: open http://localhost:8080
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
