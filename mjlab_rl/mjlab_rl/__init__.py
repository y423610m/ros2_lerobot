"""mjlab-based RL task: SO-101 arm picks a block and drops it into a container."""

from mjlab_rl import _viser_patches  # noqa: F401  (monkey-patches mjviser defaults)
from mjlab_rl.tasks import block_picking  # noqa: F401  (registers the task)
from mjlab_rl.tasks import block_picking_vision  # noqa: F401  (registers the vision task)
