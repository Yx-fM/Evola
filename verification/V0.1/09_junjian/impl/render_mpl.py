"""Matplotlib renderer - for publication-quality figures and replay analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

COLOR_MAP = {
    0: "#F5F5DC",
    1: "#4A4A4A",
    2: "#4CAF50",
    3: "#F44336",
    4: "#2196F3",
}
LABEL_MAP = {0: ".", 1: "#", 2: "F", 3: "X", 4: "A"}


def render_grid(world, ax=None, title=None, show_vision=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    grid = world.get_full_grid()
    h, w = grid.shape
    cmap = mcolors.ListedColormap([COLOR_MAP[i] for i in range(5)])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(np.clip(grid, 0, 4), cmap=cmap, norm=norm, interpolation="nearest")

    for r in range(h):
        for c in range(w):
            cell_type = grid[r, c]
            if cell_type != 0:
                ax.text(c, r, LABEL_MAP.get(cell_type, "?"),
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if cell_type in (1, 3, 4) else "black")

    if show_vision:
        R = world.cfg.vision_range
        ar, ac = world.agent_pos
        rect = Rectangle(
            (ac - R - 0.5, ar - R - 0.5), 2 * R + 1, 2 * R + 1,
            linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)

    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    if title:
        ax.set_title(title)
    return ax


def render_obs(world, ax=None, title=None):
    obs = world.get_obs()
    R = world.cfg.vision_range
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    cmap = mcolors.ListedColormap([COLOR_MAP[i] for i in range(5)])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(np.clip(obs, 0, 4), cmap=cmap, norm=norm, interpolation="nearest")

    for r in range(2 * R + 1):
        for c in range(2 * R + 1):
            cell_type = obs[r, c]
            if cell_type != 0:
                ax.text(c, r, LABEL_MAP.get(cell_type, "?"),
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white" if cell_type in (1, 3, 4) else "black")

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
