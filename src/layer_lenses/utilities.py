"""Small shared helpers for notebooks and experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def complete_binary_internal_node_positions(
    num_internal_nodes: int,
) -> dict[int, tuple[float, float]]:
    """Layout positions for internal nodes of a complete binary tree (breadth order).

    Node ids are ``0 .. num_internal_nodes - 1`` as in COB-ODT / notebook trees.
    Raises if ``num_internal_nodes`` is not ``2**depth - 1`` for integer ``depth``.
    """
    depth_float = np.log2(num_internal_nodes + 1)
    depth = int(depth_float)
    if (2**depth) - 1 != num_internal_nodes:
        raise ValueError(
            "num_internal_nodes must describe a complete binary tree, "
            f"got {num_internal_nodes}."
        )

    positions: dict[int, tuple[float, float]] = {}
    for node_id in range(num_internal_nodes):
        level = int(np.floor(np.log2(node_id + 1)))
        level_start = (2**level) - 1
        index_in_level = node_id - level_start
        x = (index_in_level + 0.5) / (2**level)
        y = -float(level)
        positions[node_id] = (x, y)
    return positions


def epoch_tree_label_style(
    facecolor: str | tuple[float, ...],
) -> tuple[str, list[Any]]:
    """Return text color and path effects for a label on a colored disk (viridis, etc.)."""
    r, g, b = mcolors.to_rgba(facecolor)[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance < 0.5:
        return "white", [pe.withStroke(linewidth=2.5, foreground="#2a2a2a")]
    return "black", [pe.withStroke(linewidth=2.25, foreground="white")]


def _default_value_label(v: float) -> str:
    if np.isnan(v):
        return "NA"
    if v == np.floor(v):
        return str(int(v))
    return f"{v:.4g}"


def plot_complete_binary_tree_values(
    values: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    value_fmt: Callable[[float], str] | None = None,
    title: str | None = None,
    suptitle: str | None = None,
    cmap: str = "viridis",
    figsize_per_panel: tuple[float, float] = (6.0, 3.8),
    use_dlgn_panel_geometry: bool = False,
    scatter_size: float = 780,
    label_fontsize: float = 9,
    missing_facecolor: str = "lightgray",
) -> tuple[Figure, Axes]:
    """Draw one complete binary tree with a numeric (or missing) label per internal node.

    ``values`` must have length ``2**depth - 1`` (e.g. 31 for depth 5). Use
    ``float(\"nan\")`` for missing nodes; they are drawn in gray with ``\"NA\"``
    unless ``labels`` overrides.

    Styling matches the DLGN / ReLU notebook ODT tree figures: gray edges,
    viridis face color from the finite value range, bold labels with contrast
    outlines.

    If ``use_dlgn_panel_geometry`` is True, the axes box matches a *single panel*
    of ``plot_phase1_odt_hit_epoch_trees`` when ``ncols=2`` (left column of a
    ``(2 * figsize_per_panel[0]) × figsize_per_panel[1]`` figure).

    Parameters
    ----------
    values
        Length ``N = 2**depth - 1``; index ``i`` is ODT internal node id ``i``.
    labels
        Optional per-node strings; length must match ``values`` if given.
    value_fmt
        If ``labels`` is omitted, format finite numbers (not used for NaN).
    """
    arr = np.asarray(values, dtype=float).ravel()
    n = int(arr.size)
    complete_binary_internal_node_positions(n)  # validate tree size

    if labels is not None and len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != values length {n}.")
    if value_fmt is None:
        value_fmt = _default_value_label

    positions = complete_binary_internal_node_positions(n)
    finite = arr[np.isfinite(arr)]
    cmap_obj = plt.get_cmap(cmap)
    norm: plt.Normalize | None
    if finite.size == 0:
        norm = None
    else:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    w, h = figsize_per_panel
    if use_dlgn_panel_geometry:
        _wspace = float(plt.matplotlib.rcParams["figure.subplot.wspace"])
        fig = plt.figure(figsize=(w * 2, h), layout="constrained")
        gs = fig.add_gridspec(1, 2, figure=fig, wspace=_wspace)
        ax = fig.add_subplot(gs[0, 0])
    else:
        fig, ax = plt.subplots(figsize=(w, h), layout="constrained")

    for parent_id in range(n):
        for child_id in (2 * parent_id + 1, 2 * parent_id + 2):
            if child_id >= n:
                continue
            x0, y0 = positions[parent_id]
            x1, y1 = positions[child_id]
            ax.plot([x0, x1], [y0, y1], color="0.75", linewidth=1.2, zorder=1)

    for node_id, (x, y) in positions.items():
        v = float(arr[node_id])
        if labels is not None:
            label_str = str(labels[node_id])
        elif np.isnan(v):
            label_str = "NA"
        else:
            label_str = value_fmt(v)

        if np.isnan(v) or norm is None:
            facecolor = missing_facecolor
        else:
            facecolor = cmap_obj(norm(v))

        ax.scatter(
            [x],
            [y],
            s=scatter_size,
            color=facecolor,
            edgecolor="black",
            linewidth=1.0,
            zorder=2,
        )
        txt_color, path_eff = epoch_tree_label_style(facecolor)
        ax.text(
            x,
            y,
            label_str,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color=txt_color,
            path_effects=path_eff,
            zorder=3,
        )

    ax.set_axis_off()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-(int(np.log2(n + 1)) - 1) - 0.6, 0.6)
    if title is not None:
        ax.set_title(title)
    if suptitle is not None:
        fig.suptitle(suptitle, y=1.02, fontsize=14)
    return fig, ax
