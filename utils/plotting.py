"""
Common plotting helpers.

This module provides reusable plotting utilities shared across analysis scripts:

- figure style configuration;
- axis styling (spines, grids, labels);
- figure export to PNG and SVG;
- twin-y axis annotation patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------

def configure_style(usetex: bool = False, fonttype: str = "none") -> None:
    """
    Apply global matplotlib style settings.

    Parameters
    ----------
    usetex:
        Set to True to enable LaTeX text rendering.
    fonttype:
        SVG font rendering type. "none" keeps SVGs vector-friendly.
    """
    mpl.rcParams.update({"text.usetex": usetex, "svg.fonttype": fonttype})


# ---------------------------------------------------------------------
# Axis styling
# ---------------------------------------------------------------------

def style_axes(axes: plt.Axes | Sequence[plt.Axes], hide_top: bool = True,
               hide_right: bool = True, grid_x: bool = True,
               left: bool = False, bottom: bool = False) -> None:
    """
    Apply light styling to one or more axes.

    Parameters
    ----------
    axes:
        A single Axes object or a sequence of Axes objects.
    hide_top, hide_right:
        Whether to hide the top and right spines.
    grid_x:
        Whether to enable x-axis grid lines.
    left, bottom:
        Whether to show the left and bottom spines (useful for publication-style plots).
    """
    if hasattr(axes, "__iter__") and not isinstance(axes, plt.Axes):
        ax_list = axes
    else:
        ax_list = (axes,)

    for ax in ax_list:
        spines_to_hide = []
        if hide_top:
            spines_to_hide.append("top")
        if hide_right:
            spines_to_hide.append("right")
        ax.spines[spines_to_hide].set_visible(False)
        if left:
            ax.spines["left"].set_visible(True)
        if bottom:
            ax.spines["bottom"].set_visible(True)
        if grid_x:
            ax.grid(axis="x")


def style_single_axis(ax: plt.Axes, hide_top: bool = True,
                      hide_right: bool = True, grid: bool = True) -> None:
    """
    Apply light styling to a single axis.

    Parameters
    ----------
    ax:
        The Axes object to style.
    hide_top, hide_right:
        Whether to hide the top and right spines.
    grid:
        Whether to enable grid lines.
    """
    if hide_top:
        ax.spines["top"].set_visible(False)
    if hide_right:
        ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.4)


# ---------------------------------------------------------------------
# Figure export
# ---------------------------------------------------------------------

def export_figure(fig: plt.Figure, base_name: str, dpi: int = 600,
                  bbox_inches: str = "tight", extra_dirs: str | Path | None = None) -> tuple[Path, Path]:
    """
    Export a figure to both PNG and SVG formats.

    Parameters
    ----------
    fig:
        The matplotlib Figure to export.
    base_name:
        Base filename without extension, e.g. "r56_LC_A_concentration".
    dpi:
        PNG resolution.
    bbox_inches:
        Bounding box clipping for export.
    extra_dirs:
        Additional directory path components prepended to the output filename.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved PNG and SVG files.
    """
    out_dir = Path(__file__).resolve().parent.parent.joinpath("figs")
    if extra_dirs:
        out_dir = out_dir.joinpath(str(extra_dirs))
    out_dir.mkdir(exist_ok=True, parents=True)

    png_path = out_dir.joinpath(f"{base_name}.png")
    svg_path = out_dir.joinpath(f"{base_name}.svg")

    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(svg_path, bbox_inches=bbox_inches)

    return png_path, svg_path


# ---------------------------------------------------------------------
# Twin-y annotation helpers
# ---------------------------------------------------------------------

def add_twin_ylabel(ax_twin: plt.Axes, value: float, text: str,
                    color: str = "red", weight: str = "bold", size: int = 11) -> None:
    """
    Add a labelled value on a twin-y axis with annotation.

    Parameters
    ----------
    ax_twin:
        The twin-y Axes object.
    value:
        Y-axis value at which to place the annotation.
    text:
        Annotation text.
    color:
        Text colour.
    weight:
        Font weight.
    size:
        Font size in points.
    """
    ax_twin.text(0, value, text, va="center", color=color, weight=weight, size=size)


def setup_twin_axes(fig: plt.Figure, ax_list: Sequence[plt.Axes],
                    twin_labels: Sequence[str], twin_colors: Sequence[str] | None = None,
                    label_x_offset: float = 1.08) -> list[plt.Axes]:
    """
    Create twin-y axes and position their labels.

    Parameters
    ----------
    fig:
        The parent Figure.
    ax_list:
        Primary axes (the last one typically shares y with its twin).
    twin_labels:
        Labels for each twin-y axis.
    twin_colors:
        Colours for twin axis tick labels.
    label_x_offset:
        Horizontal offset for y-axis label positioning.

    Returns
    -------
    list[plt.Axes]
        The twin axes.
    """
    if twin_colors is None:
        twin_colors = ["red"] * len(twin_labels)

    twin_axes = []
    for i, (ax, label, color) in enumerate(zip(ax_list, twin_labels, twin_colors)):
        if i < len(ax_list) - 1:  # Skip the last ax if it shares y
            twin = ax.twinx()
            twin.yaxis.set_label_coords(label_x_offset, 0.5)
            twin.tick_params(axis="y", colors=color)
            twin_labels[i]  # Reference to use label
            twin_axes.append(twin)

    return twin_axes


# ---------------------------------------------------------------------
# Convenience: stacked subplot with twin-y for concentration estimation
# ---------------------------------------------------------------------

def create_concentration_estimation_plot(
    time_s: np.ndarray,
    plif: np.ndarray,
    estimations: dict[str, tuple[np.ndarray, str]],
    window: tuple[float, float],
    figsize: tuple[int, int] = (8, 4),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create a multi-panel concentration estimation figure.

    Parameters
    ----------
    time_s:
        Full time array.
    plif:
        Ground truth PLIF array.
    estimations:
        Dictionary mapping method names to (estimation_array, color) tuples.
    window:
        Time window (t_min, t_max) for display.
    figsize:
        Figure size in inches.

    Returns
    -------
    tuple[plt.Figure, list[plt.Axes]]
        The Figure object and list of primary axes.
    """
    # Slice to window
    mask = (time_s <= window[1]) & (time_s > window[0])
    t_win = time_s[mask]
    plif_win = plif[mask]

    n_methods = len(estimations)
    fig, axes = plt.subplots(nrows=n_methods, figsize=figsize, sharex=True)

    if n_methods == 1:
        axes = [axes]

    ycolor = "red"

    for i, (method_name, (est, color)) in enumerate(estimations.items()):
        est_win = est[mask]

        axes[i].plot(t_win, plif_win, color="black", label="Ground Truth")
        twin = axes[i].twinx()
        twin.plot(t_win, est_win, color=color, label="Estimation")

        if i == 0:
            axes[i].text(window[0] + 0.05, plif_win.max() * 0.9, method_name,
                         color="black", weight="bold", size=11)
            twin.text(window[0] + 0.05, est_win.max() * 0.9, "e-nose " + method_name,
                      color=color, weight="bold", size=11)

        axes[i].spines[["right"]].set_visible(False)
        twin.spines[["top"]].set_visible(False)
        twin.tick_params(axis="y", colors=color)

        axes[i].grid(which="major", alpha=0.6, linestyle="--")
        axes[i].minorticks_on()

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_ylabel("PLIF concentration (rel.)")

    fig.tight_layout()

    return fig, axes