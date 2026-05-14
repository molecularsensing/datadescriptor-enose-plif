#%%
import sys
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plif_io import load_plif_frame, load_px_per_mm
from utils.plotting import configure_style, export_figure
from utils.constants import ENOSE_BOUNDS, PLIF_USEFUL_BOUNDS


def _load_px_mm(loc_key: str) -> float:
    """Load pixels-to-mm calibration factor for a location key."""
    px_mm = load_px_per_mm(loc_key)
    if px_mm is None:
        raise ValueError(f"Could not load px/mm calibration for location: {loc_key}")
    return px_mm
    

def _add_reference_overlays(ax: plt.Axes, loc_key: str) -> None:
    """Overlay e-nose footprint and sensor zones for a location key."""
    ymin, ymax, xmin, xmax = ENOSE_BOUNDS[loc_key]
    width, height = xmax - xmin, ymax - ymin
    rect = Rectangle(
        (xmin - 0.5, ymin - 0.5),
        width,
        height,
        linewidth=1,
        edgecolor="lightgreen",
        facecolor="lightgreen",
        alpha=0.3,
    )
    ax.add_patch(rect)
    ax.text(xmin + 165, ymax + 45, loc_key[1:], fontsize=10, fontweight="bold", color="lightgreen")

    for zone, (z_ymin, z_ymax, z_xmin, z_xmax) in PLIF_USEFUL_BOUNDS[loc_key].items():
        z_width, z_height = z_xmax - z_xmin, z_ymax - z_ymin
        rect = Rectangle(
            (z_xmin - 0.5, z_ymin - 0.5),
            z_width,
            z_height,
            linewidth=0,
            edgecolor=None,
            facecolor="cyan",
            label=f"MOX {zone.upper()}",
        )
        ax.add_patch(rect)


def _set_mm_ticks(ax: plt.Axes, data: np.ndarray, px_mm: float) -> None:
    """Apply millimeter ticks using calibration constants."""
    y_px, x_px = data.shape

    x_pixel_per_cm = y_pixel_per_cm = px_mm * 10
    x_cm = x_px / x_pixel_per_cm
    y_cm = y_px / x_pixel_per_cm

    xticks_cm = -1 * np.arange(0, 31, 10) + x_cm
    xticklabels = np.arange(0, 301, 100)
    yticks_cm = -1 * np.arange(-30 / 2, 30 / 2 + 0.1, 5) + y_cm / 2
    yticklabels = np.arange(-150, 151, 50)

    xticks_px = xticks_cm * x_pixel_per_cm
    yticks_px = yticks_cm * y_pixel_per_cm

    ax.set_xticks(xticks_px)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("X position [mm]")
    ax.set_yticks(yticks_px)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Y position [mm]")

def plot_heatmap(exp_id: str = "r56", frame_idx: int = 1600, save: bool = True) -> None:
    """Plot a single PLIF frame with e-nose overlays."""
    frame = load_plif_frame(exp_id, frame_idx)

    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="black")

    masked = np.ma.masked_less_equal(frame, 0)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=colors.LogNorm(vmin=0.005, vmax=1),
        origin="lower",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Concentration (rel.)")
    cbar.set_ticks([0.01, 0.1, 1.0])
    cbar.set_ticklabels([0.01, 0.1, 1.0])

    for loc_key in ["LC", "LU", "LR"]:
        _add_reference_overlays(ax, loc_key)

    px_mm = _load_px_mm("LC")
    _set_mm_ticks(ax, frame, px_mm)

    # Rotate the entire frame to match the physical orientation.
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    if save:
        export_figure(fig, base_name="heatmap", dpi=600, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_heatmap(save=False)

# %%
