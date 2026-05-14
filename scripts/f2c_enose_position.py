#%%
"""Plot e-nose positions and sensor zones on PLIF flatfield images."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plif_io import load_px_per_mm, resolve_exp_id, get_plif_h5_path
from utils.plotting import configure_style, export_figure
from utils.constants import PLIF_USEFUL_BOUNDS

DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data", "flatfield")


def _load_plif_flatfield(loc_key: str) -> np.ndarray:
    """Load flatfield PLIF frame from HDF5 (laser4 dataset)."""
    import h5py

    h5_path = get_plif_h5_path(loc_key)
    with h5py.File(h5_path, "r") as h5f:
        return h5f["DataSets"]["flatfield_laser4"][:, :]


def _load_flatfield(loc_key: str) -> np.ndarray:
    """Load flatfield PLIF data for a location key."""
    from utils.constants import PLIF_REFS
    path = DATA_DIR.joinpath(PLIF_REFS[loc_key])
    return np.load(path)


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


def _plot_full_frame(loc_key: str, data: np.ndarray, vmin: int = 350, vmax: int = 540) -> None:
    """Plot full flatfield with a dashed bounding box around useful sensor zones."""
    ymin, ymax, xmin, xmax = (
        PLIF_USEFUL_BOUNDS[loc_key]["B"][0] - 5,
        PLIF_USEFUL_BOUNDS[loc_key]["A"][1] + 5,
        PLIF_USEFUL_BOUNDS[loc_key]["A"][2] - 5,
        PLIF_USEFUL_BOUNDS[loc_key]["C"][3] + 5,
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
    cbar = plt.colorbar(im)
    cbar.set_label("Raw pixel intensity")

    rect = Rectangle(
        (xmin - 0.5, ymin - 0.5),
        xmax - xmin,
        ymax - ymin,
        linewidth=1,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

    px_mm = load_px_per_mm(loc_key)
    _set_mm_ticks(ax, data, px_mm)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    export_figure(fig, base_name=f"enose_position_{loc_key}", dpi=600, bbox_inches="tight")
    plt.show()


def _plot_inset(loc_key: str, data: np.ndarray, vmin: int = 350, vmax: int = 540) -> None:
    """Plot inset showing sensor zones for a location key."""
    ymin, ymax, xmin, xmax = (
        PLIF_USEFUL_BOUNDS[loc_key]["B"][0] - 5,
        PLIF_USEFUL_BOUNDS[loc_key]["A"][1] + 5,
        PLIF_USEFUL_BOUNDS[loc_key]["A"][2] - 5,
        PLIF_USEFUL_BOUNDS[loc_key]["C"][3] + 5,
    )
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    for zone, (z_ymin, z_ymax, z_xmin, z_xmax) in PLIF_USEFUL_BOUNDS[loc_key].items():
        rect = Rectangle(
            (z_xmin - 0.5, z_ymin - 0.5),
            z_xmax - z_xmin,
            z_ymax - z_ymin,
            linewidth=1.5,
            edgecolor="white",
            facecolor="none",
            linestyle="--",
            label=f"MOX {zone.upper()}",
        )
        ax.add_patch(rect)

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    export_figure(fig, base_name=f"enose_position_{loc_key}_inset", dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_enose_positions() -> None:
    """Generate full-frame and inset plots for each location key."""
    for loc_key in PLIF_USEFUL_BOUNDS.keys():
        print("Location", loc_key)
        data = _load_plif_flatfield(loc_key)
        _plot_full_frame(loc_key, data)
        _plot_inset(loc_key, data)


if __name__ == "__main__":
    plot_enose_positions()

# %%