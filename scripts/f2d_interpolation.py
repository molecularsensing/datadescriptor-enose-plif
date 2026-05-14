#%%
"""Plot interpolation comparison between PLIF frames and interpolated traces."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plotting import configure_style, style_axes, export_figure
from utils.constants import LOCATIONS, PLIF_USEFUL_BOUNDS
from utils.combined_io import load_combined_csv


def plot_interpolation(exp_ids: list[str] | None = None, window: tuple[float, float] = (200, 203)) -> None:
    """Plot measured vs interpolated PLIF values for each zone across experiments."""
    exp_ids = exp_ids or ["56"]

    for exp_id in exp_ids:
        enose_plif_df = load_combined_csv(f"r{exp_id}", sampling="interpolated")
        loc_key = LOCATIONS[f"r{exp_id}"]
        measured_df = enose_plif_df[enose_plif_df["plif_image"]]

        for zone in PLIF_USEFUL_BOUNDS[loc_key].keys():
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.scatter(
                measured_df["time_s"],
                measured_df[f"plif_{zone}"],
                c="r",
                s=20,
                label="Measured (20Hz)",
                zorder=10,
            )
            ax.scatter(
                enose_plif_df["time_s"],
                enose_plif_df[f"plif_{zone}"],
                c="k",
                s=5,
                label="Interpolated (1kHz)",
                zorder=1,
            )
            ax.set_xlim(window)
            style_axes(ax, left=True, bottom=True)
            ax.grid()
            ax.set_ylabel("Concentration (norm.)")
            ax.set_xlabel("Time (s)")
            ax.legend(loc="upper left")

            export_figure(fig, base_name=f"r{exp_id}_{loc_key}_interpolation_{zone}", dpi=600, bbox_inches="tight")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    plot_interpolation()

# %%
