#%%
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.combined_io import load_combined_csv
from utils.plotting import configure_style, style_axes, export_figure


# Keep TeX disabled so SVG export stays vector-friendly.
configure_style(usetex=False, fonttype="none")


def plot_enose_sample(exp_id: int = 56, t_start: float = 39.2, window_s: float = 10.0) -> None:
    """Plot a short window of e-nose signals for the given experiment."""
    data = load_combined_csv(exp_id)
    data_window = data[(data["time_s"] >= t_start) & (data["time_s"] <= t_start + window_s)]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(4.5, 3.5))
    style_axes(ax)

    t_rel = data_window["time_s"] - t_start
    ax[0].plot(t_rel, data_window["R_gas_1"] / 1000, "#fc9272", zorder=1, alpha=0.9)
    ax[0].plot(t_rel, data_window["R_gas_5"] / 1000, "#de2d26", zorder=1, alpha=0.9)
    ax[1].plot(t_rel, data_window["R_gas_4"] / 1000, "#bcbddc", zorder=1, alpha=0.9)
    ax[1].plot(t_rel, data_window["R_gas_8"] / 1000, "#756bb1", zorder=1, alpha=0.9)

    # Annotate key traces for quick visual ID.
    ax[0].text(3.3, 25, "Sensor 1", va="center", color="#fc9272", weight="bold", size=9)
    ax[0].text(3.35, 20, "Sensor 5", va="center", color="#de2d26", weight="bold", size=9)
    ax[1].text(3.3, 211, "Sensor 4", va="center", color="#bcbddc", weight="bold", size=9)
    ax[1].text(3.6, 195, "Sensor 8", va="center", color="#756bb1", weight="bold", size=9)

    fig.text(0.0, 0.5, "Resistance (kΩ)", va="center", rotation="vertical")
    ax[1].set_xlabel("Time (s)")

    export_figure(
        fig,
        base_name=f"r{exp_id}_enose_example_data",
        dpi=600,
        bbox_inches="tight",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_enose_sample()
