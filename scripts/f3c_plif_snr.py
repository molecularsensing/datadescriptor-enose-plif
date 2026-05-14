#%%
"""Plot PLIF signal-to-noise ratio across zones for each experiment."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plotting import export_figure, configure_style
from utils.constants import LOCATIONS, PLIF_USEFUL_BOUNDS
from utils.combined_io import load_combined_csv

COLORS = {"A": "#fc9272", "B": "#de2d26", "C": "#bcbddc", "D": "#756bb1"}


def _load_enose_plif(exp_id: str):
    """Load combined e-nose/PLIF data for an experiment id."""
    return load_combined_csv(f"r{exp_id}")


def _plot_snr(exp_id: str, loc_key: str, df) -> None:
    """Plot PLIF SNR for all zones on one axis."""
    fig, ax = plt.subplots(nrows=1, figsize=(6, 2), sharex=True)
    for zone in PLIF_USEFUL_BOUNDS[loc_key].keys():
        color = COLORS[zone]
        ax.plot(df["time_s"], df[f"plif_{zone}"] / (3 * 10 ** (-4)), c=color, label=f"Zone {zone}")
    ax.set_xlim([196, 210])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Signal-to-noise ratio\n[...]")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper left", ncols=4)
    ax.grid(True, axis="x")

    export_figure(fig, base_name=f"r{exp_id}_{loc_key}_plif_SNR", bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_plif_snr(exp_ids: list[str] | None = None) -> None:
    """Plot SNR traces for each experiment."""
    exp_ids = exp_ids or ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]
    for exp_id in exp_ids:
        print(f"Experiment r{exp_id}")
        df = _load_enose_plif(exp_id)
        loc_key = LOCATIONS[f"r{exp_id}"]
        _plot_snr(exp_id, loc_key, df)


if __name__ == "__main__":
    plot_plif_snr()
# %%
