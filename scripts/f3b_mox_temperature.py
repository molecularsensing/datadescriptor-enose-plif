#%%
"""Plot heater temperature traces for selected experiments."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plotting import configure_style, export_figure
from utils.constants import LOCATIONS
from utils.combined_io import load_combined_csv

COLORS = {"A": "#fc9272", "B": "#de2d26", "C": "#bcbddc", "D": "#756bb1"}
SENSOR_LABELS = {"A": "Sensor 1", "B": "Sensor 5", "C": "Sensor 4", "D": "Sensor 8"}


def _load_enose_plif(exp_id: str) -> pd.DataFrame:
    """Load combined e-nose/PLIF data for an experiment id."""
    return load_combined_csv(f"r{exp_id}", sampling="native")


def _plot_temperatures(exp_id: str, loc_key: str, df: pd.DataFrame) -> None:
    """Plot heater temperatures for B and D zones with mean and ±std bands."""
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(5.5, 4.5))
    for i, zone in enumerate(["B", "D"]):
        mean, std = float(np.mean(df[f"T_heat_{i+1}"])), float(np.std(df[f"T_heat_{i+1}"]))
        ax[i].plot(df["time_s"], df[f"T_heat_{i+1}"], c=COLORS[zone], label=SENSOR_LABELS[zone])
        ax[i].axhline(mean, color="k")
        ax[i].axhline(mean + std, color="k", linestyle="--")
        ax[i].axhline(mean - std, color="k", linestyle="--")
        ax[i].legend(loc="upper left")
        ax[i].spines[["top", "right"]].set_visible(False)
        ax[i].set_ylabel("Temperature (°C)")
    ax[i].set_xlim(0, 275)
    ax[1].set_xlabel("Time (s)")

    for idx in range(8):
        print(f"Sensor {idx + 1}, Temperature mean & std:", np.mean(df[f'T_heat_{idx+1}']), np.std(df[f'T_heat_{idx+1}']))

    export_figure(fig, base_name=f"r{exp_id}_{loc_key}_temperature", bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_temperatures(exp_ids: list[str] | None = None) -> None:
    """Plot heater temperature stability for a list of experiments."""
    exp_ids = exp_ids or ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]
    for exp_id in exp_ids:
        print(f"Experiment r{exp_id}")
        df = _load_enose_plif(exp_id)
        loc_key = LOCATIONS[f"r{exp_id}"]
        _plot_temperatures(exp_id, loc_key, df)


if __name__ == "__main__":
    plot_temperatures()