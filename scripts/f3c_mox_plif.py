#%%
"""Legacy script: Compare PLIF/e-nose traces, temperatures, and saturation."""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({"text.usetex": False, "svg.fonttype": "none"})

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.constants import ENOSE_ZONE_SENSOR, LOCATIONS, PLIF_USEFUL_BOUNDS
from utils.combined_io import load_combined_csv
from utils.plotting import export_figure

COLORS = {"A": "#fc9272", "B": "#de2d26", "C": "#bcbddc", "D": "#756bb1"}
SENSOR_LABELS = {"A": "Sensor 1", "B": "Sensor 5", "C": "Sensor 4", "D": "Sensor 8"}


def _plot_plif_enose_pair(fig, ax, exp_id: str, loc_key: str, df, ab_cd: str) -> None:
    """Plot PLIF vs e-nose conductance for AB or CD groups."""
    for zone in PLIF_USEFUL_BOUNDS[loc_key].keys():
        if ab_cd == "AB" and zone not in ("A", "B"):
            continue
        if ab_cd == "CD" and zone not in ("C", "D"):
            continue

        color = COLORS[zone]
        col = ENOSE_ZONE_SENSOR[zone]
        ax[0].plot(df["time_s"], df[f"plif_{zone}"], c=color, label=f"Zone {zone}")
        ax[1].plot(df["time_s"], 1e6 * 1 / df[col], c=color, label=SENSOR_LABELS[zone])
        ax[0].set_ylim([0, 0.07])
        ax[1].set_xlim([196, 210])
        ax[0].grid(axis="x")
        ax[1].grid(axis="x")
        ax[0].spines[["top", "right"]].set_visible(False)
        ax[1].spines[["top", "right"]].set_visible(False)
        ax[1].set_ylabel("Conductance\n" + r"($\mu$S)")
        ax[0].set_ylabel("Concentration\n(rel.)")
        ax[1].set_xlabel("Time (s)")
        ax[0].legend(loc="upper left")
        ax[1].legend(loc="upper left")

    export_figure(fig, base_name=f"r{exp_id}_{loc_key}_{ab_cd}_plif_enose", bbox_inches="tight")


def _plot_temperatures_pair(fig, ax, exp_id: str, loc_key: str, df) -> None:
    """Plot heater temperatures for B and D zones."""
    for i, zone in enumerate(["B", "D"]):
        mean, std = np.mean(df[f"T_heat_{i+1}"]), np.std(df[f"T_heat_{i+1}"])
        ax[i].plot(df["time_s"], df[f"T_heat_{i+1}"], c=COLORS[zone], label=SENSOR_LABELS[zone])
        ax[i].axhline(mean, color="k")
        ax[i].axhline(mean + std, color="k", linestyle="--")
        ax[i].axhline(mean - std, color="k", linestyle="--")
        print(i + 1, np.mean(df[f"T_heat_{i+1}"]), np.std(df[f"T_heat_{i+1}"]))
        ax[i].legend(loc="upper left")
        ax[i].spines[["top", "right"]].set_visible(False)
        ax[i].set_ylabel("Temperature (°C)")
    ax[i].set_xlim(0, 275)
    ax[1].set_xlabel("Time (s)")
    export_figure(fig, base_name=f"r{exp_id}_{loc_key}_temperature", bbox_inches="tight")


# Iterate over experiments
exp_ids = ["56", "57", "58", "59", "60", "61", "70"]
for exp_id in exp_ids:
    print(f"Experiment r{exp_id}")

    # Load data using helper
    enose_plif_df = load_combined_csv(f"r{exp_id}")
    LOC = LOCATIONS[f"r{exp_id}"]

    # PLIF vs e-nose AB
    fig1, ax1 = plt.subplots(nrows=2, sharex=True, figsize=(6, 4))
    _plot_plif_enose_pair(fig1, ax1, exp_id, LOC, enose_plif_df, "AB")
    plt.tight_layout()
    plt.show()

    # PLIF vs e-nose CD
    fig2, ax2 = plt.subplots(nrows=2, sharex=True, figsize=(6, 4))
    _plot_plif_enose_pair(fig2, ax2, exp_id, LOC, enose_plif_df, "CD")
    plt.tight_layout()
    plt.show()

    # Temperatures
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(5.5, 4.5))
    _plot_temperatures_pair(fig, ax, exp_id, LOC, enose_plif_df)
    plt.tight_layout()
    plt.show()

# %%
# Check for saturation
exp_ids = ["56", "57", "58", "59", "60", "61", "70"]
saturation = {
    exp_id: {f"R_gas_{i}": False for i in range(1, 9)}
    for exp_id in exp_ids
}

# %%
for exp_id in exp_ids:
    print(exp_id)
    enose_plif_df = load_combined_csv(f"r{exp_id}")
    len_total = enose_plif_df.shape[0]
    pd.set_option("display.max_columns", None)
    print(enose_plif_df.head())
    break

    LOC = LOCATIONS[f"r{exp_id}"]

    for i in range(1, 9):
        saturation[exp_id][f"R_gas_{i}"] = np.round(
            100 - 100 * enose_plif_df[enose_plif_df[f"R_gas_{i}"] == 0].shape[0] / len_total, 1
        )

print(saturation)
# %%
df_saturation = pd.DataFrame.from_dict(saturation, orient="index")
df_saturation = df_saturation.rename(
    columns={f"R_gas_{i}": f"Sensor {i}" for i in range(1, 9)}
)
df_percent = df_saturation.applymap(lambda x: f"{x}%")
df_percent.to_csv("../figs/saturation.csv", index=True)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(df_saturation, cmap="viridis")
plt.show()

print(df_percent)

# %%