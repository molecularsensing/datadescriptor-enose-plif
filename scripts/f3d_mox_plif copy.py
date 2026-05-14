#%%
"""Compare PLIF and e-nose traces, plus saturation summary across experiments."""
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({"text.usetex": False, "svg.fonttype": "none"})

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.constants import ENOSE_ZONE_SENSOR, LOCATIONS, PLIF_USEFUL_BOUNDS, SSD_COMBINED_FOLDER, SPEEDS

COLORS = {"A": "#fc9272", "B": "#de2d26", "C": "#bcbddc", "D": "#756bb1"}
SENSOR_LABELS = {"A": "Sensor 1", "B": "Sensor 5", "C": "Sensor 4", "D": "Sensor 8"}
DEFAULT_EXPERIMENTS = ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]


def _load_enose_plif(exp_id: str) -> pd.DataFrame:
    """Load combined e-nose/PLIF data for an experiment id."""
    df = pd.read_csv(SSD_COMBINED_FOLDER.joinpath(f"enose_plif_r{exp_id}.csv"))
    df["timestamp"] = pd.to_timedelta(df["timestamp"])
    return df

def _plot_plif_enose(exp_id: str, loc_key: str, df: pd.DataFrame) -> None:
    """Plot PLIF vs e-nose conductance for AB and CD groups."""
    fig_ab, ax_ab = plt.subplots(nrows=2, sharex=True, figsize=(6, 4))
    fig_cd, ax_cd = plt.subplots(nrows=2, sharex=True, figsize=(6, 4))
    for zone in PLIF_USEFUL_BOUNDS[loc_key].keys():
        color = COLORS[zone]
        sensor_col = ENOSE_ZONE_SENSOR[zone]
        target_axes = ax_ab if zone in ["A", "B"] else ax_cd
        target_axes[0].plot(df["time_s"], df[f"plif_{zone}"], c=color, label=f"Zone {zone}")
        target_axes[1].plot(df["time_s"], 1e6 * 1 / df[sensor_col], c=color, label=SENSOR_LABELS[zone])
        target_axes[0].set_ylim([0, 0.07])
        target_axes[1].set_xlim([196, 210])
        for axes in target_axes:
            axes.grid(axis="x")
            axes.spines[["top", "right"]].set_visible(False)
        target_axes[1].set_ylabel("Conductance\n" + r"($\mu$S)")
        target_axes[0].set_ylabel("Concentration\n(rel.)")
        target_axes[1].set_xlabel("Time (s)")
        target_axes[0].legend(loc="upper left")
        target_axes[1].legend(loc="upper left")

    out_dir = Path(__file__).resolve().parent.parent.joinpath("figs")
    out_dir.mkdir(exist_ok=True)
    fig_ab.tight_layout()
    fig_cd.tight_layout()
    fig_ab.savefig(out_dir.joinpath(f"r{exp_id}_{loc_key}_AB_plif_enose.svg"), bbox_inches="tight")
    fig_cd.savefig(out_dir.joinpath(f"r{exp_id}_{loc_key}_CD_plif_enose.svg"), bbox_inches="tight")
    plt.show()

def plot_plif_enose(exp_ids: list[str] | None = None) -> None:
    """Generate PLIF/e-nose comparisons and saturation table."""
    exp_ids = exp_ids or DEFAULT_EXPERIMENTS
    out_dir = Path(__file__).resolve().parent.parent.joinpath("figs")
    out_dir.mkdir(exist_ok=True)

    for exp_id in exp_ids:
        print(f"Experiment r{exp_id}")
        df = _load_enose_plif(exp_id)
        loc_key = LOCATIONS[f"r{exp_id}"]
        _plot_plif_enose(exp_id, loc_key, df)


if __name__ == "__main__":
    plot_plif_enose()
