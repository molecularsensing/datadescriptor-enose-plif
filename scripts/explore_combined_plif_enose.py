#%%
"""
Explore combined e-nose/PLIF CSV data.

This script demonstrates access to the combined, temporally aligned e-nose and
PLIF traces stored as CSV files. It supports three common sampling strategies:

1. interpolated:
   Use the combined CSV directly, typically containing interpolated PLIF traces
   on the e-nose time base.

2. plif_native:
   Select only rows corresponding to original PLIF image acquisition times,
   typically around 20 Hz.

3. common_rate:
   Resample the combined dataset to a user-defined common sampling frequency.
   This allows, for example, upsampling native PLIF to 100 Hz while downsampling
   high-rate e-nose/MOx data to 100 Hz. For downsampling, a centred zero-lag
   smoother is applied before interpolation onto the target time base.

The goal is to provide lightweight examples for inspecting, plotting, and loading
the combined dataset without imposing a single preferred analysis workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plotting import configure_style, export_figure
from utils.constants import LOCATIONS, PLIF_USEFUL_BOUNDS
from utils.combined_io import (
    load_combined_csv,
    get_plif_columns,
    get_enose_columns,
    infer_native_rate_hz,
    select_plif_native_rows,
    make_target_timebase,
    zero_lag_smooth_for_downsampling,
    interpolate_to_timebase,
    resample_combined_to_common_rate,
    load_combined_dataset,
    inspect_combined_csv,
    print_combined_info,
)


SamplingStrategy = Literal["interpolated", "plif_native", "common_rate"]


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_sampling_strategies(
    exp_id: str | int = "56",
    zone: str | None = None,
    enose_col: str | None = None,
    window: tuple[float, float] = (200.0, 203.0),
    target_rate_hz: float = 100.0,
    save: bool = False,
) -> None:
    """
    Compare native PLIF, interpolated PLIF, and common-rate PLIF traces.

    Optionally overlay one e-nose channel on a secondary y-axis.
    """
    exp_id_norm = str(exp_id).lower().strip()
    exp_id_norm = exp_id_norm[1:] if exp_id_norm.startswith("r") else exp_id_norm
    loc_key = LOCATIONS[f"r{exp_id_norm}"]

    raw = load_combined_dataset(exp_id_norm, strategy="interpolated")
    native = load_combined_dataset(exp_id_norm, strategy="plif_native")
    common = load_combined_dataset(
        exp_id_norm,
        strategy="common_rate",
        target_rate_hz=target_rate_hz,
        plif_source="native",
    )

    if zone is None:
        zone = next(iter(PLIF_USEFUL_BOUNDS[loc_key].keys()))

    plif_col = f"plif_{zone}"

    if plif_col not in raw.columns:
        raise KeyError(f"Could not find PLIF column: {plif_col}")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.scatter(
        native["time_s"],
        native[plif_col],
        s=25,
        label="Native PLIF frames, ~20 Hz",
        zorder=10,
    )

    ax.plot(
        raw["time_s"],
        raw[plif_col],
        linewidth=1.0,
        label="Interpolated PLIF on full CSV time base",
        zorder=3,
    )

    ax.plot(
        common["time_s"],
        common[plif_col],
        linewidth=1.5,
        label=f"PLIF resampled to {target_rate_hz:g} Hz",
        zorder=5,
    )

    ax.set_xlim(window)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("PLIF concentration [norm.]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.4)

    if enose_col is not None:
        if enose_col not in raw.columns:
            raise KeyError(f"Could not find e-nose column: {enose_col}")

        ax2 = ax.twinx()
        ax2.plot(
            common["time_s"],
            common[enose_col],
            linewidth=1.0,
            linestyle="--",
            label=f"{enose_col}, {target_rate_hz:g} Hz",
        )
        ax2.set_ylabel(enose_col)
        ax2.spines["top"].set_visible(False)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    else:
        ax.legend(loc="upper left")

    ax.set_title(f"r{exp_id_norm}, {loc_key}, {plif_col}")

    plt.tight_layout()

    if save:
        export_figure(
            fig,
            base_name=f"r{exp_id_norm}_{loc_key}_sampling_strategies_{plif_col}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()


def plot_common_rate_effect(
    exp_id: str | int = "56",
    zone: str | None = None,
    target_rates_hz: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0),
    window: tuple[float, float] = (200.0, 203.0),
    save: bool = False,
) -> None:
    """
    Compare several user-defined common sampling rates for the same PLIF trace.
    """
    exp_id_norm = str(exp_id).lower().strip()
    exp_id_norm = exp_id_norm[1:] if exp_id_norm.startswith("r") else exp_id_norm
    loc_key = LOCATIONS[f"r{exp_id_norm}"]

    raw = load_combined_dataset(exp_id_norm, strategy="interpolated")
    native = load_combined_dataset(exp_id_norm, strategy="plif_native")

    if zone is None:
        zone = next(iter(PLIF_USEFUL_BOUNDS[loc_key].keys()))

    plif_col = f"plif_{zone}"

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.scatter(
        native["time_s"],
        native[plif_col],
        s=25,
        label="Native PLIF frames",
        zorder=10,
    )

    ax.plot(
        raw["time_s"],
        raw[plif_col],
        linewidth=1.0,
        alpha=0.6,
        label="Interpolated full CSV",
    )

    for rate in target_rates_hz:
        common = load_combined_dataset(
            exp_id_norm,
            strategy="common_rate",
            target_rate_hz=rate,
            plif_source="native",
        )

        ax.plot(
            common["time_s"],
            common[plif_col],
            linewidth=1.2,
            label=f"Common rate {rate:g} Hz",
        )

    ax.set_xlim(window)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("PLIF concentration [norm.]")
    ax.set_title(f"r{exp_id_norm}, {loc_key}, common-rate comparison")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save:
        export_figure(
            fig,
            base_name=f"r{exp_id_norm}_{loc_key}_common_rate_effect_{plif_col}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------

def export_resampled_combined_csv(
    exp_id: str | int,
    out_path: Path,
    target_rate_hz: float = 100.0,
    plif_source: Literal["native", "interpolated"] = "native",
    smoothing_window_s: float | None = None,
) -> Path:
    """
    Export a common-rate version of the combined dataset to CSV.
    """
    df = load_combined_dataset(
        exp_id,
        strategy="common_rate",
        target_rate_hz=target_rate_hz,
        plif_source=plif_source,
        smoothing_window_s=smoothing_window_s,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix != ".csv":
        out_path = out_path.with_suffix(".csv")

    df.to_csv(out_path, index=False)

    print(f"Exported resampled combined CSV: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Target rate: {target_rate_hz:g} Hz")
    print(f"PLIF source: {plif_source}")

    return out_path


# ---------------------------------------------------------------------
# Example workflow
# ---------------------------------------------------------------------

def example_workflow(exp_id: str = "56", export: bool = False) -> None:
    """
    Example use of the combined-dataset loader and plotting functions.
    """
    # 1. Inspect the CSV schema and sampling rates.
    inspect_combined_csv(exp_id)

    # 2. Load the three main sampling representations.
    full_1khz = load_combined_dataset(exp_id, strategy="interpolated")
    native_plif = load_combined_dataset(exp_id, strategy="plif_native")
    common_100hz = load_combined_dataset(
        exp_id,
        strategy="common_rate",
        target_rate_hz=100.0,
        plif_source="native",
    )

    print("Loaded representations")
    print("-" * 60)
    print(f"Interpolated/full CSV: {full_1khz.shape}")
    print(f"Native PLIF rows:      {native_plif.shape}")
    print(f"Common 100 Hz:         {common_100hz.shape}")
    print()

    # 3. Visualise one zone.
    plot_sampling_strategies(
        exp_id=exp_id,
        zone=None,
        window=(200.0, 203.0),
        target_rate_hz=100.0,
        save=False,
    )

    # 4. Compare several common-rate choices.
    plot_common_rate_effect(
        exp_id=exp_id,
        zone=None,
        target_rates_hz=(20.0, 50.0, 100.0, 200.0),
        window=(200.0, 203.0),
        save=False,
    )

    # 5. Optional export.
    if export:
        out_dir = Path(__file__).resolve().parent.parent.joinpath("processed_examples")
        export_resampled_combined_csv(
            exp_id,
            out_path=out_dir.joinpath("enose_plif_r56_common_100hz.csv"),
            target_rate_hz=100.0,
            plif_source="native",
        )


if __name__ == "__main__":
    exp_id = "56"
    example_workflow(exp_id=exp_id, export=False)

# %%