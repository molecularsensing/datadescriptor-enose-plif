"""
Concentration estimation workflow helpers.

This module provides reusable functions for the combined e-nose/PLIF concentration
estimation pipeline. It centralises common patterns shared across scripts and
notebooks:

- extracting zone-specific signals from combined data;
- running the full estimation pipeline (alignment, derivative, deconvolution);
- computing metrics and plotting results;
- processing single zones or entire experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import constants from utils.constants
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.constants import ENOSE_ZONE_SENSOR, LOCATIONS, PLIF_USEFUL_BOUNDS
from utils.misc import metrics


def prepare_zone_signals(
    df: pd.DataFrame,
    zone: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Extract time, PLIF concentration, and e-nose conductance for one PLIF zone.

    The sensor mapping is taken from ENOSE_ZONE_SENSOR.

    Parameters
    ----------
    df:
        Combined e-nose/PLIF DataFrame.
    zone:
        Zone identifier (e.g., 'A', 'B', 'C', 'D').

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, str]
        time_s, plif, conductance, sensor_col.
    """
    plif_col = f"plif_{zone}"
    if plif_col not in df.columns:
        raise KeyError(f"Missing PLIF column: {plif_col}")

    if zone not in ENOSE_ZONE_SENSOR:
        raise KeyError(f"No e-nose sensor mapping defined for zone: {zone}")

    sensor_col = ENOSE_ZONE_SENSOR[zone]
    if sensor_col not in df.columns:
        raise KeyError(f"Missing e-nose sensor column: {sensor_col}")

    time_s = df["time_s"].to_numpy(dtype=float)
    plif = df[plif_col].to_numpy(dtype=float)

    # Conductance is the inverse of resistance-like sensor signal.
    conductance = 1.0 / df[sensor_col].to_numpy(dtype=float)

    return time_s, plif, conductance, sensor_col


def default_derivative_transform(conductance: np.ndarray, dt: float) -> np.ndarray:
    """
    Differentiate and smooth conductance using Savitzky-Golay filter.

    The window length is adjusted to be valid for the signal length.

    Parameters
    ----------
    conductance:
        Conductance signal array.
    dt:
        Time step.

    Returns
    -------
    np.ndarray
        Smoothed derivative of conductance.
    """
    from scipy.signal import savgol_filter

    conductance = np.asarray(conductance, dtype=float)

    if len(conductance) < 7:
        return np.gradient(conductance, dt)

    window_length = min(101, len(conductance) - 1)
    if window_length % 2 == 0:
        window_length -= 1

    polyorder = min(5, window_length - 2)
    derivative = np.gradient(conductance, dt)

    return savgol_filter(
        derivative,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp",
    )


def align_signals(
    aligner: object,
    time_s: np.ndarray,
    conductance: np.ndarray,
    plif: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align conductance and PLIF using an object with a fit_transform method.

    Parameters
    ----------
    aligner:
        Object providing a fit_transform(time, signal, target) method.
    time_s:
        Time array.
    conductance:
        Conductance signal.
    plif:
        PLIF concentration signal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Aligned time, conductance, and PLIF arrays.
    """
    if not hasattr(aligner, "fit_transform"):
        raise TypeError("aligner must provide a fit_transform(time, signal, target) method.")

    return aligner.fit_transform(time_s, conductance, plif)


def run_deconvolution(
    model: "SupervisedDeconvolution",  # noqa: F821
    conductance_aligned: np.ndarray,
    plif_aligned: np.ndarray,
) -> np.ndarray:
    """
    Fit and run the supervised deconvolution model.

    Parameters
    ----------
    model:
        SupervisedDeconvolution instance.
    conductance_aligned:
        Aligned conductance signal.
    plif_aligned:
        Aligned PLIF signal.

    Returns
    -------
    np.ndarray
        Reconstructed PLIF signal.
    """
    model.feed_data(y=conductance_aligned, u=plif_aligned)
    model.estimate_params()
    return model.reconstruct_signal()


def safe_metrics(reference: np.ndarray, estimate: np.ndarray, label: str) -> None:
    """
    Compute metrics while guarding against NaNs and constant signals.

    Parameters
    ----------
    reference:
        Reference signal (e.g., PLIF).
    estimate:
        Estimate signal.
    label:
        Label for the output.
    """
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    valid = np.isfinite(reference) & np.isfinite(estimate)
    if valid.sum() < 3:
        print(f"{label}: not enough finite samples for metrics.")
        return

    print(f"\n{label}")
    print("-" * len(label))
    metrics(reference[valid], estimate[valid])


def slice_window(
    time_s: np.ndarray,
    *series: np.ndarray,
    window: tuple[float, float],
) -> tuple[np.ndarray, ...]:
    """
    Return time and series masked to a time window.

    Parameters
    ----------
    time_s:
        Time array.
    *series:
        Signal arrays to mask.
    window:
        Time window (t_min, t_max).

    Returns
    -------
    tuple[np.ndarray, ...]
        Masked time and series arrays.
    """
    t_min, t_max = window
    time_s = np.asarray(time_s)
    mask = (time_s > t_min) & (time_s <= t_max)
    return tuple(np.asarray(arr)[mask] for arr in (time_s, *series))


def plot_estimations(
    exp_id: str,
    loc_key: str,
    zone: str,
    window: tuple[float, float],
    time_s_aligned: np.ndarray,
    plif_aligned: np.ndarray,
    raw_est: np.ndarray,
    deriv_est: np.ndarray,
    deconv_est: np.ndarray,
    save: bool = True,
    fig_dir: Path | None = None,
) -> None:
    """
    Plot PLIF ground truth against raw, derivative, and deconvolution estimates.

    Parameters
    ----------
    exp_id:
        Experiment identifier.
    loc_key:
        Location key.
    zone:
        Zone identifier.
    window:
        Time window for display.
    time_s_aligned:
        Aligned time array.
    plif_aligned:
        Aligned PLIF array.
    raw_est:
        Raw conductance estimate.
    deriv_est:
        Derivative estimate.
    deconv_est:
        Deconvolution estimate.
    save:
        Whether to save the figure.
    fig_dir:
        Directory to save figures. Defaults to <project_root>/figs.
    """
    t_masked, plif_masked, raw_masked, deriv_masked, deconv_masked = slice_window(
        time_s_aligned,
        plif_aligned,
        raw_est,
        deriv_est,
        deconv_est,
        window=window,
    )

    fig, ax = plt.subplots(nrows=3, figsize=(8, 4), sharex=True)
    ycolor = "red"

    ax[0].plot(t_masked, plif_masked, color="black", label="PLIF")
    ax0_twin = ax[0].twinx()
    ax0_twin.plot(t_masked, 1e6 * raw_masked, color=ycolor, label="e-nose raw")

    ax[1].plot(t_masked, plif_masked, color="black", label="PLIF")
    ax1_twin = ax[1].twinx()
    ax1_twin.plot(t_masked, 1e6 * deriv_masked, color=ycolor, label="e-nose d/dt")

    ax[2].plot(t_masked, plif_masked, color="black", label="PLIF")
    ax2_twin = ax[2].twinx()
    ax2_twin.plot(t_masked, deconv_masked, color=ycolor, label="e-nose deconvolution")
    ax2_twin.sharey(ax[2])

    ax[2].set_xlabel("Time [s]")
    ax[1].set_ylabel("Concentration [rel.]")
    ax0_twin.set_ylabel("Raw\nconductance\n" + r"[$\mu$S]")
    ax1_twin.set_ylabel("d/dt\nconductance\n" + r"[$\mu$S s$^{-1}$]")
    ax2_twin.set_ylabel("Estimated\nconcentration\n[rel.]")

    for twin in [ax0_twin, ax1_twin, ax2_twin]:
        twin.tick_params(axis="y", colors=ycolor)
        twin.yaxis.label.set_color(ycolor)
        twin.spines["top"].set_visible(False)

    label_x = 1.08
    ax0_twin.yaxis.set_label_coords(label_x, 0.5)
    ax1_twin.yaxis.set_label_coords(label_x, 0.5)
    ax2_twin.yaxis.set_label_coords(label_x, 0.5)

    for primary in ax:
        primary.grid(which="major", alpha=0.6, linestyle="--")
        primary.minorticks_on()
        primary.spines[["top", "right"]].set_visible(False)

    x0 = window[0]
    text_y = np.nanpercentile(plif_masked, 88) if len(plif_masked) else 0.0
    ax[0].text(x0 + 0.05, text_y, "e-nose raw", color=ycolor, weight="bold", size=11)
    ax[1].text(x0 + 0.05, text_y, "e-nose d/dt", color=ycolor, weight="bold", size=11)
    ax[2].text(x0 + 0.05, text_y, "e-nose deconvolution", color=ycolor, weight="bold", size=11)
    ax[0].text(
        window[1] - 0.25 * (window[1] - window[0]),
        text_y,
        "PLIF",
        color="black",
        weight="bold",
        size=11,
    )

    fig.suptitle(f"r{exp_id}, {loc_key}, zone {zone}", y=1.02)
    plt.tight_layout()

    if save and fig_dir is not None:
        out_path = fig_dir / f"r{exp_id}_{loc_key}_{zone}_concentration_estimation.svg"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


def process_zone(
    exp_id: str | int,
    zone: str,
    *,
    df: pd.DataFrame,
    sampling_strategy: Literal["interpolated", "plif_native", "common_rate"] = "interpolated",
    alignment_factory: Callable[[], object] | None = None,
    derivative_transform: Callable[[np.ndarray, float], np.ndarray] | None = None,
    deconv_factory: Callable[[float], "SupervisedDeconvolution"] | None = None,
    window: tuple[float, float] = (120.0, 130.0),
    save_figures: bool = True,
    fig_dir: Path | None = None,
) -> dict[str, np.ndarray | str]:
    """
    Run the concentration-estimation workflow for one experiment and one zone.

    Parameters
    ----------
    exp_id:
        Experiment identifier.
    zone:
        Zone identifier.
    df:
        Combined e-nose/PLIF DataFrame (already loaded with desired sampling strategy).
    sampling_strategy:
        Sampling strategy used (for logging).
    alignment_factory:
        Optional factory for the alignment object. Defaults to DeadTimeCompensator.
    derivative_transform:
        Optional derivative transform function. Defaults to Savitzky-Golay.
    deconv_factory:
        Optional factory for the deconvolution model. Defaults to SupervisedDeconvolution.
    window:
        Time window for plotting.
    save_figures:
        Whether to save figures.
    fig_dir:
        Directory to save figures.

    Returns
    -------
    dict[str, np.ndarray | str]
        Results dictionary with aligned signals.
    """
    from src.deadtime_compensation import DeadTimeCompensator
    from src.supervised_deconv import SupervisedDeconvolution

    exp_id_norm = str(exp_id).lower().strip()
    exp_id_norm = exp_id_norm[1:] if exp_id_norm.startswith("r") else exp_id_norm
    loc_key = LOCATIONS[f"r{exp_id_norm}"]

    time_s, plif, conductance, sensor_col = prepare_zone_signals(df, zone)

    print(f"\nExperiment r{exp_id_norm}, location {loc_key}, zone {zone}")
    print("-" * 70)
    print(f"Sensor column:       {sensor_col}")
    print(f"Sampling strategy:   {sampling_strategy}")
    print(f"Input samples:       {len(time_s)}")

    # Use defaults if not provided
    if alignment_factory is None:
        alignment_factory = DeadTimeCompensator
    if derivative_transform is None:
        derivative_transform = default_derivative_transform
    if deconv_factory is None:
        deconv_factory = lambda dt: SupervisedDeconvolution(delta_t=dt)

    aligner = alignment_factory()
    time_aligned, conductance_aligned, plif_aligned = align_signals(
        aligner, time_s, conductance, plif
    )

    dt = float(np.median(np.diff(time_aligned)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time step after alignment.")

    deriv_est = derivative_transform(conductance_aligned, dt)

    deconv = deconv_factory(dt)
    deconv_est = run_deconvolution(deconv, conductance_aligned, plif_aligned)

    safe_metrics(plif_aligned, conductance_aligned, "Raw conductance vs PLIF")
    safe_metrics(plif_aligned, deriv_est, "Conductance derivative vs PLIF")
    safe_metrics(plif_aligned, deconv_est, "Supervised deconvolution vs PLIF")

    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent.parent / "figs"

    plot_estimations(
        exp_id=exp_id_norm,
        loc_key=loc_key,
        zone=zone,
        window=window,
        time_s_aligned=time_aligned,
        plif_aligned=plif_aligned,
        raw_est=conductance_aligned,
        deriv_est=deriv_est,
        deconv_est=deconv_est,
        save=save_figures,
        fig_dir=fig_dir,
    )

    return {
        "experiment_id": f"r{exp_id_norm}",
        "location": loc_key,
        "zone": zone,
        "sensor_col": sensor_col,
        "time_s": time_aligned,
        "plif": plif_aligned,
        "raw_conductance": conductance_aligned,
        "conductance_derivative": deriv_est,
        "deconvolution": deconv_est,
    }


def process_experiment(
    exp_id: str | int,
    *,
    df: pd.DataFrame,
    zones: list[str] | None = None,
    **kwargs,
) -> dict[str, dict[str, np.ndarray | str]]:
    """
    Run the concentration-estimation workflow for all selected zones.

    Parameters
    ----------
    exp_id:
        Experiment identifier.
    df:
        Combined e-nose/PLIF DataFrame.
    zones:
        List of zones to process. Defaults to all zones for the location.
    **kwargs:
        Additional keyword arguments passed to process_zone.

    Returns
    -------
    dict[str, dict[str, np.ndarray | str]]
        Results dictionary keyed by zone.
    """
    exp_id_norm = str(exp_id).lower().strip()
    exp_id_norm = exp_id_norm[1:] if exp_id_norm.startswith("r") else exp_id_norm
    loc_key = LOCATIONS[f"r{exp_id_norm}"]

    if zones is None:
        zones = list(PLIF_USEFUL_BOUNDS[loc_key].keys())

    results = {}
    for zone in zones:
        results[zone] = process_zone(exp_id_norm, zone, df=df, **kwargs)

    return results