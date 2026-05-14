#%%
"""Estimate PLIF concentration from e-nose signals with swappable stages."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.deadtime_compensation import DeadTimeCompensator
from src.supervised_deconv import SupervisedDeconvolution
from utils.plotting import configure_style, export_figure
from utils.constants import ENOSE_ZONE_SENSOR, LOCATIONS, PLIF_USEFUL_BOUNDS
from utils.combined_io import load_combined_csv
from utils.misc import metrics

# DEFAULT_EXPERIMENTS = ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]
DEFAULT_EXPERIMENTS = ["56"]


def load_enose_plif(exp_id: str) -> pd.DataFrame:
    """Load combined e-nose/PLIF data for an experiment id."""
    return load_combined_csv(f"r{exp_id}", sampling="interpolated")


def default_alignment():
    """Factory for the default dead-time compensator (swap for custom aligners)."""
    return DeadTimeCompensator()


def default_derivative_transform(conductance: np.ndarray, dt: float) -> np.ndarray:
    """Differentiate and smooth conductance (swap for custom filters).
    
    Uses an adaptive window length based on data size to avoid
    savgol_filter errors when the array is smaller than the fixed window.
    """
    gradient = np.gradient(conductance, dt)
    # window_length must be odd and <= len(gradient)
    window_length = min(100, len(gradient))
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, 5)  # minimum valid window
    return savgol_filter(gradient, window_length, 5)


def default_deconv_model(delta_t: float = 1e-3):
    """Factory for the default supervised deconvolution model (plug in your own)."""
    return SupervisedDeconvolution(delta_t=delta_t)


def align_signals(aligner, time_s, conductance, plif):
    """Align conductance and PLIF using an alignment object with fit_transform."""
    return aligner.fit_transform(time_s, conductance, plif)


def run_deconvolution(model, conductance_aligned: np.ndarray, plif_aligned: np.ndarray) -> np.ndarray:
    """Run supervised deconvolution model to reconstruct PLIF."""
    model.feed_data(y=conductance_aligned, u=plif_aligned)
    model.estimate_params()
    return model.reconstruct_signal()


def slice_window(time_s: np.ndarray, *series: np.ndarray, window: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Return time and series masked to a time window."""
    t_min, t_max = window
    mask = (time_s <= t_max) & (time_s > t_min)
    return tuple(arr[mask] for arr in (time_s, *series))


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
) -> None:
    """Plot PLIF ground truth against three estimation methods."""
    t_masked, plif_masked, raw_masked, deriv_masked, deconv_masked = slice_window(
        time_s_aligned, plif_aligned, raw_est, deriv_est, deconv_est, window=window
    )

    fig, ax = plt.subplots(nrows=3, figsize=(8, 4), sharex=True)
    ycolor = "red"

    ax[0].plot(t_masked, plif_masked, color="black", label="Ground Truth")
    ax0_twin = ax[0].twinx()
    ax0_twin.plot(t_masked, 1e6 * raw_masked, color=ycolor, label="Estimation")

    ax[1].plot(t_masked, plif_masked, color="black", label="Ground Truth")
    ax1_twin = ax[1].twinx()
    ax1_twin.plot(t_masked, 1e6 * deriv_masked, color=ycolor, label="Estimation")

    ax[2].plot(t_masked, plif_masked, color="black", label="Ground Truth")
    ax2_twin = ax[2].twinx()
    ax2_twin.plot(t_masked, deconv_masked, color=ycolor, label="Estimation")
    ax2_twin.sharey(ax[2])

    ax[2].set_xlabel("Time (s)")
    ax[1].set_ylabel("Concentration (rel.)")
    ax0_twin.set_ylabel("raw\nConductance\n" + r"($\mu$S)")
    ax1_twin.set_ylabel("d/dt\nConductance\n" + r"($\mu$S/s)")
    ax2_twin.set_ylabel("Estimated\nConcentration\n(rel.)")

    ax0_twin.tick_params(axis="y", colors=ycolor)
    ax1_twin.tick_params(axis="y", colors=ycolor)
    ax2_twin.tick_params(axis="y", colors=ycolor)

    label_x = 1.08
    ax0_twin.yaxis.set_label_coords(label_x, 0.5)
    ax1_twin.yaxis.set_label_coords(label_x, 0.5)
    ax2_twin.yaxis.set_label_coords(label_x, 0.5)

    for axx in [ax[0], ax[1], ax[2], ax0_twin, ax1_twin, ax2_twin]:
        axx.spines[["top"]].set_visible(False)

    for axx in [ax[0], ax[1], ax[2]]:
        axx.grid(which="major", alpha=0.6, linestyle="--")
        axx.minorticks_on()
        axx.spines[["right"]].set_visible(False)

    ax[0].text(window[0] + 3.5, 0.037, "PLIF", color="k", weight="bold", size=11)
    ax[0].text(window[0] + 0.05, 0.035, "e-nose raw", color=ycolor, weight="bold", size=11)
    ax[1].text(window[0] + 0.05, 0.035, "e-nose d/dt", color=ycolor, weight="bold", size=11)
    ax[2].text(window[0] + 0.05, 0.035, "e-nose deconvolution", color=ycolor, weight="bold", size=11)

    export_figure(fig, base_name=f"r{exp_id}_{loc_key}_{zone}_concentration_estimation", bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def process_experiment(
    exp_id: str,
    alignment_factory=default_alignment,
    derivative_transform=default_derivative_transform,
    deconv_factory=default_deconv_model,
    window: tuple[float, float] = (120, 130),
) -> None:
    """Run estimation pipeline for one experiment with pluggable stages.

    Pass custom factories/transforms to inject alternative alignment (dead-time
    models), filtering/derivative logic, or deconvolution/inversion methods.
    """
    df = load_enose_plif(exp_id)
    loc_key = LOCATIONS[f"r{exp_id}"]

    for zone, _ in PLIF_USEFUL_BOUNDS[loc_key].items():
        print(f"Zone {zone}")
        sensor_col = ENOSE_ZONE_SENSOR[zone]
        time_s = df["time_s"].to_numpy()
        plif = df[f"plif_{zone}"].to_numpy()
        conductance = 1.0 / df[sensor_col].to_numpy()

        aligner = alignment_factory()
        time_aligned, conductance_aligned, plif_aligned = align_signals(aligner, time_s, conductance, plif)

        dt = float(np.mean(np.diff(time_aligned)))
        deriv_est = derivative_transform(conductance_aligned, dt)

        deconv = deconv_factory()
        deconv_est = run_deconvolution(deconv, conductance_aligned, plif_aligned)

        metrics(plif_aligned, conductance_aligned)
        metrics(plif_aligned, deriv_est)
        metrics(plif_aligned, deconv_est)

        plot_estimations(
            exp_id=exp_id,
            loc_key=loc_key,
            zone=zone,
            window=window,
            time_s_aligned=time_aligned,
            plif_aligned=plif_aligned,
            raw_est=conductance_aligned,
            deriv_est=deriv_est,
            deconv_est=deconv_est,
        )


def run_all_experiments(exp_ids: list[str] | None = None, **kwargs) -> None:
    """Run the estimation workflow across experiments."""
    exp_ids = exp_ids or DEFAULT_EXPERIMENTS
    for exp_id in exp_ids:
        print(f"Experiment r{exp_id}")
        process_experiment(exp_id, **kwargs)


if __name__ == "__main__":
    run_all_experiments()

# %%
