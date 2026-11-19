#%%
"""Estimate PLIF concentration from e-nose signals with swappable stages.

The e-nose signals are processed at their native 1 kHz rate. Two deconvolution
tracks are compared:

- frame rate: the model is fitted at the measured PLIF frames only (20 Hz,
  50 ms step between the two e-nose samples it uses);
- full rate: the model is fitted at every e-nose sample (1 kHz, 1 ms step),
  with the PCHIP-interpolated PLIF (plif_*_interp) as training target.

Both tracks are scored only against the measured PLIF frames, with blocked
cross-validation: the run is split into contiguous blocks, and each block is
estimated by a model fitted on the other blocks, leaving out a gap around the
held-out block. Interpolated PLIF values are never used as ground truth.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.deadtime_compensation import DeadTimeCompensator
from src.supervised_deconv import SupervisedDeconvolution
from utils.plotting import configure_style, export_figure
from utils.constants import ENOSE_ZONE_SENSOR, LOCATIONS, PLIF_USEFUL_BOUNDS, PLIF_SAMPLING_FREQUENCY, SPEEDS
from utils.combined_io import load_combined_csv, zero_lag_smooth_for_downsampling

ALL_EXPERIMENTS = ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]
DEFAULT_EXPERIMENTS = ["56"]
N_FOLDS = 5
GAP_S = 1.0
OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "concentration_estimation"


def load_enose_plif(exp_id: str) -> pd.DataFrame:
    """Load combined e-nose/PLIF data for an experiment id (saturated readings as NaN)."""
    return load_combined_csv(f"r{exp_id}", sampling="interpolated", mask_saturated=True)


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


def default_deconv_model(delta_t: float = 1.0 / PLIF_SAMPLING_FREQUENCY):
    """Factory for the default supervised deconvolution model (plug in your own)."""
    return SupervisedDeconvolution(delta_t=delta_t)


def align_signals(aligner, time_s, conductance, frame_mask, plif_frames):
    """Fit the dead time against the measured PLIF frames (aligner needs fit_frames)."""
    return aligner.fit_frames(time_s, conductance, frame_mask, plif_frames)


def cross_validated_deconvolution(model_factory, y: np.ndarray, u: np.ndarray, time_s: np.ndarray,
                                  n_folds: int = N_FOLDS, gap_s: float = GAP_S) -> np.ndarray:
    """Out-of-fold deconvolution estimates of u from y.

    The time range is split into `n_folds` contiguous blocks. For each block the
    model is fitted on the samples outside the block and outside a gap of
    `gap_s` on either side of it, and the block is reconstructed from y.
    """
    edges = np.linspace(time_s[0], time_s[-1], n_folds + 1)
    edges[-1] = np.inf
    estimate = np.full(len(y), np.nan)
    for start, stop in zip(edges[:-1], edges[1:]):
        held_out = (time_s >= start) & (time_s < stop)
        excluded = (time_s >= start - gap_s) & (time_s < stop + gap_s)
        model = model_factory()
        model.feed_data(y=y, u=np.where(excluded, np.nan, u))
        model.estimate_params()
        estimate[held_out] = model.reconstruct_signal(y)[held_out]
    return estimate


def correlations(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman and Pearson correlation over the samples where both signals are finite."""
    finite = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[finite], b[finite])[0]), float(np.corrcoef(a[finite], b[finite])[0, 1])


def estimate_zone(df: pd.DataFrame, zone: str, alignment_factory=default_alignment,
                  derivative_transform=default_derivative_transform,
                  deconv_factory=default_deconv_model) -> dict:
    """Run both deconvolution tracks for one zone and score them at the measured frames."""
    sensor_col = ENOSE_ZONE_SENSOR[zone]
    frames = df["plif_image"].to_numpy(dtype=bool)
    frame_idx = np.flatnonzero(frames)
    time_s = df["time_s"].to_numpy()
    plif = df.loc[frames, f"plif_{zone}"].to_numpy()
    plif_interp = df[f"plif_{zone}_interp"].to_numpy()
    conductance = 1.0 / df[sensor_col].to_numpy()

    # E-nose processing at 1 kHz: derivative, and low-pass (cutoff 8 Hz).
    dt = float(np.mean(np.diff(time_s)))
    derivative = derivative_transform(conductance, dt)
    conductance_lowpass = zero_lag_smooth_for_downsampling(
        conductance, source_rate_hz=1.0 / dt, target_rate_hz=PLIF_SAMPLING_FREQUENCY
    )

    # Dead time from the measured PLIF frames; e-nose signals shifted back by it.
    aligner = align_signals(alignment_factory(), time_s, conductance, frames, plif)
    lag = aligner.lag_points

    def shifted(signal):
        out = np.full(len(signal), np.nan)
        out[:len(signal) - lag] = signal[lag:]
        return out

    conductance_shifted = shifted(conductance)
    conductance_lowpass_shifted = shifted(conductance_lowpass)
    derivative_shifted = shifted(derivative)

    # Frame-rate track: fitted at the measured frames only.
    frame_rate = cross_validated_deconvolution(
        lambda: deconv_factory(1.0 / PLIF_SAMPLING_FREQUENCY),
        conductance_lowpass_shifted[frame_idx], plif, time_s[frame_idx],
    )
    # Full-rate track: fitted at every e-nose sample, interpolated PLIF as target.
    full_rate = cross_validated_deconvolution(
        lambda: deconv_factory(dt), conductance_lowpass_shifted, plif_interp, time_s,
    )

    scores = {}
    for name, estimate in (("raw", conductance_lowpass_shifted[frame_idx]), ("ddt", derivative_shifted[frame_idx]),
                           ("frame_rate", frame_rate), ("full_rate", full_rate[frame_idx])):
        scores[f"spearman_{name}"], scores[f"pearson_{name}"] = correlations(plif, estimate)
    return dict(
        sensor_col=sensor_col, dead_time_s=aligner.get_dead_time(), n_frames=len(frame_idx),
        time_s=time_s, frame_idx=frame_idx, plif=plif, conductance=conductance_shifted,
        derivative=derivative_shifted, frame_rate=frame_rate, full_rate=full_rate, scores=scores,
    )


def plot_estimations(exp_id: str, loc_key: str, zone: str, window: tuple[float, float], est: dict) -> None:
    """Plot measured PLIF frames against the e-nose estimates in a time window."""
    time_s, frame_idx = est["time_s"], est["frame_idx"]
    in_window = (time_s > window[0]) & (time_s <= window[1])
    frame_in_window = in_window[frame_idx]
    t_frames = time_s[frame_idx][frame_in_window]
    plif = est["plif"][frame_in_window]
    t = time_s[in_window]
    ycolor = "red"

    fig, ax = plt.subplots(nrows=4, figsize=(8, 5.2), sharex=True)
    panels = [
        (t, 1e6 * est["conductance"][in_window], "raw\nConductance\n" + r"($\mu$S)", "e-nose raw (1 kHz)"),
        (t, 1e6 * est["derivative"][in_window], "d/dt\nConductance\n" + r"($\mu$S/s)", "e-nose d/dt (1 kHz)"),
        (t_frames, est["frame_rate"][frame_in_window], "Estimated\nConcentration\n(rel.)",
         "deconvolution, frame rate (20 Hz)"),
        (t, est["full_rate"][in_window], "Estimated\nConcentration\n(rel.)", "deconvolution, full rate (1 kHz)"),
    ]
    for i, (axx, (tx, values, ylabel, label)) in enumerate(zip(ax, panels)):
        axx.plot(t_frames, plif, color="black", lw=0.6)
        axx.plot(t_frames, plif, "o", color="black", ms=1.8)
        twin = axx.twinx()
        twin.plot(tx, values, color=ycolor, lw=0.9, marker="o" if i == 2 else None, ms=1.8)
        if i >= 2:
            twin.sharey(axx)
        twin.set_ylabel(ylabel)
        twin.tick_params(axis="y", colors=ycolor)
        twin.yaxis.set_label_coords(1.08, 0.5)
        twin.spines[["top"]].set_visible(False)
        axx.spines[["top", "right"]].set_visible(False)
        axx.grid(which="major", alpha=0.6, linestyle="--")
        axx.minorticks_on()
        axx.text(0.01, 0.97, label, transform=axx.transAxes, color=ycolor, weight="bold", size=10, va="top")
    ax[0].text(0.99, 1.02, "PLIF (measured frames)", transform=ax[0].transAxes, color="k",
               weight="bold", size=10, va="bottom", ha="right")
    ax[-1].set_xlabel("Time (s)")
    fig.supylabel("Concentration (rel.)", x=0.04, size=plt.rcParams["axes.labelsize"])

    export_figure(fig, base_name=f"r{exp_id}_{loc_key}_{zone}_concentration_estimation", bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def process_experiment(
    exp_id: str,
    alignment_factory=default_alignment,
    derivative_transform=default_derivative_transform,
    deconv_factory=default_deconv_model,
    window: tuple[float, float] = (120, 130),
    plot: bool = True,
) -> list[dict]:
    """Run estimation pipeline for one experiment with pluggable stages.

    Pass custom factories/transforms to inject alternative alignment (dead-time
    models), filtering/derivative logic, or deconvolution/inversion methods.
    Returns one row of scores per zone.
    """
    df = load_enose_plif(exp_id)
    loc_key = LOCATIONS[f"r{exp_id}"]

    rows = []
    for zone in PLIF_USEFUL_BOUNDS[loc_key]:
        est = estimate_zone(df, zone, alignment_factory, derivative_transform, deconv_factory)
        print(f"  Zone {zone}, {est['sensor_col']}: dead time {1e3 * est['dead_time_s']:.0f} ms, "
              + ", ".join(f"{k} {v:.2f}" for k, v in est["scores"].items()))
        rows.append(dict(exp=f"r{exp_id}", location=loc_key[1:], speed_m_s=SPEEDS[f"r{exp_id}"], zone=zone,
                         sensor=est["sensor_col"].replace("R_gas_", ""),
                         dead_time_ms=round(1e3 * est["dead_time_s"]), n_frames=est["n_frames"],
                         **{k: round(v, 3) for k, v in est["scores"].items()}))
        if plot:
            plot_estimations(exp_id, loc_key, zone, window, est)
    return rows


def run_all_experiments(exp_ids: list[str] | None = None, **kwargs) -> pd.DataFrame:
    """Run the estimation workflow across experiments and return the scores."""
    exp_ids = exp_ids or DEFAULT_EXPERIMENTS
    rows = []
    for exp_id in exp_ids:
        print(f"Experiment r{exp_id}")
        rows += process_experiment(exp_id, **kwargs)
    return pd.DataFrame(rows)


def save_scores(exp_ids: list[str] | None = None) -> pd.DataFrame:
    """Score both tracks for all experiments and zones and save the tables."""
    scores = run_all_experiments(exp_ids or ALL_EXPERIMENTS, plot=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    scores["device"] = np.where(scores["zone"].isin(["A", "B"]), "MiCS-6814", "CCS801")
    means = scores.groupby(["device", "location", "speed_m_s"]).mean(numeric_only=True)
    # One column per method, "Spearman / Pearson"
    summary = pd.DataFrame({
        name: means[f"spearman_{name}"].map("{:.2f}".format) + " / " + means[f"pearson_{name}"].map("{:.2f}".format)
        for name in ("raw", "ddt", "frame_rate", "full_rate")
    })
    summary.to_csv(OUT_DIR / "scores_summary.csv")
    print(summary)
    return scores


if __name__ == "__main__":
    run_all_experiments()
    save_scores()

# %%
