#%%
"""Compute the technical-validation tables for all experiments.

Tables are written as CSV files to ``out/technical_validation``:

- ``saturation.csv``: percentage of non-saturated samples per sensor;
- ``heater_temperature.csv``: hotplate temperature statistics per sensor;
- ``environment.csv``: ambient temperature, relative humidity and pressure;
- ``synchronisation.csv``: timing of the PLIF trigger pulses recorded by the e-nose;
- ``plif_bandwidth.csv``: temporal content of the measured PLIF zone signals;
- ``sensor_plif_coherence.csv``: coherence between each sensor and its PLIF zone;
- ``response_time.csv``: dead time and effective first-order time constant per sensor;
- ``roi_geometry.csv``: position and size of the PLIF regions of interest (needs HDF5);
- ``roi_sensitivity.csv``: sensitivity of the zone signals to the ROI choice (needs HDF5).

Only measured PLIF frames are used; interpolated PLIF values are not.

Usage
-----
python scripts/technical_validation.py [--combined-dir DIR] [--plif-dir DIR] [--skip-plif]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.constants import (
    LOCAL_OUT_FOLDER,
    LOCATIONS,
    MOX_PLIF_BOUND_RELATIONSHIPS,
    PLIF_SAMPLING_FREQUENCY,
    PLIF_USEFUL_BOUNDS,
    SPEEDS,
    SSD_COMBINED_FOLDER,
    SSD_PLIF_FOLDER,
)
from utils.combined_io import load_combined_csv, zero_lag_smooth_for_downsampling
from utils.plif_io import open_plif_dataset

DEFAULT_EXPERIMENTS = ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]
N_SENSORS = 8
ZONES = ["A", "B", "C", "D"]
# Sensors that are not saturated in any experiment.
UNSATURATED_SENSORS = [1, 4, 5, 8]


def _run_info(exp_id: str) -> dict[str, object]:
    """Return the identifying columns shared by all tables."""
    return {
        "experiment": f"r{exp_id}",
        "location": LOCATIONS[f"r{exp_id}"][1:],
        "velocity_cm_s": SPEEDS[f"r{exp_id}"],
    }


# ---------------------------------------------------------------------
# E-nose channels
# ---------------------------------------------------------------------

def saturation_table(exp_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    """Percentage of non-saturated samples (R_gas not exactly 0) per sensor."""
    row = _run_info(exp_id)
    for i in range(1, N_SENSORS + 1):
        row[f"valid_percent_sensor_{i}"] = round(100 * df[f"R_gas_valid_{i}"].mean(), 1)
    return [row]


def heater_table(exp_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    """Hotplate temperature statistics per sensor, from all 1 kHz samples."""
    rows = []
    for i in range(1, N_SENSORS + 1):
        temperature = df[f"T_heat_{i}"].to_numpy()
        rows.append({
            **_run_info(exp_id),
            "sensor": i,
            "mean_degC": temperature.mean(),
            "std_degC": temperature.std(),
            "p05_degC": np.percentile(temperature, 5),
            "p95_degC": np.percentile(temperature, 95),
            "min_degC": temperature.min(),
            "max_degC": temperature.max(),
        })
    return rows


def environment_table(exp_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    """Ambient temperature, relative humidity and pressure ranges."""
    temperature = df["t_celsius"].to_numpy()
    humidity = df["rh_percent"].to_numpy()
    # Absolute humidity from the Magnus formula (g m^-3).
    saturation_pressure = 6.112 * np.exp(17.62 * temperature / (243.12 + temperature))
    absolute_humidity = 216.7 * (humidity / 100 * saturation_pressure) / (273.15 + temperature)

    row = _run_info(exp_id)
    for name, values in (("t_celsius", temperature), ("rh_percent", humidity), ("p_mbar", df["p_mbar"].to_numpy())):
        row[f"{name}_min"] = values.min()
        row[f"{name}_median"] = np.median(values)
        row[f"{name}_max"] = values.max()
    row["absolute_humidity_g_m3_median"] = np.median(absolute_humidity)
    return [row]


def synchronisation_table(exp_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    """
    Timing of the PLIF trigger pulses as timestamped by the e-nose.

    The trigger times are compared with an ideal, evenly spaced pulse train
    (linear fit of trigger time against frame number). The slope gives the PLIF
    frame period on the e-nose clock, the residuals the timing error per frame.
    """
    time_s = df["time_s"].to_numpy()
    trigger_times = time_s[df["plif_image"].to_numpy(dtype=bool)]
    frame_number = np.arange(len(trigger_times))

    period, offset = np.polyfit(frame_number, trigger_times, 1)
    residual_ms = 1e3 * (trigger_times - (period * frame_number + offset))
    enose_step_ms = 1e3 * np.diff(time_s)
    trigger_step_ms = 1e3 * np.diff(trigger_times)

    return [{
        **_run_info(exp_id),
        "n_enose_samples": len(time_s),
        "enose_step_mean_ms": enose_step_ms.mean(),
        "enose_step_std_ms": enose_step_ms.std(),
        "enose_step_max_ms": enose_step_ms.max(),
        "n_plif_triggers": len(trigger_times),
        "trigger_step_min_ms": trigger_step_ms.min(),
        "trigger_step_max_ms": trigger_step_ms.max(),
        "frame_period_ms": 1e3 * period,
        "clock_rate_difference_ppm": 1e6 * (period * PLIF_SAMPLING_FREQUENCY - 1),
        "residual_rms_ms": residual_ms.std(),
        "residual_p95_abs_ms": np.percentile(np.abs(residual_ms), 95),
        "residual_max_abs_ms": np.abs(residual_ms).max(),
    }]


# ---------------------------------------------------------------------
# PLIF zone signals and their relation to the sensors
# ---------------------------------------------------------------------

def plif_bandwidth_table(exp_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    """Temporal content of the measured (20 Hz) PLIF zone signals."""
    frames = df[df["plif_image"]]
    rows = []
    for zone in ZONES:
        concentration = frames[f"plif_{zone}"].to_numpy()
        freq, psd = signal.welch(concentration - concentration.mean(), fs=PLIF_SAMPLING_FREQUENCY, nperseg=512)
        upper_band = freq >= PLIF_SAMPLING_FREQUENCY / 4
        # The two laser heads fire alternately, so they map onto even and odd frames.
        head_offset = concentration[0::2].mean() - concentration[1::2].mean()
        rows.append({
            **_run_info(exp_id),
            "zone": zone,
            "mean": concentration.mean(),
            "std": concentration.std(),
            "consecutive_frame_correlation": np.corrcoef(concentration[1:], concentration[:-1])[0, 1],
            "variance_fraction_5_10_Hz": np.trapezoid(psd[upper_band], freq[upper_band]) / np.trapezoid(psd, freq),
            "laser_head_offset_even_minus_odd": head_offset,
            "laser_head_offset_relative_to_mean": abs(head_offset) / concentration.mean(),
        })
    return rows


def _frame_log_conductance(df: pd.DataFrame, sensor: int) -> np.ndarray:
    """Log conductance at 1 kHz, low-pass filtered for sampling at the PLIF frame rate."""
    log_conductance = np.log(1.0 / df[f"R_gas_{sensor}"].to_numpy())
    dt = float(np.mean(np.diff(df["time_s"].to_numpy())))
    return zero_lag_smooth_for_downsampling(
        log_conductance, source_rate_hz=1.0 / dt, target_rate_hz=PLIF_SAMPLING_FREQUENCY
    )


def coherence_table(exp_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    """Magnitude-squared coherence between each sensor and its PLIF zone, at the PLIF frames."""
    frame_mask = df["plif_image"].to_numpy(dtype=bool)
    rows = []
    for sensor in UNSATURATED_SENSORS:
        zone = MOX_PLIF_BOUND_RELATIONSHIPS[str(sensor)]
        log_conductance = _frame_log_conductance(df, sensor)[frame_mask]
        concentration = df.loc[frame_mask, f"plif_{zone}"].to_numpy()
        freq, coherence = signal.coherence(log_conductance, concentration, fs=PLIF_SAMPLING_FREQUENCY, nperseg=256)

        # Lowest frequency above the coherence maximum at which coherence falls below 0.5.
        peak = int(np.argmax(coherence))
        below = np.flatnonzero(coherence[peak:] < 0.5)
        if coherence[peak] < 0.5:
            limit_hz = np.nan
        elif len(below) == 0:
            limit_hz = freq[-1]
        else:
            limit_hz = freq[peak + below[0]]

        rows.append({
            **_run_info(exp_id),
            "sensor": sensor,
            "zone": zone,
            "coherence_0.1_1_Hz": coherence[(freq >= 0.1) & (freq <= 1)].mean(),
            "coherence_1_5_Hz": coherence[(freq > 1) & (freq <= 5)].mean(),
            "coherence_5_10_Hz": coherence[freq > 5].mean(),
            "coherence_above_0.5_up_to_Hz": limit_hz,
        })
    return rows


def response_time_table(exp_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    """
    Dead time and effective first-order time constant of each sensor.

    Model: log G(t + dead time) ~ low-pass_tau[log C](t), evaluated at the measured
    PLIF frames. The dead time is searched on the e-nose time grid, the time
    constant on a logarithmic grid; both maximise the correlation. Slow drift
    (below 0.05 Hz) is removed from both signals beforehand.
    """
    frame_idx = np.flatnonzero(df["plif_image"].to_numpy(dtype=bool))
    frame_dt = 1.0 / PLIF_SAMPLING_FREQUENCY
    highpass = signal.butter(2, 0.05, "highpass", fs=PLIF_SAMPLING_FREQUENCY, output="sos")
    time_constants = np.geomspace(0.05, 5.0, 41)
    dead_times_ms = np.arange(0, 301, 10)
    settle = 10 * PLIF_SAMPLING_FREQUENCY

    rows = []
    for sensor in UNSATURATED_SENSORS:
        zone = MOX_PLIF_BOUND_RELATIONSHIPS[str(sensor)]
        log_conductance = _frame_log_conductance(df, sensor)
        concentration = df[f"plif_{zone}"].to_numpy()[frame_idx]
        log_concentration = np.log(concentration + 1e-3)

        usable = frame_idx + dead_times_ms[-1] < len(df)
        best = (-np.inf, np.nan, np.nan)
        for tau in time_constants:
            pole = np.exp(-frame_dt / tau)
            model = signal.lfilter([1 - pole], [1, -pole], log_concentration)
            model = signal.sosfiltfilt(highpass, model)[usable][settle:]
            for dead_time_ms in dead_times_ms:
                response = signal.sosfiltfilt(highpass, log_conductance[frame_idx[usable] + dead_time_ms])[settle:]
                r = np.corrcoef(model, response)[0, 1]
                if r > best[0]:
                    best = (r, tau, dead_time_ms)

        rows.append({
            **_run_info(exp_id),
            "sensor": sensor,
            "zone": zone,
            "dead_time_ms": best[2],
            "time_constant_s": best[1],
            "rise_time_10_90_s": 2.2 * best[1],
            "correlation": best[0],
        })
    return rows


# ---------------------------------------------------------------------
# Regions of interest (HDF5)
# ---------------------------------------------------------------------

def roi_tables(exp_id: str, df: pd.DataFrame, plif_dir: Path) -> tuple[list[dict], list[dict]]:
    """ROI geometry and sensitivity of the zone signals to the ROI choice."""
    loc_key = LOCATIONS[f"r{exp_id}"]
    margin = 12
    frames = df[df["plif_image"]]

    h5f, data = open_plif_dataset(f"r{exp_id}", plif_root=plif_dir)
    try:
        px_per_mm = float(h5f["Mapping"]["scaling: pixels to mm"][()].decode().split()[0])
        streamwise_grid = h5f["Mapping"]["x_grid"][()]
        cross_stream_grid = h5f["Mapping"]["y_grid"][()]
        measured_mask = h5f["Mapping"]["mask for interpolated areas"][()].astype(bool)

        bounds = PLIF_USEFUL_BOUNDS[loc_key]
        row0 = min(b[0] for b in bounds.values()) - margin
        row1 = max(b[1] for b in bounds.values()) + margin
        col0 = min(b[2] for b in bounds.values()) - margin
        col1 = max(b[3] for b in bounds.values()) + margin
        block = data[:, row0:row1, col0:col1]
    finally:
        h5f.close()

    def roi_mean(rmin: int, rmax: int, cmin: int, cmax: int) -> np.ndarray:
        return block[:, rmin - row0:rmax - row0, cmin - col0:cmax - col0].mean(axis=(1, 2))

    geometry, sensitivity = [], []
    for zone, (rmin, rmax, cmin, cmax) in bounds.items():
        base = roi_mean(rmin, rmax, cmin, cmax)
        geometry.append({
            **_run_info(exp_id),
            "zone": zone,
            "row_min": rmin, "row_max": rmax, "col_min": cmin, "col_max": cmax,
            "cross_stream_size_mm": (rmax - rmin) / px_per_mm,
            "streamwise_size_mm": (cmax - cmin) / px_per_mm,
            "centre_streamwise_mm": streamwise_grid[rmin:rmax, cmin:cmax].mean(),
            "centre_cross_stream_mm": cross_stream_grid[rmin:rmax, cmin:cmax].mean(),
            "n_pixels": (rmax - rmin) * (cmax - cmin),
            "n_interpolated_pixels": int((~measured_mask[rmin:rmax, cmin:cmax]).sum()),
            "max_abs_difference_to_csv": np.abs(base - frames[f"plif_{zone}"].to_numpy()).max(),
        })

        centre_pixel = block[:, (rmin + rmax) // 2 - row0, (cmin + cmax) // 2 - col0]
        shrunk = roi_mean(rmin + 2, rmax - 2, cmin + 2, cmax - 2)
        enlarged = roi_mean(rmin - 3, rmax + 3, cmin - 3, cmax + 3)
        row = {
            **_run_info(exp_id),
            "zone": zone,
            "r_centre_pixel": np.corrcoef(base, centre_pixel)[0, 1],
            "std_ratio_centre_pixel": centre_pixel.std() / base.std(),
            "r_shrunk_2px": np.corrcoef(base, shrunk)[0, 1],
            "std_ratio_shrunk_2px": shrunk.std() / base.std(),
            "r_enlarged_3px": np.corrcoef(base, enlarged)[0, 1],
            "std_ratio_enlarged_3px": enlarged.std() / base.std(),
        }
        for shift_px in (3, 7):
            shifted = [
                np.corrcoef(base, roi_mean(rmin + dr, rmax + dr, cmin + dc, cmax + dc))[0, 1]
                for dr, dc in ((shift_px, 0), (-shift_px, 0), (0, shift_px), (0, -shift_px))
            ]
            row[f"r_shifted_{shift_px}px_min"] = min(shifted)
        sensitivity.append(row)

    return geometry, sensitivity


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--combined-dir", type=Path, default=SSD_COMBINED_FOLDER)
    parser.add_argument("--plif-dir", type=Path, default=SSD_PLIF_FOLDER)
    parser.add_argument("--out-dir", type=Path, default=LOCAL_OUT_FOLDER.joinpath("technical_validation"))
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--skip-plif", action="store_true", help="Skip the tables that need the PLIF HDF5 files.")
    args = parser.parse_args()

    tables: dict[str, list[dict[str, object]]] = {
        name: [] for name in (
            "saturation", "heater_temperature", "environment", "synchronisation",
            "plif_bandwidth", "sensor_plif_coherence", "response_time",
            "roi_geometry", "roi_sensitivity",
        )
    }

    for exp_id in args.experiments:
        print(f"Experiment r{exp_id}")
        df = load_combined_csv(exp_id, combined_root=args.combined_dir)

        tables["saturation"] += saturation_table(exp_id, df)
        tables["heater_temperature"] += heater_table(exp_id, df)
        tables["environment"] += environment_table(exp_id, df)
        tables["synchronisation"] += synchronisation_table(exp_id, df)
        tables["plif_bandwidth"] += plif_bandwidth_table(exp_id, df)
        tables["sensor_plif_coherence"] += coherence_table(exp_id, df)
        tables["response_time"] += response_time_table(exp_id, df)

        if not args.skip_plif:
            geometry, sensitivity = roi_tables(exp_id, df, args.plif_dir)
            tables["roi_geometry"] += geometry
            tables["roi_sensitivity"] += sensitivity

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            out_path = args.out_dir.joinpath(f"{name}.csv")
            pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6g")
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
