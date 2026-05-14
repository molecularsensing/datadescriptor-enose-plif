"""
Combined e-nose/PLIF CSV I/O helpers.

This module provides utilities for loading, inspecting, and resampling the
combined e-nose/PLIF CSV data stored on disk. It centralises common patterns
shared across analysis scripts:

- locating combined CSV files from experiment IDs;
- loading and parsing combined CSV data;
- detecting PLIF, e-nose, and numeric columns;
- resampling to native PLIF, interpolated, or common-rate time bases;
- exporting resampled data back to CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Import constants from utils.constants
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from utils.constants import SSD_COMBINED_FOLDER
except ImportError:
    SSD_COMBINED_FOLDER = Path(__file__).resolve().parent.parent / "data" / "enose-plif-combined"


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

SamplingStrategy = Literal["interpolated", "plif_native", "common_rate"]


# ---------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------

def normalise_exp_id(exp_id: str | int) -> str:
    """
    Normalise experiment identifiers.

    Examples
    --------
    >>> normalise_exp_id(56)
    '56'
    >>> normalise_exp_id("56")
    '56'
    >>> normalise_exp_id("r56")
    '56'
    """
    exp_id = str(exp_id).lower().strip()
    return exp_id[1:] if exp_id.startswith("r") else exp_id


def get_combined_csv_path(
    exp_id: str | int,
    combined_root: Path = SSD_COMBINED_FOLDER,
) -> Path:
    """
    Return the expected combined e-nose/PLIF CSV path.
    """
    exp_id = normalise_exp_id(exp_id)
    return combined_root.joinpath(f"enose_plif_r{exp_id}.csv")


def _load_raw_csv(
    exp_id: str | int,
    combined_root: Path = SSD_COMBINED_FOLDER,
) -> pd.DataFrame:
    """
    Internal: Load raw CSV without any sampling or processing.
    """
    csv_path = get_combined_csv_path(exp_id, combined_root=combined_root)

    if not csv_path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_timedelta(df["timestamp"])

    if "time_s" not in df.columns:
        if "timestamp" in df.columns:
            df["time_s"] = df["timestamp"].dt.total_seconds()
        else:
            raise KeyError(
                "Expected either a `time_s` column or a `timestamp` column "
                f"in {csv_path}"
            )

    df = df.sort_values("time_s").reset_index(drop=True)

    return df


def load_combined_csv(
    exp_id: str | int,
    combined_root: Path = SSD_COMBINED_FOLDER,
    sampling: Literal["native", "interpolated", "common_rate"] | None = None,
    target_rate_hz: float = 100.0,
    plif_source: Literal["native", "interpolated"] = "interpolated",
    smoothing_window_s: float | None = None,
) -> pd.DataFrame:
    """
    Load the combined e-nose/PLIF CSV for one experiment.

    The returned DataFrame contains a numeric `time_s` column and, if present,
    a parsed timedelta `timestamp` column.

    Parameters
    ----------
    exp_id:
        Experiment identifier (e.g., '56', 'r56', or 56).
    sampling:
        Optional sampling strategy:
        - "native": Return only rows corresponding to original PLIF images.
        - "interpolated": Return the full combined CSV as provided (default).
        - "common_rate": Return a regular, user-defined common time base.
    target_rate_hz:
        Used only when sampling="common_rate".
    plif_source:
        Used only when sampling="common_rate".
    smoothing_window_s:
        Optional centred zero-lag smoothing window for downsampling.
    """
    # If sampling is specified, use the full load_combined_dataset pipeline
    if sampling is not None:
        strategy_map = {
            "native": "plif_native",
            "interpolated": "interpolated",
            "common_rate": "common_rate",
        }
        strategy = strategy_map.get(sampling, sampling)
        return load_combined_dataset(
            exp_id,
            strategy=strategy,
            target_rate_hz=target_rate_hz,
            plif_source=plif_source,
            smoothing_window_s=smoothing_window_s,
        )

    return _load_raw_csv(exp_id, combined_root=combined_root)


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

def get_plif_columns(df: pd.DataFrame) -> list[str]:
    """
    Return PLIF concentration columns.

    By convention these are expected to start with `plif_`, excluding helper
    columns such as `plif_image`.
    """
    return [
        col for col in df.columns
        if col.startswith("plif_") and col != "plif_image"
    ]


def get_enose_columns(df: pd.DataFrame) -> list[str]:
    """
    Return likely e-nose columns.

    This intentionally uses a conservative exclusion rule. Adjust the prefixes
    here if your final CSV schema uses more specific column names.
    """
    excluded = {
        "timestamp",
        "time_s",
        "plif_image",
    }

    excluded_prefixes = (
        "plif_",
    )

    return [
        col for col in df.columns
        if col not in excluded
        and not col.startswith(excluded_prefixes)
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def get_numeric_data_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all numeric data columns excluding time/index/helper columns.
    """
    excluded = {
        "timestamp",
        "time_s",
        "plif_image",
    }

    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def infer_native_rate_hz(time_s: np.ndarray | pd.Series) -> float:
    """
    Estimate sampling frequency from a time vector using the median time step.
    """
    time = np.asarray(time_s, dtype=float)
    dt = np.diff(time)

    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if len(dt) == 0:
        raise ValueError("Could not infer sampling rate from time vector.")

    return 1.0 / np.median(dt)


# ---------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------

def inspect_combined_csv(exp_id: str | int, verbose: bool = True) -> dict[str, object]:
    """
    Inspect the combined CSV without modifying it.
    """
    df = load_combined_csv(exp_id)
    plif_cols = get_plif_columns(df)
    enose_cols = get_enose_columns(df)

    info: dict[str, object] = {
        "experiment_id": f"r{normalise_exp_id(exp_id)}",
        "path": str(get_combined_csv_path(exp_id)),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "time_min_s": float(df["time_s"].min()),
        "time_max_s": float(df["time_s"].max()),
        "estimated_csv_rate_hz": infer_native_rate_hz(df["time_s"]),
        "has_plif_image_column": "plif_image" in df.columns,
        "n_original_plif_rows": int(df["plif_image"].sum()) if "plif_image" in df.columns else None,
        "plif_columns": plif_cols,
        "enose_columns": enose_cols,
    }

    if "plif_image" in df.columns and df["plif_image"].any():
        measured = df[df["plif_image"]]
        info["estimated_plif_native_rate_hz"] = infer_native_rate_hz(measured["time_s"])

    if verbose:
        print_combined_info(info)

    return info


def print_combined_info(info: dict[str, object]) -> None:
    """
    Pretty-print combined CSV information.
    """
    print("\nCombined e-nose/PLIF CSV")
    print("-" * 60)
    print(f"Experiment ID:          {info['experiment_id']}")
    print(f"Path:                   {info['path']}")
    print(f"Rows:                   {info['n_rows']}")
    print(f"Columns:                {info['n_columns']}")
    print(f"Time range:             {info['time_min_s']:.3f}–{info['time_max_s']:.3f} s")
    print(f"Estimated CSV rate:     {info['estimated_csv_rate_hz']:.2f} Hz")
    print(f"Has PLIF image marker:  {info['has_plif_image_column']}")

    if info.get("n_original_plif_rows") is not None:
        print(f"Original PLIF rows:     {info['n_original_plif_rows']}")

    if info.get("estimated_plif_native_rate_hz") is not None:
        print(f"Estimated PLIF rate:    {info['estimated_plif_native_rate_hz']:.2f} Hz")

    print("\nPLIF columns")
    print("-" * 60)
    for col in info["plif_columns"]:
        print(f"  {col}")

    print("\nE-nose / auxiliary numeric columns")
    print("-" * 60)
    for col in info["enose_columns"]:
        print(f"  {col}")

    print()


# ---------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------

def select_plif_native_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select rows corresponding to original PLIF image acquisitions.

    Requires a Boolean `plif_image` column.
    """
    if "plif_image" not in df.columns:
        raise KeyError(
            "Cannot select native PLIF rows because column `plif_image` is absent."
        )

    return df[df["plif_image"].astype(bool)].copy().reset_index(drop=True)


def make_target_timebase(
    time_min_s: float,
    time_max_s: float,
    target_rate_hz: float,
) -> np.ndarray:
    """
    Create a regular target time base.

    The endpoint is included only if it falls exactly on the requested grid.
    """
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive.")

    dt = 1.0 / target_rate_hz
    n_samples = int(np.floor((time_max_s - time_min_s) / dt)) + 1

    return time_min_s + np.arange(n_samples) * dt


def zero_lag_smooth_for_downsampling(
    values: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
    smoothing_window_s: float | None = None,
) -> np.ndarray:
    """
    Apply a centred, zero-lag rolling mean before downsampling.

    Parameters
    ----------
    values:
        One-dimensional signal.
    source_rate_hz:
        Estimated source sampling frequency.
    target_rate_hz:
        Requested target sampling frequency.
    smoothing_window_s:
        Optional smoothing window in seconds. If None, a default window of one
        target-sampling interval is used.

    Returns
    -------
    np.ndarray
        Smoothed signal with the same length as `values`.
    """
    values = np.asarray(values, dtype=float)

    if target_rate_hz >= source_rate_hz:
        return values

    if smoothing_window_s is None:
        smoothing_window_s = 1.0 / target_rate_hz

    window_samples = int(round(smoothing_window_s * source_rate_hz))
    window_samples = max(window_samples, 1)

    # Prefer an odd-length window for symmetric smoothing around each sample.
    if window_samples % 2 == 0:
        window_samples += 1

    return (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def interpolate_to_timebase(
    source_time_s: np.ndarray,
    source_values: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    """
    Interpolate one signal onto a target time base.

    NaNs are ignored during interpolation. Values outside the valid source range
    are returned as NaN.
    """
    source_time_s = np.asarray(source_time_s, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_time_s = np.asarray(target_time_s, dtype=float)

    valid = np.isfinite(source_time_s) & np.isfinite(source_values)

    if valid.sum() < 2:
        return np.full_like(target_time_s, np.nan, dtype=float)

    x = source_time_s[valid]
    y = source_values[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    out = np.interp(target_time_s, x, y)
    out[target_time_s < x[0]] = np.nan
    out[target_time_s > x[-1]] = np.nan

    return out


def resample_combined_to_common_rate(
    df: pd.DataFrame,
    target_rate_hz: float = 100.0,
    plif_source: Literal["native", "interpolated"] = "native",
    smoothing_window_s: float | None = None,
) -> pd.DataFrame:
    """
    Resample combined e-nose/PLIF data to a common sampling frequency.

    Parameters
    ----------
    df:
        Combined e-nose/PLIF DataFrame.
    target_rate_hz:
        Desired output rate, e.g. 100 Hz.
    plif_source:
        Source used for PLIF resampling.
    smoothing_window_s:
        Centred smoothing window used before downsampling high-rate channels.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with regular `time_s` and timedelta `timestamp`.
    """
    if plif_source not in {"native", "interpolated"}:
        raise ValueError("plif_source must be either 'native' or 'interpolated'.")

    df = df.sort_values("time_s").reset_index(drop=True)

    target_time_s = make_target_timebase(
        float(df["time_s"].min()),
        float(df["time_s"].max()),
        target_rate_hz,
    )

    out = pd.DataFrame({"time_s": target_time_s})
    out["timestamp"] = pd.to_timedelta(out["time_s"], unit="s")

    plif_cols = get_plif_columns(df)
    enose_cols = get_enose_columns(df)

    full_source_rate_hz = infer_native_rate_hz(df["time_s"])

    # E-nose and auxiliary numeric channels:
    for col in enose_cols:
        smoothed = zero_lag_smooth_for_downsampling(
            df[col].to_numpy(),
            source_rate_hz=full_source_rate_hz,
            target_rate_hz=target_rate_hz,
            smoothing_window_s=smoothing_window_s,
        )

        out[col] = interpolate_to_timebase(
            df["time_s"].to_numpy(),
            smoothed,
            target_time_s,
        )

    # PLIF channels:
    if plif_source == "native":
        native_df = select_plif_native_rows(df)
        plif_time = native_df["time_s"].to_numpy()
        plif_source_rate_hz = infer_native_rate_hz(plif_time)

        for col in plif_cols:
            values = native_df[col].to_numpy()
            values = zero_lag_smooth_for_downsampling(
                values,
                source_rate_hz=plif_source_rate_hz,
                target_rate_hz=target_rate_hz,
                smoothing_window_s=smoothing_window_s,
            )

            out[col] = interpolate_to_timebase(
                plif_time,
                values,
                target_time_s,
            )

    else:
        for col in plif_cols:
            values = zero_lag_smooth_for_downsampling(
                df[col].to_numpy(),
                source_rate_hz=full_source_rate_hz,
                target_rate_hz=target_rate_hz,
                smoothing_window_s=smoothing_window_s,
            )

            out[col] = interpolate_to_timebase(
                df["time_s"].to_numpy(),
                values,
                target_time_s,
            )

    out["sampling_strategy"] = "common_rate"
    out["target_rate_hz"] = target_rate_hz
    out["plif_source"] = plif_source

    return out


def load_combined_dataset(
    exp_id: str | int,
    strategy: SamplingStrategy = "interpolated",
    target_rate_hz: float = 100.0,
    plif_source: Literal["native", "interpolated"] = "native",
    smoothing_window_s: float | None = None,
) -> pd.DataFrame:
    """
    Load the combined e-nose/PLIF dataset using a selected sampling strategy.

    Parameters
    ----------
    strategy:
        - "interpolated": Return the full combined CSV as provided.
        - "plif_native": Return only rows corresponding to original PLIF images.
        - "common_rate": Return a regular, user-defined common time base.
    target_rate_hz:
        Used only for strategy="common_rate".
    plif_source:
        Used only for strategy="common_rate".
    smoothing_window_s:
        Optional centred zero-lag smoothing window for downsampling.
    """
    df = _load_raw_csv(exp_id)

    if strategy == "interpolated":
        out = df.copy()
        out["sampling_strategy"] = "interpolated"
        return out

    if strategy == "plif_native":
        out = select_plif_native_rows(df)
        out["sampling_strategy"] = "plif_native"
        return out

    if strategy == "common_rate":
        return resample_combined_to_common_rate(
            df,
            target_rate_hz=target_rate_hz,
            plif_source=plif_source,
            smoothing_window_s=smoothing_window_s,
        )

    raise ValueError(
        "strategy must be one of 'interpolated', 'plif_native', or 'common_rate'."
    )


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
# Backward-compatible aliases (for notebook compatibility)
# ---------------------------------------------------------------------

# Alias: infer_rate_hz -> infer_native_rate_hz
infer_rate_hz = infer_native_rate_hz

# Alias: inspect_combined_dataset -> inspect_combined_csv
inspect_combined_dataset = inspect_combined_csv
