#%%
"""
Example utilities for inspecting and selectively loading raw PLIF HDF5 data.

This script illustrates typical data-handling workflows for the raw high-resolution
PLIF image sequences. All core PLIF I/O functions have been moved to utils.plif_io.

Example usage:
    python scripts/explore_raw_plif.py

Or import specific functions:
    from utils.plif_io import inspect_plif_file, load_plif_frame
"""

import sys
from pathlib import Path

# Optional dependency. The script still works without xarray unless export_xarray()
# is called.
try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plif_io import (
    compute_temporal_mean,
    compute_temporal_max,
    export_numpy,
    inspect_plif_file,
    load_plif_frame,
    load_plif_range,
    resolve_exp_id,
)

# Optional: export_xarray is not included in utils.plif_io exports yet.
# If you need it, import directly:
# from utils.plif_io import export_xarray


def example_workflow(exp_or_loc: str = "r56", export: bool = False) -> None:
    """
    Demonstrate typical raw PLIF data handling on a small selected subset.
    """
    # 1. Inspect metadata without reading the full image sequence.
    inspect_plif_file(exp_or_loc)

    # 2. Load and plot a single frame.
    frame = load_plif_frame(exp_or_loc, frame_idx=1600)
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

    # 3. Load every 20th frame from a limited time window.
    subset = load_plif_range(
        exp_or_loc,
        start=1000,
        stop=2000,
        step=20,
        y_slice=slice(None, None, 2),
        x_slice=slice(None, None, 2),
    )
    print(f"Subsampled subset shape: {subset.shape}")

    # 4. Process a long selection in chunks.
    mean_field = compute_temporal_mean(
        exp_or_loc,
        start=1000,
        stop=3000,
        step=5,
        chunk_size=100,
    )
    print(f"Temporal mean shape: {mean_field.shape}")

    # 5. Export a small subset to NumPy.
    out_dir = Path(__file__).resolve().parent.parent.joinpath("processed_examples")
    export_numpy(
        exp_or_loc,
        out_path=out_dir.joinpath(f"{resolve_exp_id(exp_or_loc)}_plif_subset.npz"),
        start=1000,
        stop=1200,
        step=10,
        y_slice=slice(None, None, 2),
        x_slice=slice(None, None, 2),
    )

    # 6. Export a small subset to xarray/NetCDF if xarray is installed.
    if export and xr is not None:
        from utils.plif_io import export_xarray
        export_xarray(
            exp_or_loc,
            out_path=out_dir.joinpath(f"{resolve_exp_id(exp_or_loc)}_plif_subset.nc"),
            start=1000,
            stop=1200,
            step=10,
            y_slice=slice(None, None, 2),
            x_slice=slice(None, None, 2),
        )


if __name__ == "__main__":
    example_workflow(exp_or_loc="r56", export=False)

# %%