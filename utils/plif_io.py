"""
PLIF HDF5 I/O helpers.

This module provides utilities for loading, inspecting, and exporting raw PLIF
HDF5 data. It centralises common patterns shared across analysis scripts:

- locating PLIF HDF5 files from experiment IDs or location keys;
- inspecting dataset shape, dtype, HDF5 chunking, and calibration metadata;
- loading single frames, frame ranges, or selected frame indices;
- temporal and spatial subsampling;
- chunk-wise iteration over long recordings;
- exporting selected PLIF fields to NumPy arrays or xarray datasets.
"""

from __future__ import annotations

import sys
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Import constants from utils.constants
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from utils.constants import SSD_PLIF_FOLDER, PLIF_REFS
except ImportError:
    # Fallback if run standalone
    SSD_PLIF_FOLDER = Path(__file__).resolve().parent.parent / "data" / "plif"
    PLIF_REFS: dict[str, str] = {}


# ---------------------------------------------------------------------
# Path and dataset helpers
# ---------------------------------------------------------------------

def resolve_exp_id(exp_or_loc: str) -> str:
    """
    Resolve either an experiment ID, e.g. 'r56', or a PLIF location key, e.g. 'LC',
    to the corresponding experiment ID.

    Parameters
    ----------
    exp_or_loc:
        Experiment ID or key in PLIF_REFS.

    Returns
    -------
    str
        Lowercase experiment ID.
    """
    if exp_or_loc in PLIF_REFS:
        return PLIF_REFS[exp_or_loc].lower()
    return exp_or_loc.lower()


def get_plif_h5_path(exp_or_loc: str, plif_root: Path = SSD_PLIF_FOLDER) -> Path:
    """
    Return the expected HDF5 path for a PLIF recording.

    Parameters
    ----------
    exp_or_loc:
        Experiment ID or PLIF location key.
    plif_root:
        Root PLIF folder. Defaults to SSD_PLIF_FOLDER from utils.constants.

    Returns
    -------
    Path
        Path to the HDF5 file.
    """
    exp_id = resolve_exp_id(exp_or_loc)
    return plif_root.joinpath("h5", f"{exp_id}.h5")


def get_plif_dataset_name(exp_id: str) -> str:
    """
    Return the conventional PLIF final-data dataset name for an experiment.

    Example
    -------
    r56 -> R56_FinalData
    """
    return f"{exp_id.upper()}_FinalData"


def open_plif_dataset(
    exp_or_loc: str,
    plif_root: Path = SSD_PLIF_FOLDER,
) -> tuple[h5py.File, h5py.Dataset]:
    """
    Open a PLIF HDF5 file and return the file handle and main dataset.

    Notes
    -----
    The caller is responsible for closing the returned HDF5 file handle.

    Example
    -------
    h5f, data = open_plif_dataset("r56")
    try:
        print(data.shape)
    finally:
        h5f.close()
    """
    exp_id = resolve_exp_id(exp_or_loc)
    h5_path = get_plif_h5_path(exp_id, plif_root=plif_root)

    if not h5_path.exists():
        raise FileNotFoundError(f"PLIF HDF5 file not found: {h5_path}")

    h5f = h5py.File(h5_path, "r")
    dataset_name = get_plif_dataset_name(exp_id)

    try:
        data = h5f["DataSets"][dataset_name]
    except KeyError as exc:
        h5f.close()
        raise KeyError(
            f"Could not find dataset 'DataSets/{dataset_name}' in {h5_path}"
        ) from exc

    return h5f, data


# ---------------------------------------------------------------------
# Metadata inspection
# ---------------------------------------------------------------------

def _decode_h5_value(value: Any) -> Any:
    """Decode common HDF5 values into plain Python objects where possible."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode(errors="replace")

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_h5_value(value.item())

        if value.dtype.kind == "S":
            return [_decode_h5_value(v) for v in value]

        if value.dtype.kind == "O":
            return [_decode_h5_value(v) for v in value]

        return value.tolist()

    return value


def _read_attrs(obj: h5py.Group | h5py.Dataset) -> dict[str, object]:
    """Read HDF5 attributes into a plain dictionary."""
    attrs: dict[str, object] = {}
    for key, value in obj.attrs.items():
        attrs[key] = _decode_h5_value(value)
    return attrs


def read_h5_object(
    obj: h5py.Group | h5py.Dataset,
    *,
    max_array_size: int = 64,
) -> dict[str, object] | object:
    """
    Recursively read an HDF5 object into Python-native metadata.

    Small datasets are loaded. Large datasets are represented by shape and dtype
    to avoid accidentally loading large PLIF arrays into memory.
    """
    if isinstance(obj, h5py.Dataset):
        out: dict[str, object] = {
            "type": "dataset",
            "shape": obj.shape,
            "dtype": str(obj.dtype),
        }

        attrs = _read_attrs(obj)
        if attrs:
            out["attrs"] = attrs

        if obj.shape == () or obj.size <= max_array_size:
            try:
                out["value"] = _decode_h5_value(obj[()])
            except Exception as exc:
                out["value_error"] = repr(exc)

        return out

    if isinstance(obj, h5py.Group):
        out: dict[str, object] = {"type": "group"}

        attrs = _read_attrs(obj)
        if attrs:
            out["attrs"] = attrs

        children: dict[str, Any] = {}
        for key in obj.keys():
            try:
                child = obj[key]
            except Exception as exc:
                children[key] = {"type": "unreadable", "error": repr(exc)}
                continue

            if isinstance(child, h5py.Group):
                children[key] = read_h5_object(child, max_array_size=max_array_size)
            elif isinstance(child, h5py.Dataset):
                children[key] = read_h5_object(child, max_array_size=max_array_size)
            else:
                children[key] = {"type": type(child).__name__, "repr": repr(child)}

        if children:
            out["children"] = children

        return out

    return {"type": type(obj).__name__, "repr": repr(obj)}


def read_h5_group_as_dict(group: h5py.Group) -> dict[str, Any]:
    """
    Recursively read small HDF5 metadata groups into nested dictionaries.

    Large arrays are not copied; they are represented by shape and dtype.
    """
    out: dict[str, Any] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            if item.shape == () or item.size <= 32:
                out[key] = _decode_h5_value(item[()])
            else:
                out[key] = {"shape": item.shape, "dtype": str(item.dtype)}
        elif isinstance(item, h5py.Group):
            out[key] = read_h5_group_as_dict(item)
    return out


def inspect_plif_file(exp_or_loc: str, verbose: bool = True) -> dict[str, object]:
    """
    Inspect a raw PLIF HDF5 file without loading the full recording.
    """
    exp_id = resolve_exp_id(exp_or_loc)
    h5_path = get_plif_h5_path(exp_id)

    with h5py.File(h5_path, "r") as h5f:
        dataset_name = get_plif_dataset_name(exp_id)
        data = h5f["DataSets"][dataset_name]

        info: dict[str, object] = {
            "experiment_id": exp_id,
            "path": str(h5_path),
            "dataset": f"DataSets/{dataset_name}",
            "shape": data.shape,
            "dtype": str(data.dtype),
            "chunks": data.chunks,
            "compression": data.compression,
            "compression_opts": data.compression_opts,
            "n_frames": data.shape[0],
            "frame_shape_yx": data.shape[1:],
            "metadata": {},
        }

        for key in h5f.keys():
            if key == "DataSets":
                continue
            obj = h5f[key]
            info["metadata"][key] = read_h5_object(obj)

    if verbose:
        print_plif_info(info)

    return info


def _format_scalar(value: object, max_len: int = 120) -> str:
    """Format scalar metadata values compactly for terminal output."""
    text = str(value)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _print_attrs(attrs: dict[str, object], indent: int = 2) -> None:
    """Pretty-print HDF5 attributes."""
    pad = " " * indent
    for key, value in attrs.items():
        print(f"{pad}@{key}: {_format_scalar(value)}")


def _print_h5_metadata_node(
    name: str,
    node: dict[str, object],
    indent: int = 0,
    max_depth: int = 4,
) -> None:
    """
    Pretty-print a metadata node produced by read_h5_object().
    """
    pad = " " * indent
    node_type = node.get("type", "unknown")

    if node_type == "group":
        print(f"{pad}{name}/")
        attrs = node.get("attrs", {})
        if attrs:
            _print_attrs(attrs, indent=indent + 2)
        if max_depth <= 0:
            children = node.get("children", {})
            if children:
                print(f"{pad}  ...")
            return
        children = node.get("children", {})
        for child_name, child_node in children.items():
            _print_h5_metadata_node(
                child_name, child_node, indent=indent + 2, max_depth=max_depth - 1
            )
    elif node_type == "dataset":
        shape = node.get("shape")
        dtype = node.get("dtype")
        value = node.get("value", None)
        attrs = node.get("attrs", {})
        if value is not None:
            print(f"{pad}{name}: {_format_scalar(value)}")
        else:
            print(f"{pad}{name}: dataset shape={shape}, dtype={dtype}")
        if attrs:
            _print_attrs(attrs, indent=indent + 2)
    else:
        print(f"{pad}{name}: {node_type}")


def print_plif_info(info: dict[str, object]) -> None:
    """Pretty-print the most important PLIF file information."""
    print("\nPLIF HDF5 file")
    print("-" * 60)
    print(f"Experiment ID:      {info['experiment_id']}")
    print(f"Path:               {info['path']}")
    print(f"Dataset:            {info['dataset']}")
    print(f"Shape:              {info['shape']}")
    print(f"Frame shape (y, x): {info['frame_shape_yx']}")
    print(f"Number of frames:   {info['n_frames']}")
    print(f"Dtype:              {info['dtype']}")
    print(f"HDF5 chunks:         {info['chunks']}")
    print(f"Compression:         {info['compression']}")

    metadata = info.get("metadata", {})
    if metadata:
        print("\nMetadata")
        print("-" * 60)
        for key, node in metadata.items():
            _print_h5_metadata_node(key, node, indent=0)
    print()


def load_px_per_mm(exp_or_loc: str) -> float | None:
    """
    Load the pixel-to-mm calibration factor from the HDF5 Mapping group.

    Returns
    -------
    float | None
        Pixels per mm if available, otherwise None.
    """
    exp_id = resolve_exp_id(exp_or_loc)
    h5_path = get_plif_h5_path(exp_id)

    with h5py.File(h5_path, "r") as h5f:
        try:
            raw = h5f["Mapping"]["scaling: pixels to mm"][()]
        except KeyError:
            return None

    decoded = _decode_h5_value(raw)
    if isinstance(decoded, str):
        return float(decoded.split()[0])
    return float(decoded)


# ---------------------------------------------------------------------
# Selective PLIF loading
# ---------------------------------------------------------------------

def load_plif_frame(exp_or_loc: str, frame_idx: int) -> np.ndarray:
    """
    Load a single PLIF frame.

    This is the most memory-efficient way to inspect an individual frame.
    """
    h5f, data = open_plif_dataset(exp_or_loc)
    try:
        if frame_idx < 0:
            frame_idx = data.shape[0] + frame_idx
        if not 0 <= frame_idx < data.shape[0]:
            raise IndexError(f"frame_idx {frame_idx} outside valid range 0:{data.shape[0] - 1}")
        return data[frame_idx, :, :]
    finally:
        h5f.close()


def load_plif_range(
    exp_or_loc: str,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> np.ndarray:
    """
    Load a contiguous frame range, optionally with temporal and spatial subsampling.

    Parameters
    ----------
    exp_or_loc:
        Experiment ID or PLIF location key.
    start, stop, step:
        Frame selection. Equivalent to data[start:stop:step].
    y_slice, x_slice:
        Optional spatial slices. For example, slice(100, 500, 2).

    Returns
    -------
    np.ndarray
        Array with shape (time, y, x).
    """
    y_slice = y_slice if y_slice is not None else slice(None)
    x_slice = x_slice if x_slice is not None else slice(None)

    h5f, data = open_plif_dataset(exp_or_loc)
    try:
        stop = data.shape[0] if stop is None else stop
        return data[start:stop:step, y_slice, x_slice]
    finally:
        h5f.close()


def load_plif_frames(
    exp_or_loc: str,
    frame_indices: Iterable[int],
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> np.ndarray:
    """
    Load an arbitrary set of frame indices.
    """
    y_slice = y_slice if y_slice is not None else slice(None)
    x_slice = x_slice if x_slice is not None else slice(None)
    frame_indices = list(frame_indices)

    h5f, data = open_plif_dataset(exp_or_loc)
    try:
        frames = [data[idx, y_slice, x_slice] for idx in frame_indices]
        return np.stack(frames, axis=0)
    finally:
        h5f.close()


def iter_plif_chunks(
    exp_or_loc: str,
    start: int = 0,
    stop: int | None = None,
    chunk_size: int = 256,
    step: int = 1,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> Generator[tuple[slice, np.ndarray], None, None]:
    """
    Iterate over a long PLIF recording in temporal chunks.

    This is the recommended pattern for processing long raw recordings on
    standard hardware.

    Yields
    ------
    tuple[slice, np.ndarray]
        A frame slice and the corresponding PLIF array.
    """
    y_slice = y_slice if y_slice is not None else slice(None)
    x_slice = x_slice if x_slice is not None else slice(None)

    h5f, data = open_plif_dataset(exp_or_loc)

    try:
        n_frames = data.shape[0]
        stop = n_frames if stop is None else min(stop, n_frames)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        for chunk_start in range(start, stop, chunk_size * step):
            chunk_stop = min(chunk_start + chunk_size * step, stop)
            frame_slice = slice(chunk_start, chunk_stop, step)
            block = data[frame_slice, y_slice, x_slice]
            yield frame_slice, block

    finally:
        h5f.close()


def compute_temporal_mean(
    exp_or_loc: str,
    start: int = 0,
    stop: int | None = None,
    chunk_size: int = 256,
    step: int = 1,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> np.ndarray:
    """
    Compute the temporal mean PLIF field using chunked loading.

    This avoids loading the full movie into memory.
    """
    total: np.ndarray | None = None
    n_total = 0

    for _, block in iter_plif_chunks(
        exp_or_loc,
        start=start,
        stop=stop,
        chunk_size=chunk_size,
        step=step,
        y_slice=y_slice,
        x_slice=x_slice,
    ):
        block = np.asarray(block, dtype=np.float64)
        if total is None:
            total = np.zeros(block.shape[1:], dtype=np.float64)
        total += np.nansum(block, axis=0)
        n_total += block.shape[0]

    if total is None or n_total == 0:
        raise ValueError("No frames selected.")

    return total / n_total


def compute_temporal_max(
    exp_or_loc: str,
    start: int = 0,
    stop: int | None = None,
    chunk_size: int = 256,
    step: int = 1,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> np.ndarray:
    """
    Compute the temporal maximum PLIF field using chunked loading.
    """
    maximum: np.ndarray | None = None

    for _, block in iter_plif_chunks(
        exp_or_loc,
        start=start,
        stop=stop,
        chunk_size=chunk_size,
        step=step,
        y_slice=y_slice,
        x_slice=x_slice,
    ):
        block_max = np.nanmax(block, axis=0)
        if maximum is None:
            maximum = block_max
        else:
            maximum = np.maximum(maximum, block_max)

    if maximum is None:
        raise ValueError("No frames selected.")

    return maximum


def export_numpy(
    exp_or_loc: str,
    out_path: Path,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
    compressed: bool = True,
) -> Path:
    """
    Export a selected PLIF subset to a NumPy .npy or compressed .npz file.
    """
    arr = load_plif_range(
        exp_or_loc,
        start=start,
        stop=stop,
        step=step,
        y_slice=y_slice,
        x_slice=x_slice,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")
        np.savez_compressed(out_path, plif=arr)
    else:
        if out_path.suffix != ".npy":
            out_path = out_path.with_suffix(".npy")
        np.save(out_path, arr)

    print(f"Exported PLIF subset: {out_path}")
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")

    return out_path