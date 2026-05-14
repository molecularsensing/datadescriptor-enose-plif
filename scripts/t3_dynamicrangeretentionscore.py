#%%
"""Compute saturation percentage (dynamic range retention) per sensor."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.constants import LOCATIONS, SPEEDS
from utils.combined_io import load_combined_csv

DEFAULT_EXPERIMENTS = ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]


def _load_enose_plif(exp_id: str):
    """Load combined e-nose/PLIF data for an experiment id."""
    return load_combined_csv(f"r{exp_id}")


def compute_saturation(exp_ids: list[str] | None = None) -> pd.DataFrame:
    """Compute percent of non-zero samples per sensor for each experiment."""
    exp_ids = exp_ids or DEFAULT_EXPERIMENTS
    saturation = {exp_id: {f"R_gas_{i}": 0.0 for i in range(1, 9)} for exp_id in exp_ids}
    for exp_id in exp_ids:
        df = _load_enose_plif(exp_id)
        total = df.shape[0]
        for idx in range(1, 9):
            zero_count = df[df[f"R_gas_{idx}"] == 0].shape[0]
            saturation[exp_id][f"R_gas_{idx}"] = np.round(100 - 100 * zero_count / total, 1)

    df_sat = pd.DataFrame.from_dict(saturation, orient="index")
    df_sat["loc."] = [LOCATIONS[f"r{idx}"][1:] for idx in df_sat.index]
    df_sat["vel. (m/s)"] = [SPEEDS[f"r{idx}"] for idx in df_sat.index]

    cols = df_sat.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_sat = df_sat[cols]
    df_sat = df_sat.rename(columns={f"R_gas_{i}": f"Sensor {i}" for i in range(1, 9)})
    return df_sat


def save_saturation_table(df_sat: pd.DataFrame) -> None:
    """Save saturation table as percentages to the figs directory."""
    out_dir = Path(__file__).resolve().parent.parent.joinpath("figs")
    out_dir.mkdir(exist_ok=True)
    df_percent = df_sat.applymap(lambda x: f"{x}%")
    df_percent.to_csv(out_dir.joinpath("saturation.csv"), index=True)
    print(df_percent)


if __name__ == "__main__":
    save_saturation_table(compute_saturation())
