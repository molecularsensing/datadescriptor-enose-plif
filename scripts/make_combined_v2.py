#%%
"""Convert the combined e-nose/PLIF CSVs from layout v1 to layout v2.

Layout v2 keeps every v1 value and separates measured from interpolated PLIF:

- all e-nose, heater and environment columns: unchanged (1 kHz);
- ``plif_A`` .. ``plif_D``: measured PLIF only. A value on the rows where a
  PLIF image was acquired (``plif_image`` is true), empty otherwise;
- ``plif_A_interp`` .. ``plif_D_interp``: the v1 ``plif_A`` .. ``plif_D``
  columns, i.e. the measured values interpolated (PCHIP) to the e-nose
  timestamps;
- ``R_gas_valid_1`` .. ``R_gas_valid_8``: false where the sensor reading is
  saturated (``R_gas`` is exactly 0), true otherwise.

Fields are copied as text, so unchanged columns are identical to v1.

Usage
-----
python scripts/make_combined_v2.py --in-dir <v1 folder> --out-dir <v2 folder>
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.combined_io import get_combined_csv_path

DEFAULT_EXPERIMENTS = ["56", "57", "58", "59", "60", "61", "70", "71", "75", "76"]
ZONES = ["A", "B", "C", "D"]
N_SENSORS = 8


def convert_file(in_path: Path, out_path: Path) -> dict[str, int]:
    """Convert one combined CSV from layout v1 to v2 and return row counts."""
    n_rows = 0
    n_frames = 0
    with open(in_path, newline="") as f_in, open(out_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out, lineterminator="\n")

        header = next(reader)
        if any(col.endswith("_interp") for col in header):
            raise ValueError(f"{in_path} already has layout v2.")

        idx = {name: i for i, name in enumerate(header)}
        plif_idx = [idx[f"plif_{zone}"] for zone in ZONES]
        r_gas_idx = [idx[f"R_gas_{i}"] for i in range(1, N_SENSORS + 1)]
        image_idx = idx["plif_image"]

        writer.writerow(
            header
            + [f"plif_{zone}_interp" for zone in ZONES]
            + [f"R_gas_valid_{i}" for i in range(1, N_SENSORS + 1)]
        )

        for row in reader:
            interp = [row[i] for i in plif_idx]
            valid = ["false" if float(row[i]) == 0.0 else "true" for i in r_gas_idx]

            if row[image_idx] == "true":
                n_frames += 1
            elif row[image_idx] == "false":
                for i in plif_idx:
                    row[i] = ""
            else:
                raise ValueError(f"Unexpected plif_image value: {row[image_idx]!r}")

            writer.writerow(row + interp + valid)
            n_rows += 1

    return {"rows": n_rows, "plif_frames": n_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, required=True, help="Folder with the v1 CSVs.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Folder for the v2 CSVs.")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args()

    if args.in_dir.resolve() == args.out_dir.resolve():
        raise SystemExit("--out-dir must differ from --in-dir; the v1 files are never overwritten.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for exp_id in args.experiments:
        in_path = get_combined_csv_path(exp_id, combined_root=args.in_dir)
        out_path = get_combined_csv_path(exp_id, combined_root=args.out_dir)
        counts = convert_file(in_path, out_path)
        print(f"r{exp_id}: {counts['rows']} rows, {counts['plif_frames']} PLIF frames -> {out_path}")


if __name__ == "__main__":
    main()
