"""Prepare the UL-NCM and UL-NCMNCA datasets from the published raw CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VOLTAGE_RANGE = (3.6, 4.15)
NUM_POINTS = 50
OUTPUT_COLUMNS = [
    "Cycle",
    "Charge_Current",
    "Discharge_Current",
    "Temperature",
    "Capacity_Increment",
    "Relaxation_Voltage",
    "Discharge_Capacity",
]
RAW_COLUMNS = [
    "control/V/mA",
    "Ecell/V",
    "Q discharge/mA.h",
    "Q charge/mA.h",
    "control/mA",
    "cycle number",
]
CHEMISTRIES = {
    "NCM": ("Dataset_2_NCM_battery", "UL-NCM"),
    "NCMNCA": ("Dataset_3_NCM_NCA_battery", "UL-NCMNCA"),
}


def generate_dataframe(raw_path: Path) -> pd.DataFrame:
    """Apply the UL notebook transformation used by the checked-in examples."""
    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS)
    file_name = raw_path.name
    temperature = int(file_name[2:4])
    current_fields = file_name.split("-")[1].split("_")
    charge_current = float(current_fields[0]) / 10
    discharge_current = current_fields[1]
    cycle_life = len(np.unique(raw["cycle number"].values))

    rows = []
    for cycle in range(2, cycle_life + 1):
        cycle_data = raw[raw["cycle number"] == cycle]
        cc_data = cycle_data[
            (cycle_data["control/mA"] > 0)
            & cycle_data["Ecell/V"].between(*VOLTAGE_RANGE)
        ]
        voltage = cc_data["Ecell/V"].values
        capacity = cc_data["Q charge/mA.h"].values
        if len(voltage) < 2 or len(capacity) < 2:
            continue

        sample_voltage = np.linspace(*VOLTAGE_RANGE, NUM_POINTS)
        sample_capacity = np.interp(sample_voltage, voltage, capacity)
        capacity_increment = sample_capacity - sample_capacity[0]
        relaxation_voltage = cycle_data[
            (cycle_data["control/V/mA"] == 0) & (cycle_data["Ecell/V"] > 4)
        ]["Ecell/V"].tolist()
        rows.append(
            [
                cycle,
                charge_current,
                discharge_current,
                temperature,
                capacity_increment.tolist(),
                relaxation_voltage,
                cycle_data["Q discharge/mA.h"].max(),
            ]
        )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    to_drop = []
    for index in range(1, len(result)):
        capacity_change = abs(
            result.loc[index, "Discharge_Capacity"]
            - result.loc[index - 1, "Discharge_Capacity"]
        )
        if capacity_change >= 100:
            to_drop.append(index)
    return result.drop(to_drop)


def find_source_files(source_root: Path, chemistry: str) -> list[Path]:
    source_folder, _ = CHEMISTRIES[chemistry]
    chemistry_root = source_root / source_folder
    files = sorted(chemistry_root.rglob("*.csv"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {chemistry_root}")
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate CSV names found under {chemistry_root}")
    return files


def prepare_chemistry(
    source_root: Path,
    output_root: Path,
    chemistry: str,
    overwrite: bool = False,
) -> tuple[int, int]:
    source_files = find_source_files(source_root, chemistry)
    _, output_folder = CHEMISTRIES[chemistry]
    output_dir = output_root / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    for source_path in source_files:
        output_path = output_dir / source_path.name
        if output_path.exists() and not overwrite:
            print(f"SKIP {chemistry}: {source_path.name}")
            skipped += 1
            continue
        generate_dataframe(source_path).to_csv(output_path, index=False)
        print(f"WRITE {chemistry}: {source_path.name}")
        generated += 1
    return generated, skipped


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="../../DATASET/UL",
        help="UL raw-data root (relative paths are resolved from this script)",
    )
    parser.add_argument(
        "--output-root",
        default="dataset",
        help="Output dataset root (relative paths are resolved from this script)",
    )
    parser.add_argument(
        "--chemistry",
        type=str.upper,
        choices=["ALL", *CHEMISTRIES],
        default="ALL",
        help="Chemistry to prepare (default: ALL)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output CSVs (default: skip them)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = project_path(args.source_root)
    output_root = project_path(args.output_root)
    chemistries = CHEMISTRIES if args.chemistry == "ALL" else [args.chemistry]

    total_generated = 0
    total_skipped = 0
    for chemistry in chemistries:
        generated, skipped = prepare_chemistry(
            source_root, output_root, chemistry, args.overwrite
        )
        total_generated += generated
        total_skipped += skipped
    print(f"DONE generated={total_generated} skipped={total_skipped}")


if __name__ == "__main__":
    main()
