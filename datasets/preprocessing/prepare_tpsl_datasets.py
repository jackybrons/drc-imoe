"""Prepare the TPSL cells used by the current train/validation/test loader."""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "Cycle_Index",
    "QV_Curve",
    "Charge_Current(A)",
    "Temperture",
    "Discharge_Current(A)",
    "Max_Charge_Capacity(Ah)",
    "Max_Discharge_Capacity(Ah)",
]
VOLTAGE_RANGE = (3.6, 4.15)
NUM_SAMPLES = 50
WINDOW_SIZE = 50
WINDOW_COUNT = 19

SPLITS = {
    "ARBITRARY": {
        "train": [1, 3, 7, 9, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28,
                  30, 31, 34, 36, 37, 39, 40, 42, 46, 47, 50, 54, 55, 56,
                  59, 60, 74, 75, 76, 77, 67, 68, 69, 73],
        "val": [66, 70],
        "test": [5, 8, 11, 12, 71, 72, 33, 43, 61, 62, 63, 64, 65],
    },
    "FIXED": {
        "train": [6, 22, 26, 29, 32, 38, 41, 44, 49, 52, 53],
        "val": [45, 58],
        "test": [23, 35, 48, 57],
    },
}
CATEGORY_FOLDERS = {
    "ARBITRARY": "Cycled with Arbitrary Uses Profiles",
    "FIXED": "Cycled with Fixed Current Profiles",
}
OUTPUT_FOLDERS = {
    "ARBITRARY": "TPSL-Arbitrary",
    "FIXED": "TPSL-Fixed",
}
EXCLUSIONS = {
    "complete_but_not_used_by_loader": ["ARBITRARY:#4", "ARBITRARY:#51"],
    "incomplete_source_pair": ["FIXED:#2 (missing continuation workbook)"],
    "paper_cells_missing_from_source": ["FIXED:#10", "FIXED:#13", "FIXED:#16", "FIXED:#19"],
}


def loader_cells(condition: str):
    for split, cell_ids in SPLITS[condition].items():
        for cell_id in cell_ids:
            yield split, cell_id


def find_source_pair(source_root: Path, condition: str, cell_id: int) -> tuple[Path, Path]:
    cell_dir = source_root / CATEGORY_FOLDERS[condition] / f"#{cell_id}"
    workbooks = sorted(cell_dir.glob("*.xlsx"), key=lambda path: path.name)
    first20 = [path for path in workbooks if "first20cycle" in path.name]
    continuation = [path for path in workbooks if "first20cycle" not in path.name]
    if len(first20) != 1 or len(continuation) != 1:
        raise ValueError(
            f"{condition} #{cell_id} requires one first20cycle and one continuation "
            f"workbook; found first20={len(first20)}, continuation={len(continuation)}"
        )
    return first20[0], continuation[0]


def extract_workbook(workbook: Path) -> list[dict]:
    data = pd.read_excel(workbook)
    rows = []
    for cycle in np.unique(data["Cycle_Index"].values):
        cycle_data = data[data["Cycle_Index"] == cycle]
        charge_data = cycle_data[cycle_data["Current(A)"] > 0]
        discharge_data = cycle_data[cycle_data["Current(A)"] < 0]

        voltage = charge_data["Voltage(V)"].values
        capacity = charge_data["Capacity(Ah)"].values
        mask = (voltage >= VOLTAGE_RANGE[0]) & (voltage <= VOLTAGE_RANGE[1])
        voltage = voltage[mask]
        capacity = capacity[mask]
        if len(voltage) > 0:
            sampled_voltage = np.linspace(voltage.min(), voltage.max(), NUM_SAMPLES)
            qv_curve = np.interp(sampled_voltage, voltage, capacity)
        else:
            qv_curve = np.full(NUM_SAMPLES, np.nan)

        charge_currents = charge_data["Current(A)"].values
        discharge_currents = discharge_data["Current(A)"].values
        rows.append(
            {
                "Cycle_Index": cycle,
                "QV_Curve": qv_curve.tolist(),
                "Charge_Current(A)": (
                    np.round(charge_currents[0], 1) if len(charge_currents) else None
                ),
                "Temperture": 25,
                "Discharge_Current(A)": (
                    np.round(discharge_currents[0], 1) if len(discharge_currents) else None
                ),
                "Max_Charge_Capacity(Ah)": charge_data["Capacity(Ah)"].max(),
                "Max_Discharge_Capacity(Ah)": discharge_data["Capacity(Ah)"].max(),
            }
        )
    return rows


def build_combined_data(first20: Path, continuation: Path) -> pd.DataFrame:
    combined = extract_workbook(first20) + extract_workbook(continuation)
    cleaned = []
    for row in combined:
        qv_has_nan = any(pd.isna(value) for value in row["QV_Curve"])
        other_has_nan = any(
            pd.isna(value) or value is None
            for key, value in row.items()
            if key != "QV_Curve"
        )
        if not qv_has_nan and not other_has_nan:
            cleaned.append(row)
    return pd.DataFrame(cleaned, columns=OUTPUT_COLUMNS)


def parse_qv(value) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def validate_combined_data(data: pd.DataFrame, label: str) -> None:
    if list(data.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{label}: columns do not match the notebook schema")
    minimum_rows = WINDOW_SIZE + WINDOW_COUNT - 1
    if len(data) < minimum_rows:
        raise ValueError(f"{label}: {len(data)} rows cannot provide 19 windows of length 50")
    for row_number, value in enumerate(data["QV_Curve"]):
        curve = parse_qv(value)
        if len(curve) != NUM_SAMPLES or not np.isfinite(curve).all():
            raise ValueError(f"{label}: invalid QV_Curve at row {row_number}")
    numeric = data.drop(columns="QV_Curve").apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite scalar field")


def write_csv_atomically(data: pd.DataFrame, output_path: Path, overwrite: bool) -> str:
    if output_path.exists() and not overwrite:
        return "skipped"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.", suffix=".tmp", dir=output_path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        data.to_csv(temporary_path, index=False)
        validate_combined_data(pd.read_csv(temporary_path), str(output_path))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "written"


def build_manifest(source_root: Path, output_root: Path) -> dict:
    mappings = []
    for condition in SPLITS:
        for split, cell_id in loader_cells(condition):
            first20, continuation = find_source_pair(source_root, condition, cell_id)
            mappings.append(
                {
                    "condition": condition,
                    "split": split,
                    "cell_id": f"#{cell_id}",
                    "first20_workbook": first20.relative_to(source_root).as_posix(),
                    "continuation_workbook": continuation.relative_to(source_root).as_posix(),
                    "output": (
                        Path(OUTPUT_FOLDERS[condition]) / f"#{cell_id}" / "combined_data.csv"
                    ).as_posix(),
                }
            )
    return {
        "schema_version": 1,
        "loader_cell_count": len(mappings),
        "counts": {
            "ARBITRARY": sum(item["condition"] == "ARBITRARY" for item in mappings),
            "FIXED": sum(item["condition"] == "FIXED" for item in mappings),
        },
        "notebook_transform": {
            "voltage_range": list(VOLTAGE_RANGE),
            "qv_points": NUM_SAMPLES,
            "source_order": ["first20_workbook", "continuation_workbook"],
            "output_columns": OUTPUT_COLUMNS,
        },
        "exclusions": EXCLUSIONS,
        "mappings": mappings,
    }


def write_manifest_atomically(manifest: dict, output_root: Path) -> Path:
    path = output_root / "TPSL-provenance.json"
    output_root.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".TPSL-provenance.", suffix=".tmp", dir=output_root, delete=False
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        json.loads(temporary_path.read_text(encoding="utf-8"))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="../../DATASET/TPSL")
    parser.add_argument("--output-root", default="dataset")
    parser.add_argument(
        "--condition", type=str.upper, choices=["ALL", *SPLITS], default="ALL"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = project_path(args.source_root)
    output_root = project_path(args.output_root)
    manifest = build_manifest(source_root, output_root)
    conditions = SPLITS if args.condition == "ALL" else [args.condition]
    written = 0
    skipped = 0
    for condition in conditions:
        for _, cell_id in loader_cells(condition):
            output_path = output_root / OUTPUT_FOLDERS[condition] / f"#{cell_id}" / "combined_data.csv"
            if output_path.exists() and not args.overwrite:
                print(f"SKIP {condition}: #{cell_id}")
                skipped += 1
                continue
            first20, continuation = find_source_pair(source_root, condition, cell_id)
            data = build_combined_data(first20, continuation)
            validate_combined_data(data, f"{condition} #{cell_id}")
            status = write_csv_atomically(data, output_path, args.overwrite)
            print(f"WRITE {condition}: #{cell_id} rows={len(data)}")
            written += status == "written"
    manifest_path = write_manifest_atomically(manifest, output_root)
    print(f"MANIFEST {manifest_path}")
    print(f"DONE written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
