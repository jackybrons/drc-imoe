from pathlib import Path

import numpy as np
import pandas as pd

from datasets.preprocessing.prepare_tpsl_datasets import (
    NUM_SAMPLES,
    OUTPUT_COLUMNS,
    build_combined_data,
    validate_combined_data,
    write_csv_atomically,
)


def workbook(path: Path, cycles: range) -> None:
    rows = []
    for cycle in cycles:
        for voltage, capacity in [(3.5, 0.0), (3.6, 0.1), (4.15, 2.1), (4.2, 2.2)]:
            rows.append(
                {
                    "Cycle_Index": cycle,
                    "Current(A)": 1.24,
                    "Voltage(V)": voltage,
                    "Capacity(Ah)": capacity,
                }
            )
        rows.append(
            {
                "Cycle_Index": cycle,
                "Current(A)": -2.36,
                "Voltage(V)": 3.2,
                "Capacity(Ah)": 2.0,
            }
        )
    pd.DataFrame(rows).to_excel(path, index=False)


def test_build_combined_data_preserves_notebook_order_and_values(tmp_path):
    first20 = tmp_path / "first20cycle.xlsx"
    continuation = tmp_path / "cycles.xlsx"
    workbook(first20, range(1, 21))
    workbook(continuation, range(21, 69))

    result = build_combined_data(first20, continuation)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["Cycle_Index"].tolist() == list(range(1, 69))
    assert result.loc[0, "Charge_Current(A)"] == 1.2
    assert result.loc[0, "Discharge_Current(A)"] == -2.4
    assert len(result.loc[0, "QV_Curve"]) == NUM_SAMPLES
    assert np.allclose(result.loc[0, "QV_Curve"], np.linspace(0.1, 2.1, 50))
    validate_combined_data(result, "synthetic")


def test_atomic_writer_skips_existing_file_by_default(tmp_path):
    first20 = tmp_path / "first20cycle.xlsx"
    continuation = tmp_path / "cycles.xlsx"
    workbook(first20, range(1, 21))
    workbook(continuation, range(21, 69))
    data = build_combined_data(first20, continuation)
    output = tmp_path / "combined_data.csv"

    assert write_csv_atomically(data, output, overwrite=False) == "written"
    original = output.read_bytes()
    assert write_csv_atomically(data.iloc[::-1], output, overwrite=False) == "skipped"
    assert output.read_bytes() == original
