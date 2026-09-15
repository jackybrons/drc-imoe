from pathlib import Path

import numpy as np
import pandas as pd

from datasets.preprocessing.prepare_ul_datasets import (
    OUTPUT_COLUMNS,
    generate_dataframe,
    prepare_chemistry,
)


def write_raw_csv(path: Path) -> None:
    rows = []
    for cycle, discharge_capacity in [(1, 1000.0), (2, 990.0), (3, 800.0)]:
        for voltage, charge_capacity in [(3.6, 10.0), (4.15, 20.0)]:
            rows.append(
                {
                    "control/V/mA": 1,
                    "Ecell/V": voltage,
                    "Q discharge/mA.h": discharge_capacity,
                    "Q charge/mA.h": charge_capacity,
                    "control/mA": 1,
                    "cycle number": cycle,
                }
            )
        rows.append(
            {
                "control/V/mA": 0,
                "Ecell/V": 4.2,
                "Q discharge/mA.h": discharge_capacity,
                "Q charge/mA.h": 20.0,
                "control/mA": 0,
                "cycle number": cycle,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_dataframe_matches_notebook_transform(tmp_path):
    raw_path = tmp_path / "CY25-05_1-#1.csv"
    write_raw_csv(raw_path)

    result = generate_dataframe(raw_path)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["Cycle"].tolist() == [2]
    assert result.loc[0, "Charge_Current"] == 0.5
    assert result.loc[0, "Discharge_Current"] == "1"
    assert result.loc[0, "Temperature"] == 25
    assert len(result.loc[0, "Capacity_Increment"]) == 50
    assert np.allclose(result.loc[0, "Capacity_Increment"], np.linspace(0, 10, 50))
    assert result.loc[0, "Relaxation_Voltage"] == [4.2]
    assert result.loc[0, "Discharge_Capacity"] == 990.0


def test_prepare_chemistry_skips_existing_output_by_default(tmp_path):
    source_root = tmp_path / "source"
    source_dir = source_root / "Dataset_2_NCM_battery"
    source_dir.mkdir(parents=True)
    raw_path = source_dir / "CY25-05_1-#1.csv"
    write_raw_csv(raw_path)
    output_root = tmp_path / "output"

    generated, skipped = prepare_chemistry(source_root, output_root, "NCM")
    output_path = output_root / "UL-NCM" / raw_path.name
    original = output_path.read_bytes()
    generated_again, skipped_again = prepare_chemistry(source_root, output_root, "NCM")

    assert (generated, skipped) == (1, 0)
    assert (generated_again, skipped_again) == (0, 1)
    assert output_path.read_bytes() == original
