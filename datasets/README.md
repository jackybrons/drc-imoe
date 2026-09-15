# Datasets

This project contains standardized multi-condition, multi-chemistry datasets for degradation forecasting of retired lithium-ion batteries in second-life applications. All official preprocessed data is stored under `datasets/processed/`.

---

## 1. Directory Structure

```text
datasets/
├── processed/              # Official preprocessed-data root used by training and evaluation
│   ├── UL-NCA/             # UL laboratory NCA cell data (66 cells)
│   ├── UL-NCM/             # UL laboratory NCM cell data (55 cells)
│   ├── UL-NCMNCA/          # UL laboratory mixed-chemistry cell data (9 cells)
│   ├── TPSL-Arbitrary/     # TPSL cells under dynamic arbitrary charge/discharge conditions (53 cells)
│   ├── TPSL-Fixed/         # TPSL cells under fixed charge/discharge conditions (17 cells; cross-condition testing)
│   ├── TPSL-provenance.json# TPSL cell provenance and operating-condition metadata
│   └── LSD/                # Long-term degradation dataset (86 cells)
├── preprocessing/          # Raw-data download guide and preprocessing scripts
│   ├── DATA_DOWNLOAD.md    # Public dataset URLs and literature DOIs
│   ├── prepare_ul_datasets.py   # UL dataset cleaning and conversion
│   ├── prepare_tpsl_datasets.py # TPSL dataset cleaning and conversion
│   └── LSD-data-generate.ipynb  # LSD dataset cleaning and conversion notebook
└── loader.py               # Fixed battery-level splits, feature construction, and DataLoader implementations
```

---

## 2. Dataset Specifications and Battery-Level Splits

The project uses **fixed battery-level splits**. All degradation cycles from a given cell belong to exactly one subset, preventing temporal information leakage between training and test sets.

| Dataset | Chemistry / operating condition | Total cells | Train | Validation | Test | DataLoader class |
|---|---|---|---|---|---|---|
| **UL-NCA** | NCA 18650, CY25-025_1 (25°C, 0.25C) | 66 | 3 | 1 | 62 | `BatteryDataset` |
| **UL-NCM** | NCM 18650, CY25-05_1 (25°C, 0.5C) | 55 | 3 | 1 | 51 | `BatteryDataset` |
| **UL-NCMNCA** | Mixed NCM+NCA, CY25-05_1 | 9 | 3 | 1 | 5 | `BatteryDataset` |
| **TPSL (Arbitrary)** | Arbitrary dynamic conditions with changing current and temperature | 53 | 38 | 4 | 11 | `BatteryDataset1` |
| **TPSL (Fixed)** | Fixed-rate conditions for cross-condition transfer evaluation | 17 | - | - | 17 (target domain) | `BatteryDataset1` |
| **LSD** | Long-term, multistage second-life degradation | 86 | 57 | 6 | 23 | `BatteryDataset2` |

---

## 3. Feature Representation and Input Specification

All data loaders extract and construct the multimodal degradation representations required by the dual-router architecture:

1. **Partial Charge Curve**:
   - Extract a readily available, fixed 50-step voltage-rise segment from each charge/discharge cycle as the local-shape input to the curve-aware router.
2. **Relaxation and State Statistical Descriptors**:
   - Extract four global statistical features, including voltage-relaxation variance, skewness, and kurtosis, as input to the statistical router.
3. **Future Operating Conditions**:
   - Use the prescribed current and ambient temperature over the future prediction window as conditioning variables for the downstream condition-aware bidirectional recurrent network.
4. **Prediction Target**:
   - Predict the cell discharge-capacity degradation sequence over the specified horizon. The default is `pred_len=50`, with `10` and `25` used in ablation experiments.

---

## 4. Data-Efficiency Protocol (`--train_battery_count`)

To avoid duplicate or inconsistent sampling caused by percentage rounding in low-data experiments, the project uses a deterministic nested battery-sampling protocol:

- Specify the number of training cells with `--train_battery_count N`.
- Shuffle with a fixed seed and select the first $N$ cells, ensuring that each smaller training set is strictly nested within the larger sets.
- Record the selected cell IDs and effective training-cell count in the result's `metrics.json` for subset verification.
