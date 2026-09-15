# Reproducibility guide

This document describes how to reconstruct the data layout, run the locked
experimental protocol, and locate the evidence committed with the repository.
Commands are intended to be run from the repository root.

## 1. Environment

Use Python 3.11 and install the dependencies in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation before training:

```bash
python -m pytest -q
python -m experiments.main.experiment_matrix list
```

The first command should report 114 passed tests and 6 passed subtests. The
second prints the registered paper experiments and their acceptance criteria.

## 2. Data preparation

Download the UL, TPSL, and LSD datasets from the DOI links in the root README.
The expected input and output structure is:

```text
raw_data/
├── UL/
├── TPSL/
└── LSD/

datasets/processed/
├── UL-NCA/
├── UL-NCM/
├── UL-NCMNCA/
├── TPSL-Arbitrary/
├── TPSL-Fixed/
└── LSD/
```

Prepare UL and TPSL with:

```bash
python -m datasets.preprocessing.prepare_ul_datasets \
  --raw_dir raw_data/UL --output_dir datasets/processed

python -m datasets.preprocessing.prepare_tpsl_datasets \
  --raw_dir raw_data/TPSL --output_dir datasets/processed
```

Prepare LSD by running
`datasets/preprocessing/LSD-data-generate.ipynb`. Further source and layout
details are recorded in `datasets/preprocessing/DATA_DOWNLOAD.md`.

## 3. Locked protocol

| Item | Setting |
| --- | --- |
| Split unit | Battery cell |
| Model-selection split | Validation |
| Model-selection metric | Global prediction MSE |
| Formal seeds | 2025, 2026, 2027 |
| Maximum epochs | 1500 |
| Early-stopping patience | 250 |
| Training batch size | 32 |
| Evaluation batch size | 64 |

The test split is not used for checkpoint or hyperparameter selection. Data
efficiency runs use `--train_battery_count` to construct deterministic nested
subsets of training cells.

## 4. Running an experiment

The following command runs the main model on arbitrary-profile TPSL data:

```bash
python -m experiments.main.run \
  --model iMOE_SDR \
  --dataset TPSL \
  --condition Arbitrary \
  --seed 2025 \
  --train_epochs 1500 \
  --patience 250 \
  --batch_size 32 \
  --eval_batch_size 64
```

Available model names are `iMOE_SDR`, `iMOE`, `iMOE_CSR`, `ConditionedLSTM`,
`ConditionedMLP`, `PATCHTST`, `Informer`, and `DegradationFormer`.

Generate the complete command list without starting jobs:

```bash
python -m experiments.main.experiment_matrix formal-commands --model iMOE_SDR
```

Run the formal matrix on the selected GPU identifiers:

```bash
python -m experiments.main.experiment_matrix formal-run \
  --model iMOE_SDR --gpus 0 1
```

## 5. Generated and committed outputs

| Location | Status | Contents |
| --- | --- | --- |
| `results/runs/current/` | Retained locally, ignored | Complete outputs for the 306 current-matrix runs used by the paper and supplement |
| `checkpoints/main/` | Retained locally, ignored | The matching 306 selected model weights |
| `results/runs/local/` | Retained locally, ignored | Prediction arrays for the locked primary DRC-iMOE and full-fusion reference results |
| `checkpoints/imported/local/` | Retained locally, ignored | The 27 historical weights used by those locked results |
| `results/main/drc_imoe_legacy/` | Committed | Locked three-seed primary results and compression benchmarks |
| `results/main/dual_router_legacy/locked_multiseed/` | Committed | Locked full-fusion selection evidence |
| `results/verification/` | Committed | Numerical protocol checks |
| `paper/tables/` | Committed | Values used by the paper tables and figures |

New runs write `metrics.json`, prediction arrays, routing diagnostics, and plots
under `results/runs/current/`. The retained current matrix is limited to runs
that contribute to the manuscript or supplementary analyses. The committed
paper tables and locked JSON summaries allow the reported numerical comparisons
to be inspected without loading checkpoints.

Regenerate the paper tables and figures after the required run outputs are
available:

```bash
python paper/analysis/generate_paper_assets.py
```

The submission documents under `paper/submission/` are publication artifacts;
they are not inputs to training or evaluation.
