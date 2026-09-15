# DRC-iMOE

Official research code for **Dual-Route Compression of Mixture-of-Experts
(DRC-iMOE)**, a battery-capacity trajectory forecasting method for second-life
lithium-ion batteries.

DRC-iMOE combines a curve-aware router and a statistical router at the sample
level. Both routes dispatch a shared bank of degradation experts, followed by a
condition-aware BiLSTM decoder. The implementation also contains the seven
comparison models used in the accompanying study: iMOE, iMOE-CSR,
ConditionedLSTM, ConditionedMLP, PatchTST, Informer, and DegradationFormer.

## Repository contents

| Path | Contents |
| --- | --- |
| `models/` | DRC-iMOE and comparison-model definitions |
| `layers/` | Attention, embedding, and encoder components |
| `datasets/` | Data loaders and preprocessing programs |
| `experiments/` | Training, evaluation, transfer, and tuning workflows |
| `scripts/` | Experiment-matrix launchers |
| `results/` | Aggregated metrics and protocol-verification records |
| `paper/tables/` | Tables derived from the reported experiments |
| `paper/figures/` | Figure-generation programs and publication figures |
| `tests/` | Unit and integration tests |

Raw data, processed data, trained checkpoints, and complete per-run prediction
arrays are not stored in Git. Their expected locations are documented below.

## Installation

Python 3.11 is recommended. Install the pinned core environment in a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The release contains 114 tests plus 6 parameterized subtests.

## Data

The experiments use three public data sources. Download them from their
original repositories so that their licensing and citation information remain
attached to the data.

| Dataset | Source |
| --- | --- |
| UL battery ageing data | [Zenodo, DOI 10.5281/zenodo.6379165](https://doi.org/10.5281/zenodo.6379165) |
| Tongji second-life battery data (TPSL) | [Mendeley Data, DOI 10.17632/kw34hhw7xg.2](https://doi.org/10.17632/kw34hhw7xg/2) |
| Large-scale second-life battery degradation data (LSD) | [Zenodo, DOI 10.5281/zenodo.14859405](https://doi.org/10.5281/zenodo.14859405) |

Place the downloaded archives under `raw_data/`, then run:

```bash
python -m datasets.preprocessing.prepare_ul_datasets \
  --raw_dir raw_data/UL --output_dir datasets/processed

python -m datasets.preprocessing.prepare_tpsl_datasets \
  --raw_dir raw_data/TPSL --output_dir datasets/processed
```

Prepare LSD with `datasets/preprocessing/LSD-data-generate.ipynb`. The processed
layout and dataset-specific notes are in
[`datasets/preprocessing/DATA_DOWNLOAD.md`](datasets/preprocessing/DATA_DOWNLOAD.md).

## Reproducing the experiments

Inspect the paper experiment matrix without starting training:

```bash
python -m experiments.main.experiment_matrix list
python -m experiments.main.experiment_matrix formal-commands --model iMOE_SDR
```

Run one DRC-iMOE experiment on the arbitrary-profile TPSL data:

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

Run the formal matrix on two GPUs:

```bash
python -m experiments.main.experiment_matrix formal-run \
  --model iMOE_SDR --gpus 0 1
```

Training outputs are written to `results/runs/current/` and checkpoints to
`checkpoints/main/` by default. Both directories are intentionally excluded
from version control because they are generated artifacts. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the locked protocol, output map,
and the relationship between committed summaries and generated files.

## Experimental protocol

- Battery-level splits prevent cycles from the same cell appearing in more
  than one split.
- Checkpoints and hyperparameters are selected by global validation MSE only.
- Formal results use seeds 2025, 2026, and 2027.
- Training uses batches of 32 and a maximum of 1500 epochs with early-stopping
  patience of 250.
- Evaluation uses batches of 64; the equivalence check against batch size 32 is
  recorded in `results/verification/inference_batch_32_vs_64.json`.

## Research-data availability

The source datasets are available from the DOI links above. This repository
provides preprocessing code, model and evaluation code, fixed experiment
definitions, aggregated numerical results, and tests. Processed copies of the
datasets and trained weights are not redistributed; they can be regenerated
from the cited public data and the commands in this repository.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The file is
rendered by GitHub through the repository's **Cite this repository** action.

## License

The code is released under the [MIT License](LICENSE). The source datasets keep
the licenses specified by their respective repositories.
