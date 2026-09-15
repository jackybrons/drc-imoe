# Data Downloads and Directory Layout

Do not commit raw data to version control. The only official root directory for preprocessed data is `datasets/processed/`.

| Dataset | Source | Expected size |
|---|---|---|
| UL | [Zenodo 10.5281/zenodo.6379165](https://doi.org/10.5281/zenodo.6379165) | 66 NCA, 55 NCM, and 9 NCM+NCA CSV files |
| TPSL | [Mendeley 10.17632/kw34hhw7xg.2](https://data.mendeley.com/datasets/kw34hhw7xg/2) | Arbitrary and Fixed cell directories |
| LSD | [Zenodo 10.5281/zenodo.14859405](https://doi.org/10.5281/zenodo.14859405) | 86 CSV files |

```bash
python -m datasets.preprocessing.prepare_ul_datasets \
  --raw_dir raw_data/UL --output_dir datasets/processed

python -m datasets.preprocessing.prepare_tpsl_datasets \
  --raw_dir raw_data/TPSL --output_dir datasets/processed
```

Use `datasets/preprocessing/LSD-data-generate.ipynb` for LSD. By default, the notebook reads from `raw_data/LSD/Cell report/Second_life_phase/` within the project and writes to `datasets/processed/LSD/`; it no longer contains a machine-specific absolute path.

```text
datasets/processed/
├── LSD/
├── TPSL-Arbitrary/
├── TPSL-Fixed/
├── UL-NCA/
├── UL-NCM/
└── UL-NCMNCA/
```
