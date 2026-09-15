# Experiment Architecture

All experiment definitions, entry points, evaluation routines, and tuning code are organized under the `experiments/` module. Do not place temporary experiment scripts in the project root.

---

## Directory Structure

```text
experiments/
├── core/                   # Core training and prediction loops
│   ├── exp_basic.py        # Abstract experiment base class for device management and hooks
│   └── exp_forecasting.py  # Sequence training, early stopping (patience=250), validation selection, and test evaluation
├── main/                   # Main entry points and experiment matrix
│   ├── run.py              # Standard CLI entry point for single-model, single-configuration runs
│   ├── experiment_matrix.py# Formal 339-task matrix, parameterized ablations, and smoke checks
│   ├── run_drc_imoe_server.py # Directed search and locked three-seed execution workflow
│   └── run_high_yield_drc_imoe_server.py # Server-side auxiliary runner
├── evaluation/             # Multi-seed evaluation and model-compression benchmarks
│   ├── benchmark_compression.py # Parameters, FLOPs, GPU memory, and end-to-end latency measurements
│   ├── evaluate_locked_multiseed.py # Aggregation of metrics from fixed-seed repeated evaluations
│   ├── evaluate_dual_router.py      # Full prediction-level fusion comparison for two models
│   ├── evaluate_shared_dual_router_multiseed.py # DRC-iMOE metric computation
│   └── export_locked_expert_weights.py          # Expert weights and router activations for interpretability analysis
├── transfer/               # Cross-condition transfer evaluation
│   └── evaluate_cross_condition.py # TPSL Arbitrary ↔ Fixed zero-shot transfer tests
└── tuning/                 # Validation-only tuning tools; the test set must not be used
    ├── tune_validation.py  # Staged grid and hyperparameter search using validation data only
    └── recompute_validation_summary.py # Recompute validation summaries from checkpoints
```

---

## Ablation and Reproduction Notes

- **Architecture and sparsity ablations**:
  - Variants include `stat_only` (statistical router only), `curve_only` (curve router only), `fixed_fusion` (fixed-weight fusion), `learned_fusion` (dynamic gated fusion), `without_noise` (router noise disabled), `without_top_k` (Top-K sparsification disabled), `dense`, and `top-5`.
  - `experiments.main.experiment_matrix` generates all parameterized ablation variants, which are dispatched through the shared `experiments.main.run` entry point.
- **Baselines and reproduction**:
  - All comparison models (iMOE, iMOE_CSR, ConditionedLSTM, ConditionedMLP, PatchTST, Informer, and DegradationFormer) share the same data-loading, feature-extraction, and evaluation loops.

---

## Common Commands

Run all commands from the project root using module mode (`-m`):

1. **Single-model training run**:
   ```bash
   python -m experiments.main.run \
     --model iMOE_SDR --dataset TPSL --condition Arbitrary --seed 2025
   ```
2. **Inspect the formal experiment matrix**:
   ```bash
   python -m experiments.main.experiment_matrix list
   python -m experiments.main.experiment_matrix formal-commands
   ```
3. **Minimal-data pipeline smoke test**:
   ```bash
   python -m experiments.main.experiment_matrix smoke --model iMOE_SDR --max-batteries 2
   ```
4. **Model-compression and inference-efficiency benchmark**:
   ```bash
   python -m experiments.evaluation.benchmark_compression
   ```
