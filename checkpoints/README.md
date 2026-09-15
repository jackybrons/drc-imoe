# Model Checkpoints

- `main/`: The 306 checkpoints used by the current experiment matrix, including the main, transfer, ablation, data-efficiency, and effective-horizon experiments.
- `imported/local/shared_dual_router_{ul,tpsl_arbitrary,lsd}/`: Nine historical checkpoints used by the locked three-seed DRC-iMOE main results.
- `imported/local/locked_{ul,tpsl_arbitrary,lsd}_top4_multiseed/`: Eighteen historical checkpoints used by the full dual-model fusion comparison.
- `transfer/`, `ablation/`, `reproduction/`, and `tuning/`: Reserved output directories that currently contain no checkpoints.

Metrics, prediction arrays, and logs should not be written to this directory.
