# Experiment Results

- `runs/current/`: The 306 complete result directories used by the current experiment matrix.
- `runs/local/`: Six sets of historical prediction arrays and summaries referenced by the locked main results and the full dual-model fusion comparison.
- `main/drc_imoe_legacy/`: Locked three-seed main results and compression-comparison summaries for UL-NCA, TPSL, and LSD.
- `main/dual_router_legacy/locked_multiseed/`: Locked selection records for the full dual-model fusion comparison.
- `verification/`: Protocol-verification evidence.
- `ablation/`, `tuning/`, and `reproduction/`: Reserved output directories that currently contain no result files.

Model checkpoints are stored in `checkpoints/main/` and `checkpoints/imported/local/`. `PATH_MIGRATION.json` records the migration of historical absolute paths. Paths under `external-artifacts/...` only record the original locations of cleaned historical sources and do not indicate a current project dependency.
