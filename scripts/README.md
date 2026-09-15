# Run Scripts and Parallel Scheduling Tools

This directory contains scheduling utilities for high-throughput multi-GPU training and sharded execution of the experiment matrix.

---

## Files

```text
scripts/
├── run_parallel.py      # Recommended dynamic, thread-safe scheduler for two or more GPUs
├── commands_gpu0.txt    # 170 single-line commands alternately assigned to GPU 0
├── commands_gpu1.txt    # 169 single-line commands alternately assigned to GPU 1
├── run_gpu0.sh          # Bash script that runs commands_gpu0.txt sequentially on GPU 0
└── run_gpu1.sh          # Bash script that runs commands_gpu1.txt sequentially on GPU 1
```

---

## 1. Dynamic Parallel Scheduler (`run_parallel.py`)

`run_parallel.py` dynamically schedules experiment jobs:

1. **Skip completed jobs**:
   - Before each run, `_metrics_path_from_argv` checks whether every job already has a complete `metrics.json`.
   - Completed jobs are skipped immediately, and only unfinished jobs enter the queue.
2. **Multi-worker load balancing**:
   - Multiple concurrent workers can run on each GPU, for example with `--workers_per_gpu 1` or `2`.
   - A thread-safe job queue prevents two workers from running the same experiment.
3. **Per-job log isolation**:
   - Standard output and errors from each experiment are written to `results/runs/parallel_logs/<experiment_name>.log`.
4. **Device isolation**:
   - The scheduler sets `CUDA_VISIBLE_DEVICES` for each subprocess and normalizes the command's `--gpu` flag to the device-local index `0`.

### Examples

Start concurrent training on a server with two GPUs (GPU 0 and GPU 1):

```bash
python scripts/run_parallel.py --gpus 0 1 --workers_per_gpu 1
```

If sufficient GPU memory is available, such as on an 80 GB A100, run two workers per GPU:

```bash
python scripts/run_parallel.py --gpus 0 1 --workers_per_gpu 2
```

Run persistently in the background so jobs survive an SSH disconnect:

```bash
nohup python scripts/run_parallel.py --gpus 0 1 --workers_per_gpu 1 > parallel_dispatcher.log 2>&1 &
```

---

## 2. Static Shard Scripts (`run_gpu0.sh` / `run_gpu1.sh`)

If the server environment restricts multithreaded Python scheduling, use the static Bash-based shards:

```bash
# Start the GPU 0 shard in terminal 1
bash scripts/run_gpu0.sh

# Start the GPU 1 shard in terminal 2
bash scripts/run_gpu1.sh
```

Generate the complete job list with `python -m experiments.main.experiment_matrix formal-commands`.
