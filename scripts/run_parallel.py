#!/usr/bin/env python
"""
Multi-worker parallel experiment dispatcher for dual A100 GPUs.
- Dispatches remaining tasks dynamically across workers (e.g. 2 workers on GPU 0, 2 workers on GPU 1).
- Automatically checks if metrics.json exists for each task and skips already completed tasks (zero duplicated work!).
- Fully thread-safe queue: no two workers can ever run the same experiment.
- Zero modifications to original model / forecasting codebase.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.main.experiment_matrix import formal_commands, _metrics_path_from_argv


def parse_args():
    parser = argparse.ArgumentParser(description="Run remaining experiments with multi-worker parallelism")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1], help="GPU device IDs to use (e.g. 0 1)")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of concurrent workers per GPU")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "parallel_logs",
        help="Directory to save individual job logs",
    )
    return parser.parse_args()


def worker_loop(worker_id, gpu_id, job_queue, log_dir, total_remaining, progress_lock, progress_counter):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    while True:
        try:
            job_idx, total_all, cmd = job_queue.get_nowait()
        except queue.Empty:
            break

        argv = cmd.split()
        if argv and argv[0] == "python":
            argv.insert(1, "-u")

        # Normalise --gpu flag to 0 since CUDA_VISIBLE_DEVICES isolates the target card
        if "--gpu" in argv:
            idx = argv.index("--gpu")
            argv[idx + 1] = "0"
        else:
            argv.extend(["--gpu", "0"])

        metrics_p = _metrics_path_from_argv(argv)
        experiment_name = metrics_p.parent.name
        job_log_file = log_dir / f"{experiment_name}.log"

        with progress_lock:
            progress_counter[0] += 1
            current = progress_counter[0]
            print(
                f"[{time.strftime('%H:%M:%S')}] [Worker {worker_id} (GPU {gpu_id})] "
                f"Starting [{current}/{total_remaining}] (Task #{job_idx + 1}): {experiment_name}",
                flush=True,
            )

        start_time = time.time()
        with open(job_log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                argv,
                env=env,
                cwd=PROJECT_ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )

        elapsed = time.time() - start_time
        success = proc.returncode == 0 and metrics_p.is_file()
        status_str = "Completed" if success else f"Failed (exit {proc.returncode})"

        with progress_lock:
            print(
                f"[{time.strftime('%H:%M:%S')}] [Worker {worker_id} (GPU {gpu_id})] "
                f"{status_str} in {elapsed:.1f}s: {experiment_name}",
                flush=True,
            )
        job_queue.task_done()


def main():
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      DRC-iMOE 4-Worker Dual-GPU Training Scheduler")
    print(f"  GPUs: {args.gpus}")
    print(f"  Workers per GPU: {args.workers_per_gpu} (total: {len(args.gpus) * args.workers_per_gpu})")
    print(f"  Per-job log directory: {args.log_dir}")
    print("=" * 60)

    all_commands = list(formal_commands())
    total_formal = len(all_commands)

    job_queue = queue.Queue()
    completed_count = 0
    remaining_jobs = []

    for i, cmd in enumerate(all_commands):
        argv = cmd.split()
        expected_metrics = _metrics_path_from_argv(argv)
        if expected_metrics.is_file():
            completed_count += 1
        else:
            remaining_jobs.append((i, total_formal, cmd))

    print(f"\n[Check complete] Total experiments: {total_formal} | Completed: {completed_count} | Remaining: {len(remaining_jobs)}")

    if not remaining_jobs:
        print("\nAll experiments are complete; no jobs need to be rerun.")
        return 0

    for job in remaining_jobs:
        job_queue.put(job)

    progress_lock = threading.Lock()
    progress_counter = [0]
    total_remaining = len(remaining_jobs)

    threads = []
    worker_id = 0
    for gpu_id in args.gpus:
        for _ in range(args.workers_per_gpu):
            t = threading.Thread(
                target=worker_loop,
                args=(
                    worker_id,
                    gpu_id,
                    job_queue,
                    args.log_dir,
                    total_remaining,
                    progress_lock,
                    progress_counter,
                ),
                daemon=True,
            )
            threads.append(t)
            worker_id += 1

    print(f"\n[Started] Launched {len(threads)} worker threads for parallel training...")
    start_total_time = time.time()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    total_time = time.time() - start_total_time
    print("\n" + "=" * 60)
    print(f"All remaining experiments are complete. Elapsed time: {total_time / 3600:.2f} hours.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
