import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import DATA_LOADERS
from experiments.tuning.tune_validation import load_trained_model, make_experiment_args, set_seed
from utils.artifact_paths import portable_artifact_path, resolve_artifact_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_SEEDS = (2025, 2026, 2027)
BENCHMARK_SYSTEMS = ('iMOE', 'iMOE_CSR', 'dual_router', 'iMOE_SDR')


def _synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_inference(
    infer_batch,
    device_batches,
    device,
    warmup_runs=1,
    timed_runs=10,
):
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError('warmup_runs must be non-negative and timed_runs positive')
    if not device_batches:
        raise ValueError('device_batches must not be empty')
    num_samples = sum(int(inputs[0].shape[0]) for inputs in device_batches)

    timings_ms = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            for inputs in device_batches:
                infer_batch(inputs)
        _synchronize(device)
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(torch.device(device))

        for _ in range(timed_runs):
            _synchronize(device)
            start = time.perf_counter()
            for inputs in device_batches:
                infer_batch(inputs)
            _synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    median_ms = float(np.median(np.asarray(timings_ms, dtype=np.float64)))
    peak_allocated_memory_bytes = None
    if torch.device(device).type == 'cuda':
        peak_allocated_memory_bytes = int(
            torch.cuda.max_memory_allocated(torch.device(device))
        )
    return {
        'warmup_runs': warmup_runs,
        'timed_runs': timed_runs,
        'num_samples': num_samples,
        'timings_ms': timings_ms,
        'median_ms': median_ms,
        'ms_per_sample': median_ms / num_samples,
        'samples_per_second': num_samples / (median_ms / 1000.0),
        'peak_allocated_memory_bytes': peak_allocated_memory_bytes,
    }


def preload_test_inputs(test_loader, device):
    return [
        tuple(value.to(device=device, dtype=torch.float32) for value in inputs)
        for inputs, _ in test_loader
    ]


def parameter_counts(model):
    return {
        'total': sum(parameter.numel() for parameter in model.parameters()),
        'trainable': sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _resolved_path(value, summary_path):
    return resolve_artifact_path(value, summary_path)


def _seed_map(summary, required_keys):
    by_seed = {}
    for result in summary.get('seeds', []):
        seed = result.get('seed')
        if seed in by_seed:
            raise ValueError(f'Duplicate seed: {seed}')
        missing = [key for key in required_keys if key not in result]
        if missing:
            raise ValueError(f'Seed {seed} is missing fields: {missing}')
        by_seed[seed] = result
    if set(by_seed) != set(LOCKED_SEEDS):
        raise ValueError(f'Summary seeds must be {list(LOCKED_SEEDS)}')
    return by_seed


def _runtime_args(summary, output_dir, device, gpu):
    runtime = summary.get('runtime', {})
    return argparse.Namespace(
        workflow='iMOE_SDR',
        dataset=summary['dataset'],
        condition=summary['condition'],
        output_dir=output_dir,
        seq_len=runtime['seq_len'],
        pred_len=runtime['pred_len'],
        dataaccess=runtime['dataaccess'],
        batch_size=runtime['batch_size'],
        train_epochs=0,
        patience=0,
        device=device,
        gpu=gpu,
    )


def _aggregate_seed_benchmarks(seed_results):
    aggregate = {}
    for system in BENCHMARK_SYSTEMS:
        aggregate[system] = {
            'parameters': seed_results[0]['parameters'][system],
        }
        for key in ('median_ms', 'ms_per_sample', 'samples_per_second'):
            values = np.asarray([
                result['timing'][system][key]
                for result in seed_results
            ], dtype=np.float64)
            aggregate[system][key] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=0)),
            }
        memory_values = [
            result['timing'][system]['peak_allocated_memory_bytes']
            for result in seed_results
        ]
        if any(value is None for value in memory_values):
            aggregate[system]['peak_allocated_memory_bytes'] = {
                'mean': None,
                'std': None,
            }
        else:
            values = np.asarray(memory_values, dtype=np.float64)
            aggregate[system]['peak_allocated_memory_bytes'] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=0)),
            }
    speedups = np.asarray([
        result['speedup_dual_router_over_iMOE_SDR']
        for result in seed_results
    ], dtype=np.float64)
    aggregate['speedup_dual_router_over_iMOE_SDR'] = {
        'mean': float(speedups.mean()),
        'std': float(speedups.std(ddof=0)),
    }
    return aggregate


def run_compression_benchmark(
    full_fusion_summary_path,
    shared_summary_path,
    output_path,
    device='cuda',
    gpu=0,
    warmup_runs=1,
    timed_runs=10,
):
    full_fusion_summary_path = Path(full_fusion_summary_path).resolve()
    shared_summary_path = Path(shared_summary_path).resolve()
    output_path = Path(output_path).resolve()
    with full_fusion_summary_path.open('r', encoding='utf-8') as file:
        dual_summary = json.load(file)
    with shared_summary_path.open('r', encoding='utf-8') as file:
        shared_summary = json.load(file)

    if (
        dual_summary.get('dataset') != shared_summary.get('dataset')
        or dual_summary.get('condition') != shared_summary.get('condition')
    ):
        raise ValueError('Dual and shared summaries must use the same dataset split')
    if tuple(shared_summary.get('declared_seeds', ())) != LOCKED_SEEDS:
        raise ValueError('Shared summary does not declare the locked seeds')
    dual_by_seed = _seed_map(
        dual_summary,
        ('baseline_checkpoint', 'curve_checkpoint', 'validation_alpha'),
    )
    shared_by_seed = _seed_map(shared_summary, ('checkpoint',))

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    torch_device = torch.device(f'cuda:{gpu}' if device == 'cuda' else 'cpu')
    cli_args = _runtime_args(shared_summary, output_path.parent, device, gpu)
    if cli_args.dataset not in DATA_LOADERS:
        raise ValueError(f'Unsupported dataset: {cli_args.dataset}')

    dual_locked = dual_summary['locked_config_without_seed']
    shared_locked = shared_summary['locked_config_without_seed']
    seed_results = []
    for seed in LOCKED_SEEDS:
        set_seed(seed)
        dual_config = {**dual_locked, 'seed': seed}
        shared_config = {**shared_locked, 'seed': seed}
        dual_seed = dual_by_seed[seed]
        shared_seed = shared_by_seed[seed]

        loader_args = make_experiment_args(
            cli_args, shared_config, 'iMOE_SDR', output_path.parent
        )
        loader_args.skip_test = False
        _, _, test_loader, _ = DATA_LOADERS[cli_args.dataset](loader_args)
        device_batches = preload_test_inputs(test_loader, torch_device)

        baseline_model, _ = load_trained_model(
            cli_args,
            dual_config,
            'iMOE',
            _resolved_path(dual_seed['baseline_checkpoint'], full_fusion_summary_path),
            torch_device,
        )
        baseline_model.float().eval()
        baseline_parameters = parameter_counts(baseline_model)

        def infer_imoe(inputs):
            output, _ = baseline_model(*inputs)
            return output

        imoe_timing = benchmark_inference(
            infer_imoe,
            device_batches,
            torch_device,
            warmup_runs,
            timed_runs,
        )
        if torch_device.type == 'cuda':
            baseline_model.to('cpu')
            torch.cuda.empty_cache()

        curve_model, _ = load_trained_model(
            cli_args,
            dual_config,
            'iMOE_CSR',
            _resolved_path(dual_seed['curve_checkpoint'], full_fusion_summary_path),
            torch_device,
        )
        curve_model.float().eval()
        curve_parameters = parameter_counts(curve_model)

        def infer_csr(inputs):
            output, _ = curve_model(*inputs)
            return output

        csr_timing = benchmark_inference(
            infer_csr,
            device_batches,
            torch_device,
            warmup_runs,
            timed_runs,
        )
        if torch_device.type == 'cuda':
            curve_model.to('cpu')
            torch.cuda.empty_cache()

        baseline_model.to(torch_device).float().eval()
        curve_model.to(torch_device).float().eval()
        alpha = float(dual_seed['validation_alpha'])

        def infer_ensemble(inputs):
            baseline_output, _ = baseline_model(*inputs)
            curve_output, _ = curve_model(*inputs)
            return baseline_output + alpha * (curve_output - baseline_output)

        ensemble_timing = benchmark_inference(
            infer_ensemble,
            device_batches,
            torch_device,
            warmup_runs,
            timed_runs,
        )
        if torch_device.type == 'cuda':
            baseline_model.to('cpu')
            curve_model.to('cpu')
            torch.cuda.empty_cache()

        shared_model, _ = load_trained_model(
            cli_args,
            shared_config,
            'iMOE_SDR',
            _resolved_path(shared_seed['checkpoint'], shared_summary_path),
            torch_device,
        )
        shared_model.float().eval()
        shared_parameters = parameter_counts(shared_model)

        def infer_shared(inputs):
            output, _ = shared_model(*inputs)
            return output

        shared_timing = benchmark_inference(
            infer_shared,
            device_batches,
            torch_device,
            warmup_runs,
            timed_runs,
        )
        if torch_device.type == 'cuda':
            shared_model.to('cpu')
            torch.cuda.empty_cache()
        ensemble_parameters = {
            key: baseline_parameters[key] + curve_parameters[key]
            for key in ('total', 'trainable')
        }
        seed_results.append({
            'seed': seed,
            'alpha': alpha,
            'parameters': {
                'iMOE': baseline_parameters,
                'iMOE_CSR': curve_parameters,
                'dual_router': ensemble_parameters,
                'iMOE_SDR': shared_parameters,
                'reduction_percent': {
                    key: 100.0 * (1.0 - shared_parameters[key] / ensemble_parameters[key])
                    for key in ('total', 'trainable')
                },
            },
            'timing': {
                'iMOE': imoe_timing,
                'iMOE_CSR': csr_timing,
                'dual_router': ensemble_timing,
                'iMOE_SDR': shared_timing,
            },
            'speedup_dual_router_over_iMOE_SDR': (
                ensemble_timing['median_ms'] / shared_timing['median_ms']
            ),
        })

    result = {
        'protocol': {
            'mode': 'eval plus torch.inference_mode',
            'data_loading': 'CSV parsing, loader construction, and device transfer excluded',
            'systems': {
                'iMOE': 'single statistical-router model forward',
                'iMOE_CSR': 'single curve-router model forward',
                'dual_router': 'iMOE forward + iMOE-CSR forward + on-device blend',
                'iMOE_SDR': 'single shared dual-router model forward',
            },
            'precision': 'float32; no autocast or compilation',
            'batch_size': cli_args.batch_size,
            'memory': (
                'CUDA peak allocated bytes with only the measured system model(s) '
                'and preloaded input batches resident; null on CPU'
            ),
            'warmup_runs': warmup_runs,
            'timed_runs': timed_runs,
        },
        'full_fusion_summary': portable_artifact_path(full_fusion_summary_path),
        'shared_summary': portable_artifact_path(shared_summary_path),
        'dataset': shared_summary['dataset'],
        'condition': shared_summary['condition'],
        'device': str(torch_device),
        'device_name': (
            torch.cuda.get_device_name(torch_device)
            if torch_device.type == 'cuda'
            else 'CPU'
        ),
        'declared_seeds': list(LOCKED_SEEDS),
        'seeds': seed_results,
        'aggregate': _aggregate_seed_benchmarks(seed_results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pure-forward benchmark for iMOE, iMOE-CSR, dual_router, and DRC-iMOE'
    )
    parser.add_argument('--full_fusion_summary', type=Path, required=True)
    parser.add_argument('--shared_summary', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--warmup_runs', type=int, choices=(1,), default=1)
    parser.add_argument('--timed_runs', type=int, choices=(10,), default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_compression_benchmark(
        args.full_fusion_summary,
        args.shared_summary,
        args.output_path,
        args.device,
        args.gpu,
        args.warmup_runs,
        args.timed_runs,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
