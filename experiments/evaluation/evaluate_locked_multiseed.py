import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import (
    DATA_LOADERS,
    blend_predictions,
    global_metrics,
    predict_pair,
)
from experiments.tuning.tune_validation import load_trained_model, make_experiment_args, set_seed
from utils.artifact_paths import portable_artifact_path, resolve_artifact_path


METRIC_NAMES = ('rmse', 'mae', 'mape_percent', 'r2')
SELECTION_ONLY_POLICY = 'not loaded or evaluated in selection-only mode'


def validate_locked_summary(summary):
    if summary.get('workflow') != 'dual_router':
        raise ValueError('Locked multi-seed evaluation requires a dual_router summary')
    if 'final_test' in summary:
        raise ValueError('The input must be a selection-only summary without final_test')
    if summary.get('protocol', {}).get('test_policy') != SELECTION_ONLY_POLICY:
        raise ValueError('The input summary is not marked as selection-only')

    declared_seeds = summary.get('search', {}).get('seeds', [])
    if not declared_seeds or len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError('Declared seeds must be a non-empty unique list')

    trials = summary.get('trials', [])
    if len(trials) != len(declared_seeds):
        raise ValueError('Trial count must exactly match the declared seeds')

    trials_by_seed = {}
    locked_config = None
    for trial in trials:
        config = trial.get('config', {})
        if 'seed' not in config:
            raise ValueError('Every trial config must contain seed')
        for key in ('alpha', 'baseline_checkpoint', 'curve_checkpoint'):
            if key not in trial:
                raise ValueError(f'Every trial must contain {key}')
        alpha = trial['alpha']
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError('Every validation alpha must be finite and within [0, 1]')

        seed = config['seed']
        if seed in trials_by_seed:
            raise ValueError(f'Duplicate trial seed: {seed}')
        current_locked = {key: value for key, value in config.items() if key != 'seed'}
        if locked_config is None:
            locked_config = current_locked
        elif current_locked != locked_config:
            raise ValueError('All trial configs must be identical except for seed')
        trials_by_seed[seed] = trial

    if set(trials_by_seed) != set(declared_seeds):
        raise ValueError('Trial seeds must exactly cover the declared seeds')
    return [trials_by_seed[seed] for seed in declared_seeds], locked_config


def aggregate_seed_metrics(seed_results):
    aggregate = {
        'std_definition': 'population standard deviation (ddof=0)',
        'models': {},
        'ensemble_minus_baseline': {},
    }
    for model_name in ('baseline', 'curve', 'ensemble'):
        aggregate['models'][model_name] = {}
        for metric_name in METRIC_NAMES:
            values = np.asarray([
                result['metrics'][model_name][metric_name]
                for result in seed_results
            ], dtype=np.float64)
            aggregate['models'][model_name][metric_name] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=0)),
            }

    for metric_name in METRIC_NAMES:
        differences = np.asarray([
            result['metrics']['ensemble'][metric_name]
            - result['metrics']['baseline'][metric_name]
            for result in seed_results
        ], dtype=np.float64)
        aggregate['ensemble_minus_baseline'][metric_name] = {
            'mean': float(differences.mean()),
            'std': float(differences.std(ddof=0)),
            'by_seed': {
                str(result['seed']): float(value)
                for result, value in zip(seed_results, differences)
            },
        }
    return aggregate


def _checkpoint_path(value, summary_path):
    return resolve_artifact_path(value, summary_path)


def _evaluation_args(summary, output_dir, device, gpu, eval_batch_size=64,
                     data_root=Path('datasets/processed')):
    runtime = summary.get('runtime', {})
    return argparse.Namespace(
        workflow='dual_router',
        dataset=summary['dataset'],
        condition=summary['condition'],
        output_dir=output_dir,
        seq_len=runtime['seq_len'],
        pred_len=runtime['pred_len'],
        dataaccess=runtime['dataaccess'],
        batch_size=runtime['batch_size'],
        eval_batch_size=eval_batch_size,
        data_root=Path(data_root),
        train_battery_count=None,
        train_epochs=0,
        patience=0,
        device=device,
        gpu=gpu,
    )


def run_locked_evaluation(summary_path, output_dir, device='cpu', gpu=0,
                          eval_batch_size=64,
                          data_root=Path('datasets/processed')):
    summary_path = Path(summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    with summary_path.open('r', encoding='utf-8') as file:
        search_summary = json.load(file)
    trials, locked_config = validate_locked_summary(search_summary)

    checkpoint_paths = {}
    for trial in trials:
        seed = trial['config']['seed']
        checkpoint_paths[seed] = {
            'baseline': _checkpoint_path(trial['baseline_checkpoint'], summary_path),
            'curve': _checkpoint_path(trial['curve_checkpoint'], summary_path),
        }
        for path in checkpoint_paths[seed].values():
            if not path.is_file():
                raise FileNotFoundError(path)

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    torch_device = torch.device(f'cuda:{gpu}' if device == 'cuda' else 'cpu')
    cli_args = _evaluation_args(
        search_summary,
        output_dir,
        device,
        gpu,
        eval_batch_size,
        data_root,
    )
    if cli_args.dataset not in DATA_LOADERS:
        raise ValueError(f'Unsupported dataset: {cli_args.dataset}')

    seed_results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for trial in trials:
        config = trial['config']
        seed = config['seed']
        set_seed(seed)

        loader_args = make_experiment_args(
            cli_args, config, 'iMOE_CSR', output_dir
        )
        loader_args.skip_test = False
        _, _, test_loader, _ = DATA_LOADERS[cli_args.dataset](loader_args)

        baseline_model, _ = load_trained_model(
            cli_args,
            config,
            'iMOE',
            checkpoint_paths[seed]['baseline'],
            torch_device,
        )
        curve_model, _ = load_trained_model(
            cli_args,
            config,
            'iMOE_CSR',
            checkpoint_paths[seed]['curve'],
            torch_device,
        )
        baseline_pred, curve_pred, true_values = predict_pair(
            baseline_model, curve_model, test_loader, torch_device
        )
        ensemble_pred = blend_predictions(baseline_pred, curve_pred, trial['alpha'])

        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / 'baseline_pred.npy', baseline_pred)
        np.save(seed_dir / 'curve_pred.npy', curve_pred)
        np.save(seed_dir / 'ensemble_pred.npy', ensemble_pred)
        np.save(seed_dir / 'true_values.npy', true_values)

        seed_results.append({
            'seed': seed,
            'trial_id': trial['trial_id'],
            'validation_alpha': trial['alpha'],
            'selection_validation_mse': trial['validation_mse'],
            'baseline_checkpoint': portable_artifact_path(checkpoint_paths[seed]['baseline']),
            'curve_checkpoint': portable_artifact_path(checkpoint_paths[seed]['curve']),
            'prediction_shape': list(ensemble_pred.shape),
            'metrics': {
                'baseline': global_metrics(baseline_pred, true_values),
                'curve': global_metrics(curve_pred, true_values),
                'ensemble': global_metrics(ensemble_pred, true_values),
            },
            'artifacts': {
                'baseline_pred': portable_artifact_path(seed_dir / 'baseline_pred.npy'),
                'curve_pred': portable_artifact_path(seed_dir / 'curve_pred.npy'),
                'ensemble_pred': portable_artifact_path(seed_dir / 'ensemble_pred.npy'),
                'true_values': portable_artifact_path(seed_dir / 'true_values.npy'),
            },
        })

    result = {
        'protocol': {
            'configuration': 'locked by validation-only search summary',
            'alpha': 'per-seed validation alpha from the input summary',
            'test_policy': 'each declared seed evaluated once; no test-based selection',
        },
        'source_search_summary': portable_artifact_path(summary_path),
        'dataset': search_summary['dataset'],
        'condition': search_summary['condition'],
        'device': str(torch_device),
        'declared_seeds': search_summary['search']['seeds'],
        'locked_config_without_seed': locked_config,
        'seeds': seed_results,
        'aggregate': aggregate_seed_metrics(seed_results),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Final test evaluation for a validation-locked multi-seed dual-router run'
    )
    parser.add_argument('--search_summary', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--data_root', type=Path, default=Path('datasets/processed'))
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_locked_evaluation(
        args.search_summary,
        args.output_dir,
        args.device,
        args.gpu,
        args.eval_batch_size,
        args.data_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
