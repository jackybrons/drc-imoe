import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import DATA_LOADERS, global_metrics
from experiments.tuning.tune_validation import load_trained_model, make_experiment_args, set_seed
from utils.artifact_paths import portable_artifact_path, resolve_artifact_path


METRIC_NAMES = ('rmse', 'mae', 'mape_percent', 'r2')
MODEL_NAMES = ('baseline', 'dual_router', 'iMOE_SDR')
SELECTION_ONLY_POLICY = 'not loaded or evaluated in selection-only mode'
LOCKED_SEEDS = (2025, 2026, 2027)


def validate_shared_summary(summary):
    if summary.get('workflow') != 'iMOE_SDR':
        raise ValueError('Shared evaluation requires an iMOE_SDR summary')
    if 'final_test' in summary:
        raise ValueError('The input must be a selection-only summary without final_test')
    if summary.get('protocol', {}).get('test_policy') != SELECTION_ONLY_POLICY:
        raise ValueError('The input summary is not marked as selection-only')
    declared_seeds = tuple(summary.get('search', {}).get('seeds', ()))
    if declared_seeds != LOCKED_SEEDS:
        raise ValueError(f'iMOE_SDR seeds must be {list(LOCKED_SEEDS)}')

    trials = summary.get('trials', [])
    if len(trials) != len(LOCKED_SEEDS):
        raise ValueError('The summary must contain exactly one trial per locked seed')
    by_seed = {}
    locked_config = None
    for trial in trials:
        config = trial.get('config', {})
        seed = config.get('seed')
        if seed not in LOCKED_SEEDS or seed in by_seed:
            raise ValueError(f'Invalid or duplicate trial seed: {seed}')
        for key in ('checkpoint', 'validation_mse'):
            if key not in trial:
                raise ValueError(f'Every shared trial must contain {key}')
        current_locked = {key: value for key, value in config.items() if key != 'seed'}
        if locked_config is None:
            locked_config = current_locked
        elif current_locked != locked_config:
            raise ValueError('All shared trial configs must be identical except for seed')
        by_seed[seed] = trial
    if set(by_seed) != set(LOCKED_SEEDS):
        raise ValueError('Shared trials must cover all locked seeds')
    return [by_seed[seed] for seed in LOCKED_SEEDS], locked_config


def aggregate_comparison_metrics(seed_results):
    available_models = tuple(
        model_name
        for model_name in MODEL_NAMES
        if model_name in seed_results[0]['metrics']
    )
    aggregate = {
        'std_definition': 'population standard deviation (ddof=0)',
        'models': {},
    }
    for model_name in available_models:
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

    for reference_name in ('baseline', 'dual_router'):
        if reference_name not in available_models:
            continue
        difference_key = f'iMOE_SDR_minus_{reference_name}'
        aggregate[difference_key] = {}
        for metric_name in METRIC_NAMES:
            values = np.asarray([
                result['metrics']['iMOE_SDR'][metric_name]
                - result['metrics'][reference_name][metric_name]
                for result in seed_results
            ], dtype=np.float64)
            aggregate[difference_key][metric_name] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=0)),
                'by_seed': {
                    str(result['seed']): float(value)
                    for result, value in zip(seed_results, values)
                },
            }
    return aggregate


def _resolved_path(value, summary_path):
    return resolve_artifact_path(value, summary_path)


def _reference_results(reference, reference_path):
    if tuple(reference.get('declared_seeds', ())) != LOCKED_SEEDS:
        raise ValueError('Reference summary does not contain the locked seeds')
    by_seed = {}
    for result in reference.get('seeds', []):
        seed = result.get('seed')
        if seed in by_seed:
            raise ValueError(f'Duplicate reference seed: {seed}')
        artifacts = result.get('artifacts', {})
        required = ('baseline_pred', 'ensemble_pred', 'true_values')
        if any(key not in artifacts for key in required):
            raise ValueError(f'Reference seed {seed} is missing prediction artifacts')
        by_seed[seed] = {
            key: _resolved_path(artifacts[key], reference_path)
            for key in required
        }
    if set(by_seed) != set(LOCKED_SEEDS):
        raise ValueError('Reference summary must cover all locked seeds')
    return by_seed


def _evaluation_args(summary, output_dir, device, gpu, eval_batch_size=64,
                     data_root=Path('datasets/processed')):
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
        eval_batch_size=eval_batch_size,
        data_root=Path(data_root),
        train_battery_count=None,
        train_epochs=0,
        patience=0,
        device=device,
        gpu=gpu,
    )


def _predict(model, data_loader, device):
    predictions = []
    true_values = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = tuple(value.to(device) for value in inputs)
            outputs, _ = model(*inputs)
            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.numpy())
    return np.concatenate(predictions), np.concatenate(true_values)


def run_shared_evaluation(
    search_summary_path,
    reference_summary_path,
    output_dir,
    device='cpu',
    gpu=0,
    eval_batch_size=64,
    data_root=Path('datasets/processed'),
):
    search_summary_path = Path(search_summary_path).resolve()
    reference_summary_path = (
        None
        if reference_summary_path is None
        else Path(reference_summary_path).resolve()
    )
    output_dir = Path(output_dir).resolve()
    with search_summary_path.open('r', encoding='utf-8') as file:
        search_summary = json.load(file)
    reference_summary = None
    if reference_summary_path is not None:
        with reference_summary_path.open('r', encoding='utf-8') as file:
            reference_summary = json.load(file)

    trials, locked_config = validate_shared_summary(search_summary)
    references = None
    if reference_summary is not None:
        if (
            reference_summary.get('dataset') != search_summary.get('dataset')
            or reference_summary.get('condition') != search_summary.get('condition')
        ):
            raise ValueError(
                'Shared and reference summaries must use the same dataset split'
            )
        references = _reference_results(reference_summary, reference_summary_path)

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

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for trial in trials:
        config = trial['config']
        seed = config['seed']
        set_seed(seed)
        checkpoint = _resolved_path(trial['checkpoint'], search_summary_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        loader_args = make_experiment_args(cli_args, config, 'iMOE_SDR', output_dir)
        loader_args.skip_test = False
        _, _, test_loader, _ = DATA_LOADERS[cli_args.dataset](loader_args)
        model, _ = load_trained_model(
            cli_args, config, 'iMOE_SDR', checkpoint, torch_device
        )
        prediction, true_values = _predict(model, test_loader, torch_device)

        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        pred_path = (seed_dir / 'pred_values.npy').resolve()
        true_path = (seed_dir / 'true_values.npy').resolve()
        np.save(pred_path, prediction)
        np.save(true_path, true_values)

        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        metrics = {'iMOE_SDR': global_metrics(prediction, true_values)}
        artifacts = {
            'pred_values': portable_artifact_path(pred_path),
            'true_values': portable_artifact_path(true_path),
        }
        if references is not None:
            reference_paths = references[seed]
            reference_true = np.load(reference_paths['true_values'])
            if not np.array_equal(true_values, reference_true):
                raise ValueError(
                    f'Test truth differs from the reference for seed {seed}'
                )
            baseline_pred = np.load(reference_paths['baseline_pred'])
            dual_pred = np.load(reference_paths['ensemble_pred'])
            if (
                baseline_pred.shape != true_values.shape
                or dual_pred.shape != true_values.shape
            ):
                raise ValueError(f'Reference prediction shape mismatch for seed {seed}')
            metrics.update({
                'baseline': global_metrics(baseline_pred, true_values),
                'dual_router': global_metrics(dual_pred, true_values),
            })
            artifacts.update({
                'reference_baseline_pred': str(reference_paths['baseline_pred']),
                'reference_dual_router_pred': str(reference_paths['ensemble_pred']),
                'reference_true_values': str(reference_paths['true_values']),
            })

        seed_results.append({
            'seed': seed,
            'trial_id': trial['trial_id'],
            'selection_validation_mse': trial['validation_mse'],
            'checkpoint': portable_artifact_path(checkpoint),
            'prediction_shape': list(prediction.shape),
            'parameters': {
                'total': total_parameters,
                'trainable': trainable_parameters,
            },
            'metrics': metrics,
            'artifacts': artifacts,
        })

    result = {
        'protocol': {
            'configuration': 'fixed iMOE_SDR configuration; no hyperparameter search',
            'checkpoint_selection': 'per-seed validation prediction MSE',
            'test_policy': 'each locked seed evaluated once',
            'full_fusion_comparison_role': (
                'complete iMOE + iMOE_CSR prediction-level fusion comparison; '
                'excluded from DRC-iMOE selection and not a mathematical bound'
            ),
        },
        'source_search_summary': portable_artifact_path(search_summary_path),
        'reference_summary': (
            None
            if reference_summary_path is None
            else portable_artifact_path(reference_summary_path)
        ),
        'dataset': search_summary['dataset'],
        'condition': search_summary['condition'],
        'device': str(torch_device),
        'runtime': search_summary['runtime'],
        'declared_seeds': list(LOCKED_SEEDS),
        'locked_config_without_seed': locked_config,
        'seeds': seed_results,
        'aggregate': aggregate_comparison_metrics(seed_results),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Locked multi-seed evaluation for the shared dual-router model'
    )
    parser.add_argument('--search_summary', type=Path, required=True)
    parser.add_argument('--reference_summary', type=Path)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--data_root', type=Path, default=Path('datasets/processed'))
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_shared_evaluation(
        args.search_summary,
        args.reference_summary,
        args.output_dir,
        args.device,
        args.gpu,
        args.eval_batch_size,
        args.data_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
