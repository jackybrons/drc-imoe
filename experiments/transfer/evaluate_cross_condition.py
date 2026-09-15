import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import (
    DATA_LOADERS,
    blend_predictions,
    global_metrics,
    load_checkpoint,
    predict_pair,
)
from experiments.evaluation.evaluate_conditioned_lstm_multiseed import (
    METRIC_NAMES,
    SUPPORTED_MODELS,
    make_experiment_args as make_baseline_args,
    predict as predict_single,
)
from experiments.evaluation.evaluate_locked_multiseed import (
    _checkpoint_path,
    _evaluation_args,
    aggregate_seed_metrics,
    validate_locked_summary,
)
from experiments.tuning.tune_validation import load_trained_model, make_experiment_args, set_seed
from utils.artifact_paths import portable_artifact_path, resolve_artifact_path


TPSL_TEST_CELLS = {
    'Arbitrary': ['#5', '#8', '#11', '#12', '#71', '#72', '#33', '#43', '#61', '#62', '#63', '#64', '#65'],
    'Fixed': ['#23', '#35', '#48', '#57'],
}
TPSL_WINDOWS_PER_CELL = 19


def baseline_key(model):
    return {
        'ConditionedLSTM': 'conditioned_lstm',
        'ConditionedMLP': 'conditioned_mlp',
        'Informer': 'informer',
        'PATCHTST': 'patchtst',
    }[model]


def load_baseline_model(
    model,
    checkpoint,
    dataset,
    condition,
    output_dir,
    device,
    gpu=0,
):
    """Build a baseline lazily and load one validation-locked checkpoint."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported baseline model: {model}')
    args = make_baseline_args(
        dataset,
        condition,
        output_dir,
        seed=0,
        device=device.type,
        gpu=gpu,
        skip_test=False,
        model=model,
    )
    module = importlib.import_module(f'models.{model}')
    baseline_model = module.Model(args).to(device)
    load_checkpoint(baseline_model, checkpoint, device)
    baseline_model.eval()
    return baseline_model


def load_conditioned_lstm(
    checkpoint,
    dataset,
    condition,
    output_dir,
    device,
    gpu=0,
):
    """Compatibility loader for the first battery-specific baseline."""
    return load_baseline_model(
        'ConditionedLSTM',
        checkpoint,
        dataset,
        condition,
        output_dir,
        device,
        gpu,
    )


def _resolve_baseline_checkpoints(
    baseline_checkpoints,
    conditioned_lstm_checkpoints,
    expected_seeds,
    source_condition,
):
    requested = dict(baseline_checkpoints or {})
    if conditioned_lstm_checkpoints is not None:
        if 'ConditionedLSTM' in requested:
            raise ValueError('ConditionedLSTM checkpoints were provided twice')
        requested['ConditionedLSTM'] = conditioned_lstm_checkpoints

    resolved = {}
    for model, source in requested.items():
        if model not in SUPPORTED_MODELS:
            raise ValueError(f'Unsupported baseline model: {model}')
        base_dir = Path.cwd()
        if isinstance(source, (str, Path)):
            summary_path = Path(source).resolve()
            with summary_path.open('r', encoding='utf-8') as file:
                summary = json.load(file)
            if summary.get('dataset') != 'TPSL':
                raise ValueError(f'{model} summary must use TPSL')
            if summary.get('condition') != source_condition:
                raise ValueError(f'{model} summary condition does not match the source')
            if summary.get('model') != model:
                raise ValueError(f'{model} summary model does not match')
            if summary.get('declared_seeds') != list(expected_seeds):
                raise ValueError(f'{model} summary seeds do not match the source')
            source = {item['seed']: item['checkpoint'] for item in summary['seeds']}
            base_dir = summary_path.parent
        if not isinstance(source, dict):
            raise TypeError(f'{model} checkpoints must be a seed mapping or summary path')

        by_seed = {}
        for seed in expected_seeds:
            value = source.get(seed, source.get(str(seed)))
            if value is None:
                raise ValueError(f'{model} checkpoints are missing seed {seed}')
            path = resolve_artifact_path(value, base_dir / 'summary.json')
            if not path.is_file():
                raise FileNotFoundError(path)
            by_seed[seed] = path
        if {int(seed) for seed in source} != set(expected_seeds):
            raise ValueError(f'{model} checkpoints must exactly match the source seeds')
        resolved[model] = by_seed
    return resolved


def _aggregate_with_baselines(seed_results, baseline_models):
    aggregate = aggregate_seed_metrics(seed_results)
    for model in baseline_models:
        key = baseline_key(model)
        aggregate['models'][key] = {}
        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [result['metrics'][key][metric_name] for result in seed_results],
                dtype=np.float64,
            )
            aggregate['models'][key][metric_name] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=0)),
            }
    return aggregate


def _validate_baseline_prediction(model, prediction, target, expected_target):
    if prediction.shape != target.shape:
        raise ValueError(f'{model} prediction shape does not match its target')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError(f'{model} prediction or target contains non-finite values')
    if target.shape != expected_target.shape or not np.array_equal(
        target, expected_target
    ):
        raise ValueError(f'{model} target values differ from dual-router targets')


def run_cross_condition_evaluation(
    summary_path,
    target_condition,
    output_dir,
    device='cpu',
    gpu=0,
    baseline_checkpoints=None,
    conditioned_lstm_checkpoints=None,
):
    summary_path = Path(summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    with summary_path.open('r', encoding='utf-8') as file:
        search_summary = json.load(file)
    trials, locked_config = validate_locked_summary(search_summary)

    if search_summary.get('dataset') != 'TPSL':
        raise ValueError('Cross-condition evaluation currently supports TPSL only')
    if target_condition not in TPSL_TEST_CELLS:
        raise ValueError(f'Unsupported TPSL target condition: {target_condition}')
    if search_summary['condition'] == target_condition:
        raise ValueError('Source and target conditions must be different')

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

    declared_seeds = search_summary['search']['seeds']
    extra_checkpoint_paths = _resolve_baseline_checkpoints(
        baseline_checkpoints,
        conditioned_lstm_checkpoints,
        declared_seeds,
        search_summary['condition'],
    )

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    torch_device = torch.device(f'cuda:{gpu}' if device == 'cuda' else 'cpu')
    cli_args = _evaluation_args(search_summary, output_dir, device, gpu)

    first_config = trials[0]['config']
    set_seed(first_config['seed'])
    loader_args = make_experiment_args(
        cli_args, first_config, 'iMOE_CSR', output_dir
    )
    loader_args.skip_test = False
    loader_args.test_condition = target_condition
    _, _, test_loader, _ = DATA_LOADERS['TPSL'](loader_args)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for trial in trials:
        config = trial['config']
        seed = config['seed']
        set_seed(seed)
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
        ensemble_pred = blend_predictions(
            baseline_pred, curve_pred, trial['alpha']
        )

        extra_predictions = {}
        for model, paths in extra_checkpoint_paths.items():
            if model == 'ConditionedLSTM':
                extra_model = load_conditioned_lstm(
                    paths[seed],
                    'TPSL',
                    search_summary['condition'],
                    output_dir,
                    torch_device,
                    gpu,
                )
            else:
                extra_model = load_baseline_model(
                    model,
                    paths[seed],
                    'TPSL',
                    search_summary['condition'],
                    output_dir,
                    torch_device,
                    gpu,
                )
            extra_pred, extra_true = predict_single(
                extra_model, test_loader, torch_device
            )
            _validate_baseline_prediction(
                model, extra_pred, extra_true, true_values
            )
            extra_predictions[model] = extra_pred

        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / 'baseline_pred.npy', baseline_pred)
        np.save(seed_dir / 'curve_pred.npy', curve_pred)
        np.save(seed_dir / 'ensemble_pred.npy', ensemble_pred)
        np.save(seed_dir / 'true_values.npy', true_values)

        for model, prediction in extra_predictions.items():
            np.save(seed_dir / f'{baseline_key(model)}_pred.npy', prediction)

        seed_result = {
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
                **{
                    baseline_key(model): global_metrics(prediction, true_values)
                    for model, prediction in extra_predictions.items()
                },
            },
            'artifacts': {
                'baseline_pred': portable_artifact_path(seed_dir / 'baseline_pred.npy'),
                'curve_pred': portable_artifact_path(seed_dir / 'curve_pred.npy'),
                'ensemble_pred': portable_artifact_path(seed_dir / 'ensemble_pred.npy'),
                'true_values': portable_artifact_path(seed_dir / 'true_values.npy'),
                **{
                    f'{baseline_key(model)}_pred': str(
                        (seed_dir / f'{baseline_key(model)}_pred.npy').resolve()
                    )
                    for model in extra_predictions
                },
            },
        }
        if extra_predictions:
            seed_result['baseline_checkpoints'] = {
                baseline_key(model): portable_artifact_path(extra_checkpoint_paths[model][seed])
                for model in extra_predictions
            }
        seed_results.append(seed_result)

    result = {
        'protocol': {
            'configuration': 'locked by source validation-only search summary',
            'scaler': 'fit on source training cells; target test cells transform only',
            'alpha': 'per-seed source validation alpha from the input summary',
            'target_policy': 'target train and validation cells are not used',
        },
        'source_search_summary': portable_artifact_path(summary_path),
        'dataset': 'TPSL',
        'source_condition': search_summary['condition'],
        'target_condition': target_condition,
        'target_test_cells': TPSL_TEST_CELLS[target_condition],
        'windows_per_cell': TPSL_WINDOWS_PER_CELL,
        'device': str(torch_device),
        'declared_seeds': declared_seeds,
        'additional_baselines': [baseline_key(model) for model in extra_checkpoint_paths],
        'locked_config_without_seed': locked_config,
        'seeds': seed_results,
        'aggregate': _aggregate_with_baselines(
            seed_results, extra_checkpoint_paths
        ),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Zero-shot TPSL cross-condition evaluation with source-locked preprocessing and alpha'
    )
    parser.add_argument('--search_summary', type=Path, required=True)
    parser.add_argument(
        '--target_condition', choices=tuple(TPSL_TEST_CELLS), required=True
    )
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--baseline_summary',
        action='append',
        default=[],
        metavar='MODEL=PATH',
        help='locked baseline summary; repeat for multiple models',
    )
    return parser.parse_args()


def parse_baseline_summaries(values):
    summaries = {}
    for value in values:
        model, separator, path = value.partition('=')
        if not separator or not path:
            raise ValueError('--baseline_summary must use MODEL=PATH')
        if model in summaries:
            raise ValueError(f'Duplicate baseline summary: {model}')
        summaries[model] = path
    return summaries


def main():
    args = parse_args()
    result = run_cross_condition_evaluation(
        args.search_summary,
        args.target_condition,
        args.output_dir,
        args.device,
        args.gpu,
        baseline_checkpoints=parse_baseline_summaries(args.baseline_summary),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
