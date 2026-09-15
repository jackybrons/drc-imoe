import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import DATA_LOADERS
from experiments.evaluation.evaluate_locked_multiseed import (
    _evaluation_args,
    validate_locked_summary,
)
from experiments.tuning.tune_validation import load_trained_model, make_experiment_args, set_seed
from utils.artifact_paths import portable_artifact_path, resolve_artifact_path


def resolve_checkpoint(checkpoint_value, summary_path):
    checkpoint = resolve_artifact_path(checkpoint_value, summary_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def collect_baseline_outputs(model, test_loader, device):
    predictions = []
    weights = []
    model.eval()
    with torch.inference_mode():
        for inputs, _ in test_loader:
            inputs = tuple(value.to(device) for value in inputs)
            prediction, batch_weights = model(*inputs)
            predictions.append(prediction.cpu().numpy())
            weights.append(batch_weights.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(weights)


def validate_weights(weights, num_experts):
    if weights.ndim != 2 or weights.shape[1] != num_experts:
        raise ValueError(
            f'Expected expert weights with shape [samples, {num_experts}], '
            f'got {weights.shape}'
        )
    if not np.isfinite(weights).all():
        raise ValueError('Expert weights contain non-finite values')
    if np.any(weights < 0.0):
        raise ValueError('Expert weights contain negative values')

    row_sums = weights.sum(axis=1)
    max_row_sum_error = float(np.max(np.abs(row_sums - 1.0)))
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(
            f'Expert weights do not sum to one; max error={max_row_sum_error}'
        )
    return max_row_sum_error


def export_locked_baseline_weights(
    search_summary_path, final_test_dir, device='cpu', gpu=0
):
    started = time.perf_counter()
    search_summary_path = Path(search_summary_path).resolve()
    final_test_dir = Path(final_test_dir).resolve()
    with search_summary_path.open('r', encoding='utf-8') as file:
        search_summary = json.load(file)
    trials, _ = validate_locked_summary(search_summary)

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    torch_device = torch.device(f'cuda:{gpu}' if device == 'cuda' else 'cpu')
    cli_args = _evaluation_args(
        search_summary, final_test_dir, device=device, gpu=gpu
    )
    if cli_args.dataset not in DATA_LOADERS:
        raise ValueError(f'Unsupported dataset: {cli_args.dataset}')

    results = []
    for trial in trials:
        seed_started = time.perf_counter()
        config = trial['config']
        seed = config['seed']
        set_seed(seed)

        loader_args = make_experiment_args(
            cli_args, config, 'iMOE', final_test_dir
        )
        loader_args.skip_test = False
        _, _, test_loader, _ = DATA_LOADERS[cli_args.dataset](loader_args)

        checkpoint = resolve_checkpoint(
            trial['baseline_checkpoint'], search_summary_path
        )
        model, _ = load_trained_model(
            cli_args, config, 'iMOE', checkpoint, torch_device
        )
        predictions, expert_weights = collect_baseline_outputs(
            model, test_loader, torch_device
        )
        max_row_sum_error = validate_weights(
            expert_weights, config['num_experts']
        )

        seed_dir = final_test_dir / f'seed_{seed}'
        expected_prediction_path = seed_dir / 'baseline_pred.npy'
        if not expected_prediction_path.is_file():
            raise FileNotFoundError(expected_prediction_path)
        expected_predictions = np.load(expected_prediction_path)
        if predictions.shape != expected_predictions.shape:
            raise ValueError(
                f'Baseline prediction shape changed for seed {seed}: '
                f'{predictions.shape} != {expected_predictions.shape}'
            )
        max_prediction_difference = float(
            np.max(np.abs(predictions - expected_predictions))
        )
        prediction_reproduction_rmse = float(np.sqrt(np.mean(
            (predictions - expected_predictions) ** 2
        )))
        if prediction_reproduction_rmse > 1e-4:
            raise ValueError(
                f'Baseline predictions do not match the locked evaluation for '
                f'seed {seed}; reproduction RMSE='
                f'{prediction_reproduction_rmse}, max difference='
                f'{max_prediction_difference}'
            )

        npy_path = seed_dir / 'expert_weights.npy'
        csv_path = seed_dir / 'expert_weights.csv'
        np.save(npy_path, expert_weights)
        header = ','.join(
            ['sample_index']
            + [f'expert_{index + 1}' for index in range(expert_weights.shape[1])]
        )
        csv_values = np.column_stack((
            np.arange(expert_weights.shape[0], dtype=np.int64),
            expert_weights,
        ))
        np.savetxt(
            csv_path,
            csv_values,
            delimiter=',',
            header=header,
            comments='',
            fmt=['%d'] + ['%.9g'] * expert_weights.shape[1],
        )

        results.append({
            'seed': seed,
            'checkpoint': portable_artifact_path(checkpoint),
            'shape': list(expert_weights.shape),
            'finite': True,
            'max_row_sum_error': max_row_sum_error,
            'prediction_reproduction_rmse': prediction_reproduction_rmse,
            'max_prediction_difference': max_prediction_difference,
            'npy': str(npy_path),
            'csv': str(csv_path),
            'elapsed_seconds': time.perf_counter() - seed_started,
        })
        del model
        if torch_device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'dataset': search_summary['dataset'],
        'condition': search_summary['condition'],
        'device': str(torch_device),
        'model': 'iMOE baseline/statistical branch',
        'results': results,
        'elapsed_seconds': time.perf_counter() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Export baseline iMOE expert weights from validation-locked '
            'checkpoints without training'
        )
    )
    parser.add_argument('--search_summary', type=Path, required=True)
    parser.add_argument('--final_test_dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    result = export_locked_baseline_weights(
        args.search_summary,
        args.final_test_dir,
        device=args.device,
        gpu=args.gpu,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
