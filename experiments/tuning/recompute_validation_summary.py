import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import (
    DATA_LOADERS,
    blend_predictions,
    optimal_curve_alpha,
    predict_pair,
)
from experiments.tuning.tune_validation import (
    load_trained_model,
    make_experiment_args,
    select_best_trial,
    set_seed,
    write_summary,
)
from utils.artifact_paths import portable_artifact_path, resolve_artifact_path


SELECTION_ONLY_POLICY = 'not loaded or evaluated in selection-only mode'


def _checkpoint_path(value, summary_path):
    return resolve_artifact_path(value, summary_path)


def _runtime_args(summary, output_dir, device, gpu):
    runtime = summary['runtime']
    return argparse.Namespace(
        workflow='dual_router',
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


def recompute_validation_summary(
    source_summary_path,
    output_summary_path,
    device='cpu',
    gpu=0,
):
    source_summary_path = Path(source_summary_path).resolve()
    output_summary_path = Path(output_summary_path).resolve()
    if source_summary_path == output_summary_path:
        raise ValueError('Output summary must not overwrite the source summary')

    with source_summary_path.open('r', encoding='utf-8') as file:
        source = json.load(file)
    if source.get('workflow') != 'dual_router':
        raise ValueError('Checkpoint recomputation requires a dual_router summary')
    if source.get('protocol', {}).get('test_policy') != SELECTION_ONLY_POLICY:
        raise ValueError('Source summary must be validation-only')
    if 'final_test' in source:
        raise ValueError('Source summary must not contain final_test')
    if not source.get('trials'):
        raise ValueError('Source summary has no trials')
    if source.get('dataset') not in DATA_LOADERS:
        raise ValueError(f"Unsupported dataset: {source.get('dataset')}")
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    torch_device = torch.device(f'cuda:{gpu}' if device == 'cuda' else 'cpu')
    cli_args = _runtime_args(source, output_summary_path.parent, device, gpu)
    summary = copy.deepcopy(source)
    summary.pop('winner', None)
    summary['trials'] = []
    summary['protocol']['prediction_mode'] = 'eval'
    summary['recompute'] = {
        'source_summary': portable_artifact_path(source_summary_path),
        'mode': 'validation predictions from existing checkpoints',
        'training_jobs': 0,
        'test_policy': SELECTION_ONLY_POLICY,
        'device': str(torch_device),
    }

    for trial in source['trials']:
        config = trial['config']
        baseline_checkpoint = _checkpoint_path(
            trial['baseline_checkpoint'], source_summary_path
        )
        curve_checkpoint = _checkpoint_path(
            trial['curve_checkpoint'], source_summary_path
        )
        for checkpoint in (baseline_checkpoint, curve_checkpoint):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

        set_seed(config['seed'])
        loader_args = make_experiment_args(
            cli_args, config, 'iMOE_CSR', output_summary_path.parent
        )
        _, validation_loader, test_loader, _ = DATA_LOADERS[source['dataset']](
            loader_args
        )
        if test_loader is not None:
            raise RuntimeError('Validation-only loader unexpectedly returned test data')

        baseline_model, _ = load_trained_model(
            cli_args, config, 'iMOE', baseline_checkpoint, torch_device
        )
        curve_model, _ = load_trained_model(
            cli_args, config, 'iMOE_CSR', curve_checkpoint, torch_device
        )
        baseline_pred, curve_pred, true_values = predict_pair(
            baseline_model, curve_model, validation_loader, torch_device
        )
        alpha = optimal_curve_alpha(baseline_pred, curve_pred, true_values)
        ensemble_pred = blend_predictions(baseline_pred, curve_pred, alpha)

        recomputed_trial = copy.deepcopy(trial)
        recomputed_trial.update({
            'validation_mse': float(np.mean((ensemble_pred - true_values) ** 2)),
            'baseline_validation_mse': float(
                np.mean((baseline_pred - true_values) ** 2)
            ),
            'curve_validation_mse': float(
                np.mean((curve_pred - true_values) ** 2)
            ),
            'alpha': alpha,
            'baseline_checkpoint': portable_artifact_path(baseline_checkpoint),
            'curve_checkpoint': portable_artifact_path(curve_checkpoint),
        })
        summary['trials'].append(recomputed_trial)

    summary['winner'] = select_best_trial(summary['trials'])
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(output_summary_path, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recompute validation-only dual-router summaries from checkpoints'
    )
    parser.add_argument('--source_summary', type=Path, required=True)
    parser.add_argument('--output_summary', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = recompute_validation_summary(
        args.source_summary,
        args.output_summary,
        args.device,
        args.gpu,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
