import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.evaluation.evaluate_dual_router import (
    DATA_LOADERS,
    blend_predictions,
    global_metrics,
    predict_pair,
)
from experiments.evaluation.evaluate_shared_dual_router_multiseed import _predict
from experiments.tuning.tune_validation import (
    load_trained_model,
    make_experiment_args,
    set_seed,
)
from utils.artifact_paths import (
    PROJECT_ROOT,
    portable_artifact_path,
    resolve_artifact_path,
)


def _cli_args(summary, output_dir, eval_batch_size, data_root):
    runtime = summary['runtime']
    return argparse.Namespace(
        workflow=summary['workflow'],
        dataset=summary['dataset'],
        condition=summary['condition'],
        output_dir=output_dir,
        seq_len=runtime['seq_len'],
        pred_len=runtime['pred_len'],
        dataaccess=runtime['dataaccess'],
        train_battery_count=None,
        data_root=data_root,
        batch_size=runtime['batch_size'],
        eval_batch_size=eval_batch_size,
        train_epochs=0,
        patience=0,
        device='cpu',
        gpu=0,
    )


def _rebatch(loader, batch_size):
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def _comparison(first, second):
    difference = np.abs(first - second)
    return {
        'shape': list(first.shape),
        'max_absolute_difference': float(difference.max(initial=0.0)),
        'mean_absolute_difference': float(difference.mean()),
        'allclose_atol_1e-6_rtol_1e-5': bool(
            np.allclose(first, second, atol=1e-6, rtol=1e-5)
        ),
    }


def verify_summary(summary_path, data_root, batch_a=32, batch_b=64):
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    trial = summary['trials'][0]
    config = trial['config']
    set_seed(config['seed'])
    cli_args = _cli_args(summary, summary_path.parent, batch_b, data_root)
    device = torch.device('cpu')

    loader_model = 'iMOE_SDR' if summary['workflow'] == 'iMOE_SDR' else 'iMOE_CSR'
    loader_args = make_experiment_args(
        cli_args,
        config,
        loader_model,
        summary_path.parent,
    )
    loader_args.skip_test = False
    _, _, test_loader, _ = DATA_LOADERS[summary['dataset']](loader_args)
    loader_a = _rebatch(test_loader, batch_a)
    loader_b = _rebatch(test_loader, batch_b)

    predictions = {}
    if summary['workflow'] == 'iMOE_SDR':
        checkpoint = resolve_artifact_path(trial['checkpoint'], summary_path)
        model, _ = load_trained_model(
            cli_args,
            config,
            'iMOE_SDR',
            checkpoint,
            device,
        )
        pred_a, truth_a = _predict(model, loader_a, device)
        pred_b, truth_b = _predict(model, loader_b, device)
        predictions['iMOE_SDR'] = (pred_a, pred_b)
    elif summary['workflow'] == 'dual_router':
        baseline, _ = load_trained_model(
            cli_args,
            config,
            'iMOE',
            resolve_artifact_path(trial['baseline_checkpoint'], summary_path),
            device,
        )
        curve, _ = load_trained_model(
            cli_args,
            config,
            'iMOE_CSR',
            resolve_artifact_path(trial['curve_checkpoint'], summary_path),
            device,
        )
        base_a, curve_a, truth_a = predict_pair(baseline, curve, loader_a, device)
        base_b, curve_b, truth_b = predict_pair(baseline, curve, loader_b, device)
        predictions.update({
            'iMOE': (base_a, base_b),
            'iMOE_CSR': (curve_a, curve_b),
            'dual_router': (
                blend_predictions(base_a, curve_a, trial['alpha']),
                blend_predictions(base_b, curve_b, trial['alpha']),
            ),
        })
    else:
        raise ValueError(f"Unsupported workflow: {summary['workflow']}")

    result = {
        'summary': portable_artifact_path(summary_path),
        'workflow': summary['workflow'],
        'dataset': summary['dataset'],
        'condition': summary['condition'],
        'seed': config['seed'],
        'batch_sizes': [batch_a, batch_b],
        'truth_identical': bool(np.array_equal(truth_a, truth_b)),
        'models': {},
    }
    for model_name, (pred_a, pred_b) in predictions.items():
        metric_a = global_metrics(pred_a, truth_a)
        metric_b = global_metrics(pred_b, truth_b)
        result['models'][model_name] = {
            **_comparison(pred_a, pred_b),
            'maximum_metric_difference': float(max(
                abs(metric_a[name] - metric_b[name]) for name in metric_a
            )),
        }
    result['passed'] = result['truth_identical'] and all(
        model['allclose_atol_1e-6_rtol_1e-5']
        for model in result['models'].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, action='append', required=True)
    parser.add_argument(
        '--data-root',
        type=Path,
        default=PROJECT_ROOT / 'datasets' / 'processed',
    )
    parser.add_argument('--batch-a', type=int, default=32)
    parser.add_argument('--batch-b', type=int, default=64)
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'verification' / 'inference_batch_32_vs_64.json',
    )
    args = parser.parse_args()
    results = [
        verify_summary(path, args.data_root, args.batch_a, args.batch_b)
        for path in args.summary
    ]
    payload = {
        'tolerance': {'absolute': 1e-6, 'relative': 1e-5},
        'passed': all(result['passed'] for result in results),
        'checks': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
