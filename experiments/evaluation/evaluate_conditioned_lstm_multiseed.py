import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import DATA_LOADERS, global_metrics
from experiments.core.exp_forecasting import Exp_Long_Term_Forecast1
from experiments.tuning.tune_validation import set_seed
from utils.artifact_paths import portable_artifact_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_SEEDS = (2025, 2026, 2027)
METRIC_NAMES = ('rmse', 'mae', 'mape_percent', 'r2')
SUPPORTED_MODELS = ('ConditionedLSTM', 'ConditionedMLP', 'Informer', 'PATCHTST')
SUPPORTED_DATASETS = {
    ('UL-NCA', 'CY25-025_1'): 64,
    ('TPSL', 'Arbitrary'): 128,
    ('LSD', 'LSD'): 64,
}
LOCKED_RUNTIME = {
    'learning_rate': 1e-4,
    'batch_size': 32,
    'dataaccess': 100,
    'soc': 20,
    'seq_len': 50,
    'pred_len': 50,
    'train_epochs': 1500,
    'patience': 250,
    'd_model': 64,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 64,
    'dropout': 0.2,
    'patch_size': 2,
}


def locked_hidden_dim(dataset, condition):
    try:
        return SUPPORTED_DATASETS[(dataset, condition)]
    except KeyError as error:
        supported = ', '.join(f'{name}/{split}' for name, split in SUPPORTED_DATASETS)
        raise ValueError(
            f'Unsupported dataset/condition: {dataset}/{condition}; supported: {supported}'
        ) from error


def make_experiment_args(
    dataset,
    condition,
    output_dir,
    seed,
    device,
    gpu,
    skip_test,
    model='ConditionedLSTM',
    results_dir=None,
    data_root=PROJECT_ROOT / 'datasets' / 'processed',
    eval_batch_size=64,
):
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f'Unsupported baseline model: {model}; supported: {", ".join(SUPPORTED_MODELS)}'
        )
    return argparse.Namespace(
        model=model,
        dataset=dataset,
        condition=condition,
        seq_len=LOCKED_RUNTIME['seq_len'],
        pred_len=LOCKED_RUNTIME['pred_len'],
        hidden_dim=locked_hidden_dim(dataset, condition),
        d_model=LOCKED_RUNTIME['d_model'],
        e_layers=LOCKED_RUNTIME['e_layers'],
        d_layers=LOCKED_RUNTIME['d_layers'],
        d_ff=LOCKED_RUNTIME['d_ff'],
        dropout=LOCKED_RUNTIME['dropout'],
        patch_size=LOCKED_RUNTIME['patch_size'],
        num_experts=1,
        top_k=1,
        baseline_top_k=1,
        csr_top_k=1,
        alpha=10,
        diverloss=0.0,
        soc=LOCKED_RUNTIME['soc'],
        dataaccess=LOCKED_RUNTIME['dataaccess'],
        batch_size=LOCKED_RUNTIME['batch_size'],
        eval_batch_size=eval_batch_size,
        data_root=Path(data_root),
        train_battery_count=None,
        learning_rate=LOCKED_RUNTIME['learning_rate'],
        train_epochs=LOCKED_RUNTIME['train_epochs'],
        patience=LOCKED_RUNTIME['patience'],
        checkpoints=str(output_dir),
        results_dir=str(output_dir if results_dir is None else results_dir),
        checkpoint_path=None,
        inverse='no',
        use_gpu=device == 'cuda',
        gpu=gpu,
        use_multi_gpu=False,
        devices=str(gpu),
        selection_metric='prediction_mse',
        skip_test=skip_test,
        seed=seed,
    )


def predict(model, data_loader, device):
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


def aggregate_seed_metrics(seed_results):
    aggregate = {'std_definition': 'population standard deviation (ddof=0)', 'metrics': {}}
    for metric_name in METRIC_NAMES:
        values = np.asarray(
            [result['metrics'][metric_name] for result in seed_results],
            dtype=np.float64,
        )
        aggregate['metrics'][metric_name] = {
            'mean': float(values.mean()),
            'std': float(values.std(ddof=0)),
            'by_seed': {
                str(result['seed']): float(value)
                for result, value in zip(seed_results, values)
            },
        }
    return aggregate


def write_json(path, payload):
    with Path(path).open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def run_locked_multiseed(
    dataset,
    condition,
    output_dir,
    device='cuda',
    gpu=0,
    model='ConditionedLSTM',
    resume_existing=False,
    checkpoint_dir=None,
    data_root=PROJECT_ROOT / 'datasets' / 'processed',
    eval_batch_size=64,
):
    hidden_dim = locked_hidden_dim(dataset, condition)
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f'Unsupported baseline model: {model}; supported: {", ".join(SUPPORTED_MODELS)}'
        )
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    output_dir = Path(output_dir).resolve()
    selection_root = (
        PROJECT_ROOT / 'checkpoints' / 'main' / 'baselines'
        if checkpoint_dir is None
        else Path(checkpoint_dir).resolve()
    )
    selection_results = output_dir / 'selection'
    output_dir.mkdir(parents=True, exist_ok=True)

    locked_trials = []
    for seed in LOCKED_SEEDS:
        set_seed(seed)
        training_args = make_experiment_args(
            dataset,
            condition,
            selection_root,
            seed,
            device,
            gpu,
            skip_test=True,
            model=model,
            results_dir=selection_results,
            data_root=data_root,
            eval_batch_size=eval_batch_size,
        )
        setting = f'{model}_seed_{seed}'
        checkpoint = (
            selection_root
            / training_args.model
            / dataset
            / condition
            / setting
            / 'checkpoint.pth'
        ).resolve()
        experiment = Exp_Long_Term_Forecast1(training_args)
        if resume_existing and checkpoint.is_file():
            print(f'[{model}][seed {seed}] resuming existing checkpoint: {checkpoint}')
            experiment._load_checkpoint(checkpoint)
        else:
            print(f'[{model}][seed {seed}] training')
            experiment.train(setting)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        _, validation_loader, test_loader, _ = DATA_LOADERS[dataset](training_args)
        if test_loader is not None:
            raise RuntimeError('Selection stage loaded the test split')
        validation_prediction, validation_true = predict(
            experiment.model, validation_loader, experiment.device
        )
        validation_mse = float(np.mean((validation_prediction - validation_true) ** 2))
        locked_trials.append({
            'seed': seed,
            'checkpoint': portable_artifact_path(checkpoint),
            'selection_validation_mse': validation_mse,
        })

    selection_summary = {
        'protocol': {
            'configuration': f'fixed {model} configuration; no hyperparameter search',
            'checkpoint_selection': 'per-seed validation prediction MSE',
            'test_policy': 'test split not loaded during checkpoint selection',
            'all_checkpoints_locked_before_test': True,
        },
        'dataset': dataset,
        'condition': condition,
        'model': model,
        'declared_seeds': list(LOCKED_SEEDS),
        'runtime': {**LOCKED_RUNTIME, 'hidden_dim': hidden_dim},
        'trials': locked_trials,
    }
    write_json(output_dir / 'selection_summary.json', selection_summary)

    seed_results = []
    for trial in locked_trials:
        seed = trial['seed']
        set_seed(seed)
        evaluation_args = make_experiment_args(
            dataset,
            condition,
            selection_root,
            seed,
            device,
            gpu,
            skip_test=False,
            model=model,
            results_dir=output_dir,
            data_root=data_root,
            eval_batch_size=eval_batch_size,
        )
        _, _, test_loader, _ = DATA_LOADERS[dataset](evaluation_args)
        experiment = Exp_Long_Term_Forecast1(evaluation_args)
        experiment._load_checkpoint(trial['checkpoint'])
        prediction, true_values = predict(experiment.model, test_loader, experiment.device)
        if prediction.shape != true_values.shape:
            raise ValueError(f'Prediction shape mismatch for seed {seed}')
        if not np.isfinite(prediction).all() or not np.isfinite(true_values).all():
            raise ValueError(f'Non-finite test values for seed {seed}')

        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        pred_path = (seed_dir / 'pred_values.npy').resolve()
        true_path = (seed_dir / 'true_values.npy').resolve()
        np.save(pred_path, prediction)
        np.save(true_path, true_values)

        parameter_counts = {
            'total': sum(parameter.numel() for parameter in experiment.model.parameters()),
            'trainable': sum(
                parameter.numel()
                for parameter in experiment.model.parameters()
                if parameter.requires_grad
            ),
        }
        seed_result = {
            'seed': seed,
            'checkpoint': trial['checkpoint'],
            'selection_validation_mse': trial['selection_validation_mse'],
            'prediction_shape': list(prediction.shape),
            'parameters': parameter_counts,
            'metrics': global_metrics(prediction, true_values),
            'artifacts': {
                'pred_values': portable_artifact_path(pred_path),
                'true_values': portable_artifact_path(true_path),
            },
        }
        write_json(seed_dir / 'metrics.json', seed_result)
        seed_results.append(seed_result)

    parameter_values = {
        (result['parameters']['total'], result['parameters']['trainable'])
        for result in seed_results
    }
    if len(parameter_values) != 1:
        raise ValueError('Parameter counts differ across locked seeds')
    total_parameters, trainable_parameters = parameter_values.pop()
    aggregate = aggregate_seed_metrics(seed_results)
    write_json(output_dir / 'aggregate.json', aggregate)

    summary = {
        'protocol': {
            'configuration': f'fixed {model} configuration; no hyperparameter search',
            'checkpoint_selection': 'per-seed validation prediction MSE',
            'selection_test_policy': 'test split not loaded during checkpoint selection',
            'evaluation_test_policy': 'each locked seed evaluated once after all checkpoints were locked',
        },
        'dataset': dataset,
        'condition': condition,
        'model': model,
        'device': str(torch.device(f'cuda:{gpu}' if device == 'cuda' else 'cpu')),
        'declared_seeds': list(LOCKED_SEEDS),
        'runtime': {**LOCKED_RUNTIME, 'hidden_dim': hidden_dim},
        'parameters': {
            'total': total_parameters,
            'trainable': trainable_parameters,
        },
        'selection_summary': portable_artifact_path(output_dir / 'selection_summary.json'),
        'seeds': seed_results,
        'aggregate': aggregate,
    }
    write_json(output_dir / 'summary.json', summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Locked three-seed training and evaluation for baseline models'
    )
    parser.add_argument('--model', choices=SUPPORTED_MODELS, default='ConditionedLSTM')
    parser.add_argument('--dataset', choices=('UL-NCA', 'TPSL', 'LSD'), required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument(
        '--checkpoint_dir',
        type=Path,
        default=PROJECT_ROOT / 'checkpoints' / 'main' / 'baselines',
    )
    parser.add_argument(
        '--data_root',
        type=Path,
        default=PROJECT_ROOT / 'datasets' / 'processed',
    )
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--resume_existing',
        action='store_true',
        help='reuse an existing per-seed checkpoint instead of training that seed',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_locked_multiseed(
        args.dataset,
        args.condition,
        args.output_dir,
        args.device,
        args.gpu,
        args.model,
        args.resume_existing,
        args.checkpoint_dir,
        args.data_root,
        args.eval_batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
