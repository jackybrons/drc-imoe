import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.loader import (
    LSD_trainloader,
    NCA_trainloader,
    NCM_trainloader,
    NCMNCA_trainloader,
    TPSL_trainloader,
)
from models import iMOE, iMOE_CSR
from utils.artifact_paths import portable_artifact_path


DATA_LOADERS = {
    'UL-NCA': NCA_trainloader,
    'UL-NCM': NCM_trainloader,
    'UL-NCMNCA': NCMNCA_trainloader,
    'TPSL': TPSL_trainloader,
    'LSD': LSD_trainloader,
}


def optimal_curve_alpha(baseline_pred, curve_pred, true_values):
    direction = np.asarray(curve_pred) - np.asarray(baseline_pred)
    residual = np.asarray(true_values) - np.asarray(baseline_pred)
    denominator = np.sum(direction * direction)
    if denominator == 0:
        return 0.0
    alpha = np.sum(direction * residual) / denominator
    return float(np.clip(alpha, 0.0, 1.0))


def blend_predictions(baseline_pred, curve_pred, alpha):
    return np.asarray(baseline_pred) + alpha * (
        np.asarray(curve_pred) - np.asarray(baseline_pred)
    )


def global_metrics(pred, true_values):
    pred = np.asarray(pred)
    true_values = np.asarray(true_values)
    diff = pred - true_values
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    denominator = np.where(np.abs(true_values) < 1e-8, np.nan, np.abs(true_values))
    mape = np.nanmean(np.abs(diff / denominator)) * 100.0
    centered = true_values - np.mean(true_values)
    r2 = 1.0 - np.sum(diff ** 2) / np.sum(centered ** 2)
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape_percent': float(mape),
        'r2': float(r2),
    }


def load_checkpoint(model, checkpoint_path, device, remap_legacy=False):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if remap_legacy:
        legacy_prefixes = {
            'capacity_fcs.': 'capacity_fc.',
            'relaxation_fc.': 'features_fc.',
        }
        state_dict = {
            next(
                (
                    current_prefix + key[len(legacy_prefix):]
                    for legacy_prefix, current_prefix in legacy_prefixes.items()
                    if key.startswith(legacy_prefix)
                ),
                key,
            ): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)


def predict_pair(baseline_model, curve_model, data_loader, device):
    baseline_predictions = []
    curve_predictions = []
    true_values = []
    baseline_model.eval()
    curve_model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = tuple(value.to(device) for value in inputs)
            baseline_output, _ = baseline_model(*inputs)
            curve_output, _ = curve_model(*inputs)
            baseline_predictions.append(baseline_output.cpu().numpy())
            curve_predictions.append(curve_output.cpu().numpy())
            true_values.append(targets.numpy())
    return (
        np.concatenate(baseline_predictions, axis=0),
        np.concatenate(curve_predictions, axis=0),
        np.concatenate(true_values, axis=0),
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Validation-fitted dual-router ensemble evaluation')
    parser.add_argument('--baseline_checkpoint', required=True)
    parser.add_argument('--curve_checkpoint', required=True)
    parser.add_argument('--dataset', choices=DATA_LOADERS, required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--pred_len', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_experts', type=int, default=5)
    parser.add_argument('--baseline_top_k', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=None, help='CSR top-k; defaults to num_experts')
    parser.add_argument('--curve_channels', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=10, help='baseline iMOE routing transform')
    parser.add_argument('--soc', type=int, default=20)
    parser.add_argument('--dataaccess', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)

    if args.top_k is None:
        args.top_k = args.num_experts
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    _, validation_loader, test_loader, _ = DATA_LOADERS[args.dataset](args)

    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.top_k = args.baseline_top_k
    baseline_model = iMOE.Model(baseline_args).to(device)
    curve_model = iMOE_CSR.Model(args).to(device)
    load_checkpoint(baseline_model, args.baseline_checkpoint, device, remap_legacy=True)
    load_checkpoint(curve_model, args.curve_checkpoint, device)
    baseline_model.eval()
    curve_model.eval()

    baseline_val, curve_val, true_val = predict_pair(
        baseline_model, curve_model, validation_loader, device
    )
    alpha = optimal_curve_alpha(baseline_val, curve_val, true_val)
    ensemble_val = blend_predictions(baseline_val, curve_val, alpha)

    baseline_test, curve_test, true_test = predict_pair(
        baseline_model, curve_model, test_loader, device
    )
    ensemble_test = blend_predictions(baseline_test, curve_test, alpha)
    metrics = global_metrics(ensemble_test, true_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'pred_values.npy', ensemble_test)
    np.save(output_dir / 'true_values.npy', true_test)

    metrics_payload = {
        'alpha': alpha,
        'alpha_definition': 'prediction = baseline + alpha * (curve - baseline)',
        'global_rmse': metrics['rmse'],
        'global_mae': metrics['mae'],
        'global_mape_percent': metrics['mape_percent'],
        'global_r2': metrics['r2'],
        'validation_mse_convention': 'global elementwise MSE',
        'baseline_validation_mse': float(np.mean((baseline_val - true_val) ** 2)),
        'curve_validation_mse': float(np.mean((curve_val - true_val) ** 2)),
        'ensemble_validation_mse': float(np.mean((ensemble_val - true_val) ** 2)),
        'baseline_parameters': sum(parameter.numel() for parameter in baseline_model.parameters()),
        'curve_parameters': sum(parameter.numel() for parameter in curve_model.parameters()),
        'total_parameters': sum(parameter.numel() for parameter in baseline_model.parameters())
        + sum(parameter.numel() for parameter in curve_model.parameters()),
        'dataset': args.dataset,
        'condition': args.condition,
        'prediction_shape': list(ensemble_test.shape),
        'baseline_checkpoint': portable_artifact_path(args.baseline_checkpoint),
        'curve_checkpoint': portable_artifact_path(args.curve_checkpoint),
        'baseline_top_k': args.baseline_top_k,
        'curve_top_k': args.top_k,
        'device': str(device),
    }
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as file:
        json.dump(metrics_payload, file, indent=2, ensure_ascii=False)

    horizon = np.arange(1, ensemble_test.shape[1] + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(horizon, true_test.mean(axis=0), label='True mean', linewidth=2)
    plt.plot(horizon, ensemble_test.mean(axis=0), label='Dual-router mean', linewidth=2)
    plt.xlabel('Prediction horizon')
    plt.ylabel('Discharge capacity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_prediction.png', dpi=160)
    plt.close()

    print(json.dumps(metrics_payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
