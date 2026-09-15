import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiments.evaluation.evaluate_dual_router import (
    DATA_LOADERS,
    blend_predictions,
    global_metrics,
    load_checkpoint,
    optimal_curve_alpha,
    predict_pair,
)
from experiments.core.exp_forecasting import Exp_Long_Term_Forecast1
from models import iMOE, iMOE_CSR, iMOE_SDR
from utils.artifact_paths import portable_artifact_path


MODEL_MODULES = {
    'iMOE': iMOE,
    'iMOE_CSR': iMOE_CSR,
    'iMOE_SDR': iMOE_SDR,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SDR_SEEDS = (2025, 2026, 2027)
SDR_LOCKED_VALUES = {
    'learning_rates': (1e-4,),
    'num_experts_values': (5,),
    'baseline_top_k_values': (2,),
    'csr_top_k_values': (4,),
    'router_alpha_values': (10,),
    'curve_channels_values': (4,),
    'diversity_weights': (0.5,),
    'soc_values': (20,),
}
SDR_DATASETS = {
    ('UL-NCA', 'CY25-025_1'): 64,
    ('TPSL', 'Arbitrary'): 128,
    ('LSD', 'LSD'): 64,
}

BASELINE_CONFIG_KEYS = (
    'seed',
    'hidden_dim',
    'learning_rate',
    'num_experts',
    'diverloss',
    'soc',
    'baseline_top_k',
    'router_alpha',
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_search_space(
    workflow,
    seeds,
    hidden_dims,
    learning_rates,
    num_experts_values,
    baseline_top_k_values,
    csr_top_k_values,
    router_alpha_values,
    curve_channels_values,
    diversity_weights,
    soc_values,
    fusion_mode_values=None,
    fusion_gate_bias_values=None,
):
    # Keep the published three-seed protocol callable for old summaries.  New
    # SDR searches opt in by passing explicit fusion modes and are constrained
    # by validate_sdr_locked_args to remain validation-only.
    if workflow == 'iMOE_SDR' and fusion_mode_values is None:
        locked_values = {
            'seeds': (tuple(seeds), SDR_SEEDS),
            'learning_rates': (
                tuple(learning_rates), SDR_LOCKED_VALUES['learning_rates']
            ),
            'num_experts_values': (
                tuple(num_experts_values), SDR_LOCKED_VALUES['num_experts_values']
            ),
            'baseline_top_k_values': (
                tuple(baseline_top_k_values),
                SDR_LOCKED_VALUES['baseline_top_k_values'],
            ),
            'csr_top_k_values': (
                tuple(csr_top_k_values), SDR_LOCKED_VALUES['csr_top_k_values']
            ),
            'router_alpha_values': (
                tuple(router_alpha_values),
                SDR_LOCKED_VALUES['router_alpha_values'],
            ),
            'curve_channels_values': (
                tuple(curve_channels_values),
                SDR_LOCKED_VALUES['curve_channels_values'],
            ),
            'diversity_weights': (
                tuple(diversity_weights),
                SDR_LOCKED_VALUES['diversity_weights'],
            ),
            'soc_values': (tuple(soc_values), SDR_LOCKED_VALUES['soc_values']),
        }
        for name, (actual, expected) in locked_values.items():
            if actual != expected:
                raise ValueError(f'iMOE_SDR {name} is locked to {list(expected)}')
        if tuple(hidden_dims) not in ((64,), (128,)):
            raise ValueError('iMOE_SDR hidden_dims must be [64] or [128]')

    common = itertools.product(
        seeds,
        hidden_dims,
        learning_rates,
        num_experts_values,
        diversity_weights,
        soc_values,
    )
    candidates = []
    for seed, hidden_dim, learning_rate, num_experts, diverloss, soc in common:
        base = {
            'seed': seed,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'num_experts': num_experts,
            'diverloss': diverloss,
            'soc': soc,
        }
        if workflow == 'iMOE':
            for baseline_top_k, router_alpha in itertools.product(
                baseline_top_k_values, router_alpha_values
            ):
                if baseline_top_k <= num_experts:
                    candidates.append({
                        **base,
                        'baseline_top_k': baseline_top_k,
                        'router_alpha': router_alpha,
                    })
        elif workflow == 'iMOE_CSR':
            for csr_top_k, curve_channels in itertools.product(
                csr_top_k_values, curve_channels_values
            ):
                if csr_top_k <= num_experts:
                    candidates.append({
                        **base,
                        'csr_top_k': csr_top_k,
                        'curve_channels': curve_channels,
                    })
        elif workflow in ('dual_router', 'iMOE_SDR'):
            if workflow == 'dual_router':
                fusion_modes = (None,)
                fusion_biases = (None,)
            else:
                fusion_modes = (
                    (None,) if fusion_mode_values is None else fusion_mode_values
                )
                fusion_biases = (
                    (None,)
                    if fusion_gate_bias_values is None
                    else fusion_gate_bias_values
                )
            for baseline_top_k, csr_top_k, router_alpha, curve_channels, fusion_mode, fusion_gate_bias in itertools.product(
                baseline_top_k_values,
                csr_top_k_values,
                router_alpha_values,
                curve_channels_values,
                fusion_modes,
                fusion_biases,
            ):
                if baseline_top_k <= num_experts and csr_top_k <= num_experts:
                    candidate = {
                        **base,
                        'baseline_top_k': baseline_top_k,
                        'csr_top_k': csr_top_k,
                        'router_alpha': router_alpha,
                        'curve_channels': curve_channels,
                    }
                    if fusion_mode is not None:
                        candidate['fusion_mode'] = fusion_mode
                    if fusion_gate_bias is not None:
                        candidate['fusion_gate_bias'] = fusion_gate_bias
                    candidates.append(candidate)
        else:
            raise ValueError(f'Unsupported workflow: {workflow}')
    if not candidates:
        raise ValueError('Search space is empty')
    return candidates


def choose_budgeted_trials(candidates, max_trials, search_seed):
    if max_trials < 1:
        raise ValueError('max_trials must be positive')
    if max_trials > len(candidates):
        raise ValueError(
            f'max_trials={max_trials} exceeds {len(candidates)} unique configurations'
        )
    indices = list(range(len(candidates)))
    random.Random(search_seed).shuffle(indices)
    return [candidates[index] for index in indices[:max_trials]]


def select_best_trial(trials):
    if not trials or any('validation_mse' not in trial for trial in trials):
        raise ValueError('Every completed trial must contain validation_mse')
    return min(trials, key=lambda trial: (trial['validation_mse'], trial['trial_id']))


def model_hparams(config, model_name):
    if model_name == 'iMOE':
        return {
            'top_k': config['baseline_top_k'],
            'alpha': config['router_alpha'],
            'curve_channels': 4,
        }
    if model_name == 'iMOE_CSR':
        return {
            'top_k': config['csr_top_k'],
            'alpha': 10,
            'curve_channels': config['curve_channels'],
        }
    if model_name == 'iMOE_SDR':
        return {
            'top_k': config['csr_top_k'],
            'alpha': config['router_alpha'],
            'curve_channels': config['curve_channels'],
        }
    raise ValueError(f'Unsupported model: {model_name}')


def baseline_cache_key(config):
    return tuple((key, config[key]) for key in BASELINE_CONFIG_KEYS)


def load_baseline_config(path):
    with Path(path).open('r', encoding='utf-8') as file:
        payload = json.load(file)
    if 'config' in payload:
        payload = payload['config']
    elif 'winner' in payload and 'config' in payload['winner']:
        payload = payload['winner']['config']
    missing = [key for key in BASELINE_CONFIG_KEYS if key not in payload]
    if missing:
        raise ValueError(f'Baseline config is missing keys: {missing}')
    return {key: payload[key] for key in BASELINE_CONFIG_KEYS}


def validate_baseline_reuse(config, baseline_config):
    mismatches = {
        key: (config[key], baseline_config[key])
        for key in BASELINE_CONFIG_KEYS
        if config[key] != baseline_config[key]
    }
    if mismatches:
        raise ValueError(f'Baseline checkpoint config is incompatible: {mismatches}')


def make_experiment_args(cli_args, config, model_name, checkpoint_root):
    model_values = model_hparams(config, model_name)
    return argparse.Namespace(
        model=model_name,
        dataset=cli_args.dataset,
        condition=cli_args.condition,
        seq_len=cli_args.seq_len,
        pred_len=cli_args.pred_len,
        hidden_dim=config['hidden_dim'],
        num_experts=config['num_experts'],
        top_k=model_values['top_k'],
        baseline_top_k=config.get('baseline_top_k', model_values['top_k']),
        csr_top_k=config.get('csr_top_k', model_values['top_k']),
        alpha=model_values['alpha'],
        curve_channels=model_values['curve_channels'],
        fusion_mode=config.get('fusion_mode', 'learned'),
        fusion_gate_bias=config.get('fusion_gate_bias', 0.0),
        disable_top_k=config.get('disable_top_k', False),
        disable_noisy_routing=config.get('disable_noisy_routing', False),
        diverloss=config['diverloss'],
        soc=config['soc'],
        dataaccess=cli_args.dataaccess,
        train_battery_count=getattr(cli_args, 'train_battery_count', None),
        data_root=getattr(cli_args, 'data_root', PROJECT_ROOT / 'datasets' / 'processed'),
        batch_size=cli_args.batch_size,
        eval_batch_size=getattr(cli_args, 'eval_batch_size', 64),
        learning_rate=config['learning_rate'],
        train_epochs=cli_args.train_epochs,
        patience=cli_args.patience,
        checkpoints=str(checkpoint_root),
        results_dir=str(getattr(cli_args, 'output_dir', checkpoint_root)),
        checkpoint_path=None,
        inverse='no',
        use_gpu=cli_args.device == 'cuda',
        gpu=cli_args.gpu,
        use_multi_gpu=False,
        devices=str(cli_args.gpu),
        selection_metric='prediction_mse',
        skip_test=True,
    )


def predict_single(model, data_loader, device):
    predictions = []
    true_values = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = tuple(value.to(device) for value in inputs)
            outputs, _ = model(*inputs)
            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.numpy())
    return np.concatenate(predictions), np.concatenate(true_values)


def train_model(cli_args, config, model_name, trial_root, trial_id):
    set_seed(config['seed'])
    checkpoint_trial_root = Path(cli_args.checkpoint_dir) / f'trial_{trial_id:03d}'
    experiment_args = make_experiment_args(
        cli_args,
        config,
        model_name,
        checkpoint_trial_root,
    )
    experiment_args.results_dir = str(trial_root)
    setting = f'{model_name}_trial_{trial_id:03d}'
    experiment = Exp_Long_Term_Forecast1(experiment_args)
    experiment.train(setting)
    checkpoint = (
        checkpoint_trial_root
        / model_name
        / cli_args.dataset
        / cli_args.condition
        / setting
        / 'checkpoint.pth'
    )
    return experiment.model, experiment_args, checkpoint


def validation_loader_for(cli_args, config):
    set_seed(config['seed'])
    loader_model = cli_args.workflow if cli_args.workflow in MODEL_MODULES else 'iMOE'
    loader_args = make_experiment_args(
        cli_args, config, loader_model, cli_args.output_dir
    )
    _, validation_loader, _, _ = DATA_LOADERS[cli_args.dataset](loader_args)
    return validation_loader


def run_single_trial(cli_args, config, trial_id, output_root):
    model_name = cli_args.workflow
    trial_root = output_root / 'trials' / f'trial_{trial_id:03d}'
    model, _, checkpoint = train_model(
        cli_args, config, model_name, trial_root, trial_id
    )
    validation_loader = validation_loader_for(cli_args, config)
    pred, true_values = predict_single(model, validation_loader, next(model.parameters()).device)
    return {
        'trial_id': trial_id,
        'config': config,
        'validation_mse': float(np.mean((pred - true_values) ** 2)),
        'checkpoint': portable_artifact_path(checkpoint),
    }


def recover_single_trial(cli_args, config, trial_id, output_root, checkpoint):
    """Rebuild a missing selection summary from an existing trained checkpoint."""
    if cli_args.workflow not in MODEL_MODULES:
        raise ValueError('Checkpoint-only recovery supports single-model workflows')
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    set_seed(config['seed'])
    device = torch.device(
        f'cuda:{cli_args.gpu}' if cli_args.device == 'cuda' else 'cpu'
    )
    model, _ = load_trained_model(
        cli_args,
        config,
        cli_args.workflow,
        checkpoint,
        device,
    )
    validation_loader = validation_loader_for(cli_args, config)
    prediction, true_values = predict_single(model, validation_loader, device)
    return {
        'trial_id': trial_id,
        'config': config,
        'validation_mse': float(np.mean((prediction - true_values) ** 2)),
        'checkpoint': portable_artifact_path(checkpoint),
        'recovered_from_checkpoint': True,
    }


def run_dual_trial(cli_args, config, trial_id, output_root, baseline_cache):
    trial_root = output_root / 'trials' / f'trial_{trial_id:03d}'
    cache_key = baseline_cache_key(config)
    device = torch.device(f'cuda:{cli_args.gpu}' if cli_args.device == 'cuda' else 'cpu')
    if cache_key in baseline_cache:
        baseline_checkpoint = Path(baseline_cache[cache_key]['checkpoint'])
        baseline_source = baseline_cache[cache_key]['source']
        baseline_model, _ = load_trained_model(
            cli_args, config, 'iMOE', baseline_checkpoint, device
        )
    else:
        baseline_model, _, baseline_checkpoint = train_model(
            cli_args, config, 'iMOE', trial_root, trial_id
        )
        baseline_source = 'trained'
        baseline_cache[cache_key] = {
            'checkpoint': portable_artifact_path(baseline_checkpoint),
            'source': 'cache',
        }
    curve_model, _, curve_checkpoint = train_model(
        cli_args, config, 'iMOE_CSR', trial_root, trial_id
    )
    validation_loader = validation_loader_for(cli_args, config)
    baseline_pred, curve_pred, true_values = predict_pair(
        baseline_model, curve_model, validation_loader, device
    )
    alpha = optimal_curve_alpha(baseline_pred, curve_pred, true_values)
    ensemble_pred = blend_predictions(baseline_pred, curve_pred, alpha)
    return {
        'trial_id': trial_id,
        'config': config,
        'validation_mse': float(np.mean((ensemble_pred - true_values) ** 2)),
        'baseline_validation_mse': float(np.mean((baseline_pred - true_values) ** 2)),
        'curve_validation_mse': float(np.mean((curve_pred - true_values) ** 2)),
        'alpha': alpha,
        'baseline_source': baseline_source,
        'baseline_checkpoint': portable_artifact_path(baseline_checkpoint),
        'curve_checkpoint': portable_artifact_path(curve_checkpoint),
    }


def load_trained_model(cli_args, config, model_name, checkpoint, device):
    experiment_args = make_experiment_args(cli_args, config, model_name, cli_args.output_dir)
    model = MODEL_MODULES[model_name].Model(experiment_args).to(device)
    load_checkpoint(model, checkpoint, device, remap_legacy=model_name == 'iMOE')
    model.eval()
    return model, experiment_args


def run_final_test(cli_args, winner, output_root, device):
    config = winner['config']
    set_seed(config['seed'])
    loader_model = cli_args.workflow if cli_args.workflow in MODEL_MODULES else 'iMOE'
    loader_args = make_experiment_args(cli_args, config, loader_model, output_root)
    loader_args.skip_test = False
    _, _, test_loader, _ = DATA_LOADERS[cli_args.dataset](loader_args)

    if cli_args.workflow == 'dual_router':
        baseline_model, _ = load_trained_model(
            cli_args, config, 'iMOE', winner['baseline_checkpoint'], device
        )
        curve_model, _ = load_trained_model(
            cli_args, config, 'iMOE_CSR', winner['curve_checkpoint'], device
        )
        baseline_pred, curve_pred, true_values = predict_pair(
            baseline_model, curve_model, test_loader, device
        )
        predictions = blend_predictions(baseline_pred, curve_pred, winner['alpha'])
    else:
        model, _ = load_trained_model(
            cli_args, config, cli_args.workflow, winner['checkpoint'], device
        )
        predictions, true_values = predict_single(model, test_loader, device)

    final_dir = output_root / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    np.save(final_dir / 'pred_values.npy', predictions)
    np.save(final_dir / 'true_values.npy', true_values)
    final_metrics = {
        'winner_trial_id': winner['trial_id'],
        'selection_validation_mse': winner['validation_mse'],
        'prediction_shape': list(predictions.shape),
        'global': global_metrics(predictions, true_values),
    }
    if cli_args.workflow == 'dual_router':
        final_metrics['alpha'] = winner['alpha']
    write_summary(final_dir / 'metrics.json', final_metrics)
    return final_metrics


def write_summary(path, payload):
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def validate_sdr_locked_args(args):
    if args.workflow != 'iMOE_SDR':
        return
    if not args.selection_only:
        raise ValueError('iMOE_SDR must run with --selection_only')
    runtime_values = {
        'seq_len': (args.seq_len, 50),
        'pred_len': (args.pred_len, 50),
        'dataaccess': (args.dataaccess, 100),
        'batch_size': (args.batch_size, 32),
        'eval_batch_size': (getattr(args, 'eval_batch_size', 64), 64),
        'train_epochs': (args.train_epochs, 1500),
        'patience': (args.patience, 250),
    }
    for name, (actual, expected) in runtime_values.items():
        if actual != expected:
            raise ValueError(f'iMOE_SDR {name} is locked to {expected}')
    if hasattr(args, 'fusion_mode_values'):
        if not args.fusion_mode_values:
            raise ValueError('iMOE_SDR fusion_mode_values must not be empty')
        invalid_modes = set(args.fusion_mode_values) - {'learned', 'fixed'}
        if invalid_modes:
            raise ValueError(
                f'Unsupported iMOE_SDR fusion modes: {sorted(invalid_modes)}'
            )
        if (
            not args.fusion_gate_bias_values
            or not all(np.isfinite(value) for value in args.fusion_gate_bias_values)
        ):
            raise ValueError(
                'iMOE_SDR fusion_gate_bias_values must be finite and non-empty'
            )
        return
    if tuple(args.seeds) != SDR_SEEDS:
        raise ValueError(f'iMOE_SDR seeds are locked to {list(SDR_SEEDS)}')
    if args.max_trials != len(SDR_SEEDS):
        raise ValueError('iMOE_SDR requires exactly three fixed seed runs')
    if args.search_seed != 2025:
        raise ValueError('iMOE_SDR search_seed is locked to 2025')

    dataset_key = (args.dataset, args.condition)
    if dataset_key not in SDR_DATASETS:
        raise ValueError(
            'iMOE_SDR is limited to UL-NCA/CY25-025_1, '
            'TPSL/Arbitrary, and LSD/LSD'
        )
    expected_hidden_dims = (SDR_DATASETS[dataset_key],)
    if tuple(args.hidden_dims) != expected_hidden_dims:
        raise ValueError(
            f'iMOE_SDR hidden_dims for {args.dataset}/{args.condition} '
            f'are locked to {list(expected_hidden_dims)}'
        )
    for name, expected in SDR_LOCKED_VALUES.items():
        actual = tuple(getattr(args, name))
        if actual != expected:
            raise ValueError(f'iMOE_SDR {name} is locked to {list(expected)}')
    if args.device != 'cuda':
        raise ValueError('iMOE_SDR training device is locked to cuda:0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fixed-budget hyperparameter tuning with validation-only selection'
    )
    parser.add_argument(
        '--workflow',
        choices=('iMOE', 'iMOE_CSR', 'dual_router', 'iMOE_SDR'),
        required=True,
    )
    parser.add_argument('--dataset', choices=DATA_LOADERS, required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument(
        '--checkpoint_dir',
        type=Path,
        default=PROJECT_ROOT / 'checkpoints' / 'tuning',
    )
    parser.add_argument(
        '--recover_checkpoint',
        type=Path,
        help='recompute validation summary from an existing single-model checkpoint',
    )
    parser.add_argument('--max_trials', type=int, required=True)
    parser.add_argument('--search_seed', type=int, default=2025)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025])
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64])
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[1e-4])
    parser.add_argument('--num_experts_values', type=int, nargs='+', default=[5])
    parser.add_argument('--baseline_top_k_values', type=int, nargs='+', default=[2])
    parser.add_argument('--csr_top_k_values', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--router_alpha_values', type=int, nargs='+', default=[10])
    parser.add_argument('--curve_channels_values', type=int, nargs='+', default=[4])
    parser.add_argument(
        '--fusion_mode_values',
        choices=('learned', 'fixed'),
        nargs='+',
        default=['learned'],
    )
    parser.add_argument(
        '--fusion_gate_bias_values',
        type=float,
        nargs='+',
        default=[0.0],
    )
    parser.add_argument('--disable_top_k', action='store_true')
    parser.add_argument('--disable_noisy_routing', action='store_true')
    parser.add_argument('--diversity_weights', type=float, nargs='+', default=[0.5])
    parser.add_argument('--soc_values', type=int, nargs='+', default=[20])
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--pred_len', type=int, default=50)
    parser.add_argument('--dataaccess', type=int, default=100)
    parser.add_argument('--train_battery_count', type=int)
    parser.add_argument(
        '--data_root',
        type=Path,
        default=PROJECT_ROOT / 'datasets' / 'processed',
    )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--train_epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--baseline_checkpoint', type=Path)
    parser.add_argument('--baseline_config', type=Path)
    parser.add_argument('--selection_only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    validate_sdr_locked_args(args)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    candidates = build_search_space(
        args.workflow,
        args.seeds,
        args.hidden_dims,
        args.learning_rates,
        args.num_experts_values,
        args.baseline_top_k_values,
        args.csr_top_k_values,
        args.router_alpha_values,
        args.curve_channels_values,
        args.diversity_weights,
        args.soc_values,
        args.fusion_mode_values,
        args.fusion_gate_bias_values,
    )
    for config in candidates:
        if args.disable_top_k:
            config['disable_top_k'] = True
        if args.disable_noisy_routing:
            config['disable_noisy_routing'] = True
    selected_configs = choose_budgeted_trials(
        candidates,
        args.max_trials,
        args.search_seed,
    )
    if bool(args.baseline_checkpoint) != bool(args.baseline_config):
        raise ValueError('--baseline_checkpoint and --baseline_config must be provided together')
    if args.baseline_checkpoint and args.workflow != 'dual_router':
        raise ValueError('Baseline checkpoint reuse is only valid for dual_router')
    if args.recover_checkpoint:
        if not args.selection_only or args.max_trials != 1:
            raise ValueError(
                '--recover_checkpoint requires --selection_only and --max_trials 1'
            )
        if args.workflow not in MODEL_MODULES:
            raise ValueError('--recover_checkpoint requires a single-model workflow')
    baseline_cache = {}
    if args.baseline_checkpoint:
        baseline_config = load_baseline_config(args.baseline_config)
        for config in selected_configs:
            validate_baseline_reuse(config, baseline_config)
        if not args.baseline_checkpoint.is_file():
            raise FileNotFoundError(args.baseline_checkpoint)
        baseline_cache[baseline_cache_key(baseline_config)] = {
            'checkpoint': portable_artifact_path(args.baseline_checkpoint),
            'source': 'external',
        }
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / 'search_summary.json'
    summary = {
        'protocol': {
            'selection_split': 'validation',
            'selection_metric': 'global_mse',
            'test_policy': (
                'not loaded or evaluated in selection-only mode'
                if args.selection_only
                else 'evaluated once after winner selection'
            ),
        },
        'budget': {
            'max_trials': args.max_trials,
            'max_training_jobs': args.max_trials * (2 if args.workflow == 'dual_router' else 1),
            'max_epochs_per_training': args.train_epochs,
            'patience': args.patience,
        },
        'search': {
            'search_seed': args.search_seed,
            'candidate_count': len(candidates),
            'seeds': args.seeds,
            'hidden_dims': args.hidden_dims,
            'learning_rates': args.learning_rates,
            'num_experts_values': args.num_experts_values,
            'baseline_top_k_values': args.baseline_top_k_values,
            'csr_top_k_values': args.csr_top_k_values,
            'router_alpha_values': args.router_alpha_values,
            'curve_channels_values': args.curve_channels_values,
            'fusion_mode_values': args.fusion_mode_values,
            'fusion_gate_bias_values': args.fusion_gate_bias_values,
            'disable_top_k': args.disable_top_k,
            'disable_noisy_routing': args.disable_noisy_routing,
            'diversity_weights': args.diversity_weights,
            'soc_values': args.soc_values,
        },
        'runtime': {
            'device': args.device,
            'gpu': args.gpu,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
            'dataaccess': args.dataaccess,
            'train_battery_count': args.train_battery_count,
            'data_root': portable_artifact_path(args.data_root),
        },
        'workflow': args.workflow,
        'dataset': args.dataset,
        'condition': args.condition,
        'trials': [],
    }

    for trial_id, config in enumerate(selected_configs):
        if args.recover_checkpoint:
            trial = recover_single_trial(
                args,
                config,
                trial_id,
                output_root,
                args.recover_checkpoint,
            )
        elif args.workflow == 'dual_router':
            trial = run_dual_trial(
                args, config, trial_id, output_root, baseline_cache
            )
        else:
            trial = run_single_trial(args, config, trial_id, output_root)
        summary['trials'].append(trial)
        write_summary(summary_path, summary)

    recovered_trials = sum(
        trial.get('recovered_from_checkpoint', False)
        for trial in summary['trials']
    )
    summary['budget']['actual_training_jobs'] = (
        len(summary['trials'])
        + sum(trial['baseline_source'] == 'trained' for trial in summary['trials'])
        if args.workflow == 'dual_router'
        else len(summary['trials']) - recovered_trials
    )
    configs_without_seed = [
        {
            key: value
            for key, value in trial['config'].items()
            if key != 'seed'
        }
        for trial in summary['trials']
    ]
    if (
        args.workflow == 'iMOE_SDR'
        and all(config == configs_without_seed[0] for config in configs_without_seed)
    ):
        summary['locked_config_without_seed'] = configs_without_seed[0]
    else:
        winner = select_best_trial(summary['trials'])
        summary['winner'] = winner
    if not args.selection_only:
        device = torch.device(f'cuda:{args.gpu}' if args.device == 'cuda' else 'cpu')
        summary['final_test'] = run_final_test(args, winner, output_root, device)
    write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
