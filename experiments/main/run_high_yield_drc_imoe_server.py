import argparse
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

from experiments.main.run_drc_imoe_server import PROJECT_ROOT, _run_stage, locked_configs


DATASET_SETTINGS = {
    'LSD': {
        'condition': 'LSD',
        'csr_top_ks': (3, 4),
        'output_name': 'lsd',
        'batch_size': 32,
    },
    'UL-NCA': {
        'condition': 'CY25-025_1',
        'csr_top_ks': (3,),
        'output_name': 'ul_nca',
        'batch_size': 32,
    },
}
LEARNING_RATES = (1e-4, 3e-4)
DIVERSITY_WEIGHTS = (0.05, 0.1)


def search_configs(args):
    configs = []
    csr_top_ks = DATASET_SETTINGS[args.dataset]['csr_top_ks']
    for learning_rate, diverloss, csr_top_k in itertools.product(
        LEARNING_RATES, DIVERSITY_WEIGHTS, csr_top_ks
    ):
        configs.append({
            'seed': args.search_seed,
            'hidden_dim': 64,
            'learning_rate': learning_rate,
            'num_experts': 5,
            'diverloss': diverloss,
            'soc': 20,
            'baseline_top_k': 2,
            'csr_top_k': csr_top_k,
            'router_alpha': 10,
            'curve_channels': 4,
        })
    return configs


def _evaluation_command(args, locked_summary_path, final_dir):
    return [
        sys.executable,
        '-m',
        'experiments.evaluation.evaluate_locked_multiseed',
        '--search_summary', str(locked_summary_path),
        '--output_dir', str(final_dir),
        '--device', args.device,
        '--gpu', str(args.gpus[0]),
        '--eval_batch_size', str(args.eval_batch_size),
        '--data_root', str(args.data_root),
    ]


def run_dataset(args, runner=subprocess.run):
    output_name = DATASET_SETTINGS[args.dataset]['output_name']
    dataset_root = Path(args.output_root).resolve() / output_name
    search_dir = dataset_root / 'search'
    locked_dir = dataset_root / 'locked_seeds'
    final_dir = dataset_root / 'final_test'

    candidates = search_configs(args)
    search_summary = _run_stage(args, search_dir, candidates, runner)

    fixed_configs = locked_configs(args, search_summary['winner'])
    locked_summary = _run_stage(args, locked_dir, fixed_configs, runner)
    locked_summary['locked_config_without_seed'] = {
        key: value for key, value in fixed_configs[0].items() if key != 'seed'
    }
    locked_summary_path = locked_dir / 'search_summary.json'
    with locked_summary_path.open('w', encoding='utf-8') as file:
        json.dump(locked_summary, file, indent=2, ensure_ascii=False)

    runner(
        _evaluation_command(args, locked_summary_path, final_dir),
        cwd=PROJECT_ROOT,
        check=True,
    )
    return {
        'search_summary': search_dir / 'search_summary.json',
        'locked_summary': locked_summary_path,
        'final_summary': final_dir / 'summary.json',
    }


def run_pipeline(args, runner=subprocess.run):
    datasets = ('LSD', 'UL-NCA') if args.only == 'all' else (args.only,)
    outputs = {}
    for dataset in datasets:
        dataset_args = copy.copy(args)
        dataset_args.dataset = dataset
        dataset_args.condition = DATASET_SETTINGS[dataset]['condition']
        dataset_args.batch_size = (
            args.batch_size
            if args.batch_size is not None
            else DATASET_SETTINGS[dataset]['batch_size']
        )
        outputs[dataset] = run_dataset(dataset_args, runner)
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Run the high-yield DRC-iMOE protocol: LSD search/locked evaluation, '
            'then UL-NCA search/locked evaluation'
        )
    )
    parser.add_argument('--only', choices=('all', 'LSD', 'UL-NCA'), default='all')
    parser.add_argument(
        '--output_root',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'tuning' / 'high_yield',
    )
    parser.add_argument(
        '--checkpoint_root',
        type=Path,
        default=PROJECT_ROOT / 'checkpoints' / 'tuning' / 'high_yield',
    )
    parser.add_argument('--search_seed', type=int, default=2025)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026, 2027])
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--pred_len', type=int, default=50)
    parser.add_argument('--dataaccess', type=int, default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        help='override the locked training batch size (default: 32)',
    )
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument(
        '--data_root',
        type=Path,
        default=PROJECT_ROOT / 'datasets' / 'processed',
    )
    parser.add_argument('--train_epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must be a non-empty list of unique values')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error('--gpus must be a non-empty list of unique values')
    if args.batch_size not in (None, 32):
        parser.error('--batch_size is locked to 32')
    if args.eval_batch_size != 64:
        parser.error('--eval_batch_size is locked to 64')
    if args.train_epochs != 1500 or args.patience != 250:
        parser.error('training is locked to 1500 epochs and patience 250')
    return args


def main():
    outputs = run_pipeline(parse_args())
    print(json.dumps(
        {
            dataset: {name: str(path) for name, path in paths.items()}
            for dataset, paths in outputs.items()
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == '__main__':
    main()
