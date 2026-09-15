import argparse
import copy
import itertools
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils.artifact_paths import resolve_artifact_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_ONLY_POLICY = 'not loaded or evaluated in selection-only mode'
FINAL_SEEDS = (2025, 2026, 2027)
DATASET_DEFAULTS = {
    'TPSL': {'condition': 'Arbitrary', 'hidden_dim': 128},
    'LSD': {'condition': 'LSD', 'hidden_dim': 64},
    'UL-NCA': {'condition': 'CY25-025_1', 'hidden_dim': 64},
}


def _append_value(command, name, value):
    command.extend([name, str(value)])


def tuning_command(args, output_dir, checkpoint_dir, config, gpu):
    workflow = 'iMOE_SDR' if 'fusion_mode' in config else 'dual_router'
    command = [
        sys.executable,
        '-m',
        'experiments.tuning.tune_validation',
        '--workflow', workflow,
        '--dataset', args.dataset,
        '--condition', args.condition,
        '--output_dir', str(output_dir),
        '--checkpoint_dir', str(checkpoint_dir),
        '--max_trials', '1',
        '--search_seed', str(args.search_seed),
    ]
    options = (
        ('--seeds', 'seed'),
        ('--hidden_dims', 'hidden_dim'),
        ('--learning_rates', 'learning_rate'),
        ('--num_experts_values', 'num_experts'),
        ('--baseline_top_k_values', 'baseline_top_k'),
        ('--csr_top_k_values', 'csr_top_k'),
        ('--router_alpha_values', 'router_alpha'),
        ('--curve_channels_values', 'curve_channels'),
        ('--fusion_mode_values', 'fusion_mode'),
        ('--fusion_gate_bias_values', 'fusion_gate_bias'),
        ('--diversity_weights', 'diverloss'),
        ('--soc_values', 'soc'),
    )
    for option, key in options:
        if key in config:
            _append_value(command, option, config[key])
    for option, key in (
        ('--disable_top_k', 'disable_top_k'),
        ('--disable_noisy_routing', 'disable_noisy_routing'),
    ):
        if config.get(key):
            command.append(option)
    command.extend([
        '--seq_len', str(args.seq_len),
        '--pred_len', str(args.pred_len),
        '--dataaccess', str(args.dataaccess),
        '--batch_size', str(args.batch_size),
        '--eval_batch_size', str(args.eval_batch_size),
        '--data_root', str(args.data_root),
        '--train_epochs', str(args.train_epochs),
        '--patience', str(args.patience),
        '--device', args.device,
        '--gpu', str(gpu),
    ])
    command.append('--selection_only')
    return command


def _base_config(args, seed=None):
    return {
        'seed': args.search_seed if seed is None else seed,
        'hidden_dim': args.hidden_dim,
        'learning_rate': None,
        'num_experts': 5,
        'diverloss': None,
        'soc': 20,
        'baseline_top_k': 2,
        'csr_top_k': 4,
        'router_alpha': 10,
        'curve_channels': 4,
        'fusion_mode': 'learned',
        'fusion_gate_bias': 0.0,
    }


def stage1_configs(args):
    configs = []
    for learning_rate, diverloss, baseline_top_k in itertools.product(
        (1e-4, 3e-4), (0.05, 0.1, 0.5), (1, 2)
    ):
        config = _base_config(args)
        config.update(
            learning_rate=learning_rate,
            diverloss=diverloss,
            baseline_top_k=baseline_top_k,
        )
        configs.append(config)
    return configs


def stage2_configs(stage1_winner):
    configs = []
    for csr_top_k, curve_channels, fusion_gate_bias in itertools.product(
        (3, 4, 5), (4, 8), (-1.0, 0.0, 1.0)
    ):
        config = copy.deepcopy(stage1_winner['config'])
        config.update(
            csr_top_k=csr_top_k,
            curve_channels=curve_channels,
            fusion_mode='learned',
            fusion_gate_bias=fusion_gate_bias,
        )
        configs.append(config)
    return configs


def locked_configs(args, stage2_winner):
    configs = []
    for seed in args.seeds:
        config = copy.deepcopy(stage2_winner['config'])
        config['seed'] = seed
        configs.append(config)
    return configs


def _run_gpu_queue(jobs, runner):
    for job in jobs:
        runner(job['command'], cwd=PROJECT_ROOT, check=True)


def run_parallel_jobs(jobs, gpus, runner=subprocess.run):
    """Run one serial queue per GPU while different GPU queues run in parallel."""
    queues = {gpu: [] for gpu in gpus}
    for job in jobs:
        queues[job['gpu']].append(job)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(_run_gpu_queue, queues[gpu], runner)
            for gpu in gpus
            if queues[gpu]
        ]
        for future in futures:
            future.result()


def _load_selection_summary(path, expected_workflow='iMOE_SDR'):
    with Path(path).open('r', encoding='utf-8') as file:
        summary = json.load(file)
    if summary.get('workflow') != expected_workflow:
        raise ValueError(f'{path} is not an {expected_workflow} search summary')
    if 'final_test' in summary or summary.get('protocol', {}).get('test_policy') != (
        SELECTION_ONLY_POLICY
    ):
        raise ValueError(f'{path} is not a selection-only search summary')
    if len(summary.get('trials', [])) != 1:
        raise ValueError(f'{path} must contain exactly one completed trial')
    return summary


def _search_values(configs):
    mapping = {
        'seed': 'seeds',
        'hidden_dim': 'hidden_dims',
        'learning_rate': 'learning_rates',
        'num_experts': 'num_experts_values',
        'baseline_top_k': 'baseline_top_k_values',
        'csr_top_k': 'csr_top_k_values',
        'router_alpha': 'router_alpha_values',
        'curve_channels': 'curve_channels_values',
        'fusion_mode': 'fusion_mode_values',
        'fusion_gate_bias': 'fusion_gate_bias_values',
        'disable_top_k': 'disable_top_k_values',
        'disable_noisy_routing': 'disable_noisy_routing_values',
        'diverloss': 'diversity_weights',
        'soc': 'soc_values',
    }
    return {
        search_key: list(dict.fromkeys(config[key] for config in configs))
        for key, search_key in mapping.items()
        if all(key in config for config in configs)
    }


def merge_selection_summaries(summary_paths, output_path, expected_configs, gpus,
                              search_seed):
    expected_workflow = (
        'iMOE_SDR' if 'fusion_mode' in expected_configs[0] else 'dual_router'
    )
    child_summaries = [
        _load_selection_summary(path, expected_workflow)
        for path in summary_paths
    ]
    if len(child_summaries) != len(expected_configs):
        raise ValueError('Completed trial count differs from requested configuration count')
    trials = []
    for trial_id, (child, expected_config) in enumerate(
        zip(child_summaries, expected_configs)
    ):
        trial = copy.deepcopy(child['trials'][0])
        if trial.get('config') != expected_config:
            raise ValueError(
                f'Trial {trial_id} config differs from its requested configuration'
            )
        trial['trial_id'] = trial_id
        trials.append(trial)

    summary = copy.deepcopy(child_summaries[0])
    summary['trials'] = trials
    summary['search'] = {
        'search_seed': search_seed,
        'candidate_count': len(expected_configs),
        **_search_values(expected_configs),
    }
    summary['runtime']['gpu'] = gpus[0]
    summary['runtime']['gpus'] = list(gpus)
    summary['budget']['max_trials'] = len(expected_configs)
    training_jobs = (
        len(trials)
        + sum(trial.get('baseline_source') == 'trained' for trial in trials)
    )
    summary['budget']['max_training_jobs'] = training_jobs
    summary['budget']['actual_training_jobs'] = training_jobs
    summary['winner'] = min(
        trials, key=lambda trial: (trial['validation_mse'], trial['trial_id'])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


def _completed_trial(path, expected_config):
    if not path.is_file():
        return False
    try:
        expected_workflow = (
            'iMOE_SDR' if 'fusion_mode' in expected_config else 'dual_router'
        )
        summary = _load_selection_summary(path, expected_workflow)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    trial = summary['trials'][0]
    def checkpoint_exists(value):
        checkpoint = resolve_artifact_path(value, path)
        return checkpoint.is_file()

    checkpoint_complete = (
        'checkpoint' in trial and checkpoint_exists(trial['checkpoint'])
        if expected_workflow == 'iMOE_SDR'
        else (
            'baseline_checkpoint' in trial
            and 'curve_checkpoint' in trial
            and checkpoint_exists(trial['baseline_checkpoint'])
            and checkpoint_exists(trial['curve_checkpoint'])
        )
    )
    return (
        trial.get('config') == expected_config
        and isinstance(trial.get('validation_mse'), (int, float))
        and math.isfinite(trial['validation_mse'])
        and checkpoint_complete
    )


def _run_stage(args, stage_dir, configs, runner):
    stage_path = Path(stage_dir).resolve()
    try:
        checkpoint_stage = stage_path.relative_to(Path(args.output_root).resolve())
    except ValueError:
        checkpoint_stage = Path(stage_path.name)
    jobs = []
    for trial_id, config in enumerate(configs):
        output_dir = stage_dir / 'parallel_trials' / f'trial_{trial_id:03d}'
        checkpoint_dir = (
            Path(args.checkpoint_root)
            / checkpoint_stage
            / 'parallel_trials'
            / f'trial_{trial_id:03d}'
            / 'trials'
        )
        recovery_checkpoint = (
            checkpoint_dir
            / 'trial_000'
            / 'iMOE_SDR'
            / args.dataset
            / args.condition
            / 'iMOE_SDR_trial_000'
            / 'checkpoint.pth'
        )
        gpu = args.gpus[trial_id % len(args.gpus)]
        job = {
            'gpu': gpu,
            'output_dir': output_dir,
            'recovery_checkpoint': recovery_checkpoint,
            'command': tuning_command(
                args,
                output_dir,
                checkpoint_dir,
                config,
                gpu,
            ),
        }
        jobs.append(job)
    pending_jobs = []
    for job, config in zip(jobs, configs):
        summary_path = job['output_dir'] / 'search_summary.json'
        if _completed_trial(summary_path, config):
            continue
        if (
            not summary_path.exists()
            and 'fusion_mode' in config
            and job['recovery_checkpoint'].is_file()
        ):
            job['command'].extend((
                '--recover_checkpoint',
                str(job['recovery_checkpoint']),
            ))
        pending_jobs.append(job)
    run_parallel_jobs(pending_jobs, args.gpus, runner)
    summary_paths = [job['output_dir'] / 'search_summary.json' for job in jobs]
    return merge_selection_summaries(
        summary_paths,
        stage_dir / 'search_summary.json',
        configs,
        args.gpus,
        args.search_seed,
    )


def _completed_final_evaluation(path, source_summary, reference_summary=False):
    if not path.is_file():
        return False
    try:
        with path.open('r', encoding='utf-8') as file:
            summary = json.load(file)
        with Path(source_summary).open('r', encoding='utf-8') as file:
            source = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    if resolve_artifact_path(
        summary.get('source_search_summary', ''),
        path,
    ) != Path(source_summary).resolve():
        return False
    if tuple(summary.get('declared_seeds', ())) != FINAL_SEEDS:
        return False
    if summary.get('locked_config_without_seed') != source.get(
        'locked_config_without_seed'
    ):
        return False
    if reference_summary is None:
        return summary.get('reference_summary') is None
    if reference_summary is not False:
        return resolve_artifact_path(
            summary.get('reference_summary', ''),
            path,
        ) == Path(reference_summary).resolve()
    return True


def run_pipeline(args, runner=subprocess.run):
    output_root = Path(args.output_root).resolve()
    stage1_dir = output_root / f'{args.dataset}_stage1'
    stage2_dir = output_root / f'{args.dataset}_stage2'
    locked_dir = output_root / f'{args.dataset}_locked_seeds'
    final_dir = output_root / f'{args.dataset}_final'
    ablation_root = output_root / f'{args.dataset}_ablations'
    full_fusion_root = output_root / f'{args.dataset}_full_fusion'

    first_configs = stage1_configs(args)
    stage1_summary = _run_stage(args, stage1_dir, first_configs, runner)
    stage1_winner = stage1_summary['winner']

    second_configs = stage2_configs(stage1_winner)
    stage2_summary = _run_stage(args, stage2_dir, second_configs, runner)
    stage2_winner = stage2_summary['winner']

    fixed_configs = locked_configs(args, stage2_winner)
    locked_summary = _run_stage(args, locked_dir, fixed_configs, runner)
    locked_summary.pop('winner', None)
    locked_summary['locked_config_without_seed'] = {
        key: value for key, value in fixed_configs[0].items() if key != 'seed'
    }
    with (locked_dir / 'search_summary.json').open('w', encoding='utf-8') as file:
        json.dump(locked_summary, file, indent=2, ensure_ascii=False)

    ablation_summaries = {}
    if args.run_ablations:
        ablation_overrides = {
            'fixed_fusion': {'fusion_mode': 'fixed'},
            'no_noisy_routing': {'disable_noisy_routing': True},
            'no_top_k': {'disable_top_k': True},
        }
        for name, overrides in ablation_overrides.items():
            ablation_configs = []
            for seed in args.seeds:
                config = copy.deepcopy(stage2_winner['config'])
                config.update(overrides, seed=seed)
                ablation_configs.append(config)
            summary = _run_stage(
                args,
                ablation_root / name,
                ablation_configs,
                runner,
            )
            summary.pop('winner', None)
            summary['ablation'] = name
            summary['locked_config_without_seed'] = {
                key: value
                for key, value in ablation_configs[0].items()
                if key != 'seed'
            }
            summary_path = ablation_root / name / 'search_summary.json'
            with summary_path.open('w', encoding='utf-8') as file:
                json.dump(summary, file, indent=2, ensure_ascii=False)
            ablation_summaries[name] = summary_path

    full_fusion_locked_path = None
    full_fusion_final_path = None
    comparison_summary = args.full_fusion_summary
    if args.run_full_fusion:
        full_fusion_configs = []
        for seed in args.seeds:
            config = {
                key: value
                for key, value in stage2_winner['config'].items()
                if key not in {
                    'fusion_mode',
                    'fusion_gate_bias',
                    'disable_top_k',
                    'disable_noisy_routing',
                }
            }
            config['seed'] = seed
            full_fusion_configs.append(config)
        full_fusion_locked_dir = full_fusion_root / 'locked_seeds'
        full_fusion_summary = _run_stage(
            args,
            full_fusion_locked_dir,
            full_fusion_configs,
            runner,
        )
        full_fusion_summary.pop('winner', None)
        full_fusion_summary['role'] = (
            'complete iMOE + iMOE_CSR prediction-level fusion comparison; '
            'excluded from DRC-iMOE selection and not claimed as a mathematical bound'
        )
        full_fusion_summary['locked_config_without_seed'] = {
            key: value
            for key, value in full_fusion_configs[0].items()
            if key != 'seed'
        }
        full_fusion_locked_path = full_fusion_locked_dir / 'search_summary.json'
        with full_fusion_locked_path.open('w', encoding='utf-8') as file:
            json.dump(full_fusion_summary, file, indent=2, ensure_ascii=False)
        full_fusion_final_dir = full_fusion_root / 'final'
        full_fusion_evaluation = [
            sys.executable,
            '-m',
            'experiments.evaluation.evaluate_locked_multiseed',
            '--search_summary', str(full_fusion_locked_path),
            '--output_dir', str(full_fusion_final_dir),
            '--device', args.device,
            '--gpu', str(args.gpus[0]),
            '--eval_batch_size', str(args.eval_batch_size),
            '--data_root', str(args.data_root),
        ]
        full_fusion_final_path = full_fusion_final_dir / 'summary.json'
        if not _completed_final_evaluation(
            full_fusion_final_path,
            full_fusion_locked_path,
        ):
            runner(full_fusion_evaluation, cwd=PROJECT_ROOT, check=True)
        comparison_summary = full_fusion_final_path

    evaluation = [
        sys.executable,
        '-m',
        'experiments.evaluation.evaluate_shared_dual_router_multiseed',
        '--search_summary', str(locked_dir / 'search_summary.json'),
        '--output_dir', str(final_dir),
        '--device', args.device,
        '--gpu', str(args.gpus[0]),
        '--eval_batch_size', str(args.eval_batch_size),
        '--data_root', str(args.data_root),
    ]
    if comparison_summary is not None:
        evaluation.extend([
            '--reference_summary', str(Path(comparison_summary).resolve()),
        ])
    final_summary_path = final_dir / 'summary.json'
    if not _completed_final_evaluation(
        final_summary_path,
        locked_dir / 'search_summary.json',
        comparison_summary,
    ):
        runner(evaluation, cwd=PROJECT_ROOT, check=True)
    outputs = {
        'stage1_summary': stage1_dir / 'search_summary.json',
        'stage2_summary': stage2_dir / 'search_summary.json',
        'locked_summary': locked_dir / 'search_summary.json',
        'final_summary': final_summary_path,
    }
    if ablation_summaries:
        outputs['ablation_summaries'] = ablation_summaries
    if full_fusion_locked_path is not None:
        outputs['full_fusion_locked_summary'] = full_fusion_locked_path
        outputs['full_fusion_final_summary'] = full_fusion_final_path
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the four-GPU DRC-iMOE search and locked evaluation'
    )
    parser.add_argument('--dataset', choices=DATASET_DEFAULTS, default='TPSL')
    parser.add_argument('--condition')
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument(
        '--output_root',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'tuning',
    )
    parser.add_argument(
        '--checkpoint_root',
        type=Path,
        default=PROJECT_ROOT / 'checkpoints' / 'tuning',
    )
    parser.add_argument('--search_seed', type=int, default=2025)
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(FINAL_SEEDS),
    )
    full_fusion_group = parser.add_mutually_exclusive_group()
    full_fusion_group.add_argument(
        '--full_fusion_summary',
        type=Path,
        help=(
            'Optional independent dual-model result used only as a final-test '
            'comparison; it never participates in SDR model selection'
        ),
    )
    full_fusion_group.add_argument(
        '--run_full_fusion',
        action='store_true',
        help=(
            'Train and test three-seed dual_router (complete iMOE + iMOE-CSR '
            'prediction-level fusion) after DRC-iMOE is locked'
        ),
    )
    parser.add_argument(
        '--run_ablations',
        action='store_true',
        help=(
            'After locking the SDR winner, train fixed-fusion, no-noise, and '
            'no-top-k variants with the same three seeds (selection-only)'
        ),
    )
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--pred_len', type=int, default=50)
    parser.add_argument('--dataaccess', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
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
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.condition is None:
        args.condition = defaults['condition']
    if args.hidden_dim is None:
        args.hidden_dim = defaults['hidden_dim']
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must be a non-empty list of unique values')
    if tuple(args.seeds) != FINAL_SEEDS:
        parser.error(f'--seeds is locked to {list(FINAL_SEEDS)}')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error('--gpus must be a non-empty list of unique values')
    locked_runtime = {
        '--seq_len': (args.seq_len, 50),
        '--pred_len': (args.pred_len, 50),
        '--dataaccess': (args.dataaccess, 100),
        '--batch_size': (args.batch_size, 32),
        '--eval_batch_size': (args.eval_batch_size, 64),
        '--train_epochs': (args.train_epochs, 1500),
        '--patience': (args.patience, 250),
    }
    for option, (actual, expected) in locked_runtime.items():
        if actual != expected:
            parser.error(f'{option} is locked to {expected} for the formal pipeline')
    return args


def main():
    outputs = run_pipeline(parse_args())
    serialized = {
        name: (
            {key: str(path) for key, path in value.items()}
            if isinstance(value, dict)
            else str(value)
        )
        for name, value in outputs.items()
    }
    print(json.dumps(
        serialized,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == '__main__':
    main()
