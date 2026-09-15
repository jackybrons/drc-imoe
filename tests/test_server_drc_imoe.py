import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from experiments.evaluation.evaluate_shared_dual_router_multiseed import validate_shared_summary
from experiments.main.run import build_parser
from experiments.main.run_drc_imoe_server import (
    _run_stage,
    parse_args,
    run_pipeline,
    stage1_configs,
)
from experiments.main.run_high_yield_drc_imoe_server import (
    parse_args as parse_high_yield_args,
    run_pipeline as run_high_yield_pipeline,
)
from experiments.tuning.tune_validation import load_baseline_config


def option_value(command, option, cast=str):
    return cast(command[command.index(option) + 1])


def config_from_command(command):
    config = {
        'seed': option_value(command, '--seeds', int),
        'hidden_dim': option_value(command, '--hidden_dims', int),
        'learning_rate': option_value(command, '--learning_rates', float),
        'num_experts': option_value(command, '--num_experts_values', int),
        'diverloss': option_value(command, '--diversity_weights', float),
        'soc': option_value(command, '--soc_values', int),
        'baseline_top_k': option_value(command, '--baseline_top_k_values', int),
        'csr_top_k': option_value(command, '--csr_top_k_values', int),
        'router_alpha': option_value(command, '--router_alpha_values', int),
        'curve_channels': option_value(command, '--curve_channels_values', int),
    }
    if '--fusion_mode_values' in command:
        config['fusion_mode'] = option_value(command, '--fusion_mode_values')
    if '--fusion_gate_bias_values' in command:
        config['fusion_gate_bias'] = option_value(
            command, '--fusion_gate_bias_values', float
        )
    if '--disable_top_k' in command:
        config['disable_top_k'] = True
    if '--disable_noisy_routing' in command:
        config['disable_noisy_routing'] = True
    return config


class FakeRunner:
    def __init__(self, test_case):
        self.test_case = test_case
        self.commands = []
        self.active_gpus = set()
        self.overlapped_gpu = False
        self.lock = threading.Lock()

    def __call__(self, command, cwd, check):
        self.test_case.assertTrue(check)
        self.test_case.assertTrue(Path(cwd).is_dir())
        if 'experiments.tuning.tune_validation' not in command:
            with self.lock:
                self.commands.append(command)
            return

        gpu = option_value(command, '--gpu', int)
        with self.lock:
            if gpu in self.active_gpus:
                self.overlapped_gpu = True
            self.active_gpus.add(gpu)
            self.commands.append(command)
        time.sleep(0.005)
        try:
            self._write_trial_summary(command)
        finally:
            with self.lock:
                self.active_gpus.remove(gpu)

    def _write_trial_summary(self, command):
        output_dir = Path(option_value(command, '--output_dir'))
        output_dir.mkdir(parents=True, exist_ok=True)
        config = config_from_command(command)
        output_text = str(output_dir)
        output_parts = output_dir.parts
        if 'lsd' in output_parts and 'search' in output_parts:
            validation_mse = (
                0.01
                if (
                    config['learning_rate'] == 3e-4
                    and config['diverloss'] == 0.1
                    and config['csr_top_k'] == 4
                )
                else 1.0
            )
        elif 'ul_nca' in output_parts and 'search' in output_parts:
            validation_mse = (
                0.01
                if config['learning_rate'] == 1e-4 and config['diverloss'] == 0.05
                else 1.0
            )
        elif '_stage1' in output_text:
            validation_mse = (
                0.01
                if (
                    config['learning_rate'] == 3e-4
                    and config['diverloss'] == 0.1
                    and config['baseline_top_k'] == 1
                )
                else 1.0
            )
        elif '_stage2' in output_text:
            validation_mse = (
                0.01
                if (
                    config['csr_top_k'] == 5
                    and config['curve_channels'] == 8
                    and config['fusion_mode'] == 'learned'
                    and config['fusion_gate_bias'] == 1.0
                )
                else 1.0
            )
        else:
            validation_mse = 0.1 + config['seed'] * 1e-9
        workflow = option_value(command, '--workflow')
        checkpoint = (output_dir / 'checkpoint.pth').resolve()
        checkpoint.touch()
        summary = {
            'protocol': {
                'selection_split': 'validation',
                'selection_metric': 'global_mse',
                'test_policy': 'not loaded or evaluated in selection-only mode',
            },
            'budget': {'max_trials': 1, 'max_training_jobs': 2},
            'search': {'search_seed': 2025, 'candidate_count': 1},
            'runtime': {
                'device': 'cuda',
                'gpu': option_value(command, '--gpu', int),
                'seq_len': 50,
                'pred_len': 50,
                'batch_size': 32,
                'dataaccess': 100,
            },
            'workflow': workflow,
            'dataset': option_value(command, '--dataset'),
            'condition': option_value(command, '--condition'),
            'trials': [{
                'trial_id': 0,
                'config': config,
                'validation_mse': validation_mse,
                'checkpoint': str(checkpoint),
            }],
        }
        if workflow == 'dual_router':
            baseline_checkpoint = (output_dir / 'baseline.pth').resolve()
            curve_checkpoint = (output_dir / 'curve.pth').resolve()
            baseline_checkpoint.touch()
            curve_checkpoint.touch()
            summary['trials'][0].update({
                'baseline_validation_mse': 0.2,
                'curve_validation_mse': 0.2,
                'alpha': 0.5,
                'baseline_source': (
                    'external' if '--baseline_checkpoint' in command else 'trained'
                ),
                'baseline_checkpoint': str(baseline_checkpoint),
                'curve_checkpoint': str(curve_checkpoint),
            })
        summary['winner'] = summary['trials'][0]
        (output_dir / 'search_summary.json').write_text(
            json.dumps(summary), encoding='utf-8'
        )


class TestServerPipeline(unittest.TestCase):
    def test_run_parser_accepts_fractional_diversity_weight(self):
        args = build_parser().parse_args(['--diverloss', '0.1'])
        self.assertEqual(args.diverloss, 0.1)

    def test_orphan_checkpoint_is_recovered_by_validation_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = parse_args([
                '--dataset', 'LSD',
                '--output_root', str(root / 'results'),
                '--checkpoint_root', str(root / 'checkpoints'),
            ])
            config = stage1_configs(args)[0]
            checkpoint = (
                root / 'checkpoints' / 'LSD_stage1' / 'parallel_trials'
                / 'trial_000' / 'trials' / 'trial_000' / 'iMOE_SDR'
                / 'LSD' / 'LSD' / 'iMOE_SDR_trial_000' / 'checkpoint.pth'
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            runner = FakeRunner(self)

            _run_stage(
                args,
                root / 'results' / 'LSD_stage1',
                [config],
                runner,
            )

            command = runner.commands[0]
            self.assertEqual(
                option_value(command, '--recover_checkpoint'),
                str(checkpoint),
            )

    def test_pipeline_uses_four_serial_gpu_queues_and_merges_all_trials(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = parse_args([
                '--dataset', 'TPSL',
                '--output_root', temporary_dir,
                '--gpus', '0', '1', '2', '3',
                '--run_ablations',
                '--run_full_fusion',
            ])
            runner = FakeRunner(self)
            outputs = run_pipeline(args, runner=runner)

            tuning_commands = [
                command for command in runner.commands
                if 'experiments.tuning.tune_validation' in command
            ]
            stage1 = sorted(
                (command for command in tuning_commands if '_stage1' in option_value(command, '--output_dir')),
                key=lambda command: option_value(command, '--output_dir'),
            )
            stage2 = sorted(
                (command for command in tuning_commands if '_stage2' in option_value(command, '--output_dir')),
                key=lambda command: option_value(command, '--output_dir'),
            )
            locked = sorted(
                (command for command in tuning_commands if '_locked_seeds' in option_value(command, '--output_dir')),
                key=lambda command: option_value(command, '--output_dir'),
            )

            self.assertEqual(len(stage1), 12)
            self.assertEqual(len(stage2), 18)
            self.assertEqual(len(locked), 3)
            self.assertEqual(
                [option_value(command, '--gpu', int) for command in stage1],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            )
            self.assertEqual(
                [option_value(command, '--gpu', int) for command in stage2],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
            )
            self.assertFalse(runner.overlapped_gpu)
            self.assertTrue(all('--selection_only' in command for command in tuning_commands))

            self.assertTrue(all(
                option_value(command, '--workflow') == 'iMOE_SDR'
                for command in stage1 + stage2 + locked
            ))
            stage1_configs = {(
                option_value(command, '--learning_rates', float),
                option_value(command, '--diversity_weights', float),
                option_value(command, '--baseline_top_k_values', int),
            ) for command in stage1}
            self.assertEqual(stage1_configs, {
                (learning_rate, diverloss, baseline_top_k)
                for learning_rate in (1e-4, 3e-4)
                for diverloss in (0.05, 0.1, 0.5)
                for baseline_top_k in (1, 2)
            })
            stage2_configs = {(
                option_value(command, '--csr_top_k_values', int),
                option_value(command, '--curve_channels_values', int),
                option_value(command, '--fusion_mode_values'),
                option_value(command, '--fusion_gate_bias_values', float),
            ) for command in stage2}
            self.assertEqual(stage2_configs, {
                (csr_top_k, curve_channels, 'learned', fusion_gate_bias)
                for csr_top_k in (3, 4, 5)
                for curve_channels in (4, 8)
                for fusion_gate_bias in (-1.0, 0.0, 1.0)
            })
            self.assertTrue(all('--baseline_checkpoint' not in command for command in stage2))

            stage1_summary = json.loads(outputs['stage1_summary'].read_text(encoding='utf-8'))
            stage2_summary = json.loads(outputs['stage2_summary'].read_text(encoding='utf-8'))
            locked_summary = json.loads(outputs['locked_summary'].read_text(encoding='utf-8'))
            self.assertEqual(len(stage1_summary['trials']), 12)
            self.assertEqual(len(stage2_summary['trials']), 18)
            self.assertEqual(stage1_summary['budget']['max_training_jobs'], 12)
            self.assertEqual(stage2_summary['budget']['max_training_jobs'], 18)
            self.assertEqual(locked_summary['budget']['max_training_jobs'], 3)
            self.assertEqual(stage1_summary['winner']['config']['learning_rate'], 3e-4)
            self.assertEqual(stage1_summary['winner']['config']['diverloss'], 0.1)
            self.assertEqual(stage2_summary['winner']['config']['baseline_top_k'], 1)
            self.assertEqual(stage2_summary['winner']['config']['csr_top_k'], 5)
            self.assertEqual(stage2_summary['winner']['config']['curve_channels'], 8)
            self.assertEqual(stage2_summary['winner']['config']['fusion_mode'], 'learned')
            self.assertEqual(stage2_summary['winner']['config']['fusion_gate_bias'], 1.0)
            self.assertNotIn('winner', locked_summary)
            trials, locked_config = validate_shared_summary(locked_summary)
            self.assertEqual(
                [trial['config']['seed'] for trial in trials],
                [2025, 2026, 2027],
            )
            self.assertEqual(locked_config['learning_rate'], 3e-4)
            self.assertEqual(locked_config['diverloss'], 0.1)
            self.assertEqual(locked_config['csr_top_k'], 5)
            self.assertEqual(locked_config['curve_channels'], 8)

            evaluation = [
                command for command in runner.commands
                if 'experiments.evaluation.evaluate_shared_dual_router_multiseed' in command
            ]
            self.assertEqual(len(evaluation), 1)
            self.assertEqual(option_value(evaluation[0], '--gpu', int), 0)
            self.assertEqual(
                option_value(evaluation[0], '--search_summary'),
                str(outputs['locked_summary']),
            )
            self.assertEqual(
                option_value(evaluation[0], '--reference_summary'),
                str(outputs['full_fusion_final_summary'].resolve()),
            )
            full_fusion_tuning = [
                command for command in tuning_commands
                if 'full_fusion' in option_value(command, '--output_dir')
            ]
            self.assertEqual(len(full_fusion_tuning), 3)
            self.assertTrue(all(
                option_value(command, '--workflow') == 'dual_router'
                and '--fusion_mode_values' not in command
                and '--fusion_gate_bias_values' not in command
                for command in full_fusion_tuning
            ))
            full_fusion_evaluation = [
                command for command in runner.commands
                if 'experiments.evaluation.evaluate_locked_multiseed' in command
            ]
            self.assertEqual(len(full_fusion_evaluation), 1)
            self.assertEqual(set(outputs['ablation_summaries']), {
                'fixed_fusion', 'no_noisy_routing', 'no_top_k'
            })
            for name, summary_path in outputs['ablation_summaries'].items():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                self.assertEqual(summary['ablation'], name)
                self.assertEqual(len(summary['trials']), 3)
                self.assertNotIn('winner', summary)

            resume_runner = FakeRunner(self)
            run_pipeline(args, runner=resume_runner)
            self.assertFalse(any(
                'experiments.tuning.tune_validation' in command
                for command in resume_runner.commands
            ))

    def test_high_yield_pipeline_runs_exact_lsd_then_ul_protocol(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = parse_high_yield_args([
                '--output_root', temporary_dir,
                '--gpus', '0', '1', '2', '3',
            ])
            runner = FakeRunner(self)
            outputs = run_high_yield_pipeline(args, runner=runner)

            tuning_commands = [
                command for command in runner.commands
                if 'experiments.tuning.tune_validation' in command
            ]

            def commands_in(dataset_dir, stage_dir):
                marker = str(Path(dataset_dir) / stage_dir / 'parallel_trials')
                return sorted(
                    (
                        command for command in tuning_commands
                        if marker in option_value(command, '--output_dir')
                    ),
                    key=lambda command: option_value(command, '--output_dir'),
                )

            lsd_search = commands_in('lsd', 'search')
            lsd_locked = commands_in('lsd', 'locked_seeds')
            ul_search = commands_in('ul_nca', 'search')
            ul_locked = commands_in('ul_nca', 'locked_seeds')

            self.assertEqual(
                [len(lsd_search), len(lsd_locked), len(ul_search), len(ul_locked)],
                [8, 3, 4, 3],
            )
            self.assertTrue(all(
                option_value(command, '--batch_size', int) == 32
                for command in lsd_search + lsd_locked
            ))
            self.assertTrue(all(
                option_value(command, '--batch_size', int) == 32
                for command in ul_search + ul_locked
            ))
            self.assertEqual(
                {(
                    option_value(command, '--learning_rates', float),
                    option_value(command, '--diversity_weights', float),
                    option_value(command, '--csr_top_k_values', int),
                ) for command in lsd_search},
                {
                    (learning_rate, diverloss, csr_top_k)
                    for learning_rate in (1e-4, 3e-4)
                    for diverloss in (0.05, 0.1)
                    for csr_top_k in (3, 4)
                },
            )
            self.assertEqual(
                {(
                    option_value(command, '--learning_rates', float),
                    option_value(command, '--diversity_weights', float),
                    option_value(command, '--csr_top_k_values', int),
                ) for command in ul_search},
                {
                    (learning_rate, diverloss, 3)
                    for learning_rate in (1e-4, 3e-4)
                    for diverloss in (0.05, 0.1)
                },
            )

            for command in tuning_commands:
                self.assertEqual(option_value(command, '--hidden_dims', int), 64)
                self.assertEqual(option_value(command, '--num_experts_values', int), 5)
                self.assertEqual(option_value(command, '--baseline_top_k_values', int), 2)
                self.assertEqual(option_value(command, '--router_alpha_values', int), 10)
                self.assertEqual(option_value(command, '--curve_channels_values', int), 4)
                self.assertIn('--selection_only', command)

            self.assertEqual(
                [option_value(command, '--gpu', int) for command in lsd_search],
                [0, 1, 2, 3, 0, 1, 2, 3],
            )
            self.assertEqual(
                [option_value(command, '--gpu', int) for command in ul_search],
                [0, 1, 2, 3],
            )
            self.assertEqual(
                [option_value(command, '--gpu', int) for command in lsd_locked],
                [0, 1, 2],
            )
            self.assertEqual(
                [option_value(command, '--gpu', int) for command in ul_locked],
                [0, 1, 2],
            )
            self.assertFalse(runner.overlapped_gpu)

            lsd_locked_configs = [config_from_command(command) for command in lsd_locked]
            self.assertEqual(
                [config['seed'] for config in lsd_locked_configs],
                [2025, 2026, 2027],
            )
            self.assertTrue(all(
                config['learning_rate'] == 3e-4
                and config['diverloss'] == 0.1
                and config['csr_top_k'] == 4
                for config in lsd_locked_configs
            ))
            ul_locked_configs = [config_from_command(command) for command in ul_locked]
            self.assertEqual(
                [config['seed'] for config in ul_locked_configs],
                [2025, 2026, 2027],
            )
            self.assertTrue(all(
                config['learning_rate'] == 1e-4
                and config['diverloss'] == 0.05
                and config['csr_top_k'] == 3
                for config in ul_locked_configs
            ))

            evaluations = [
                command for command in runner.commands
                if 'experiments.evaluation.evaluate_locked_multiseed' in command
            ]
            self.assertEqual(len(evaluations), 2)
            command_indexes = {
                id(command): index for index, command in enumerate(runner.commands)
            }
            self.assertLess(
                max(command_indexes[id(command)] for command in lsd_search),
                min(command_indexes[id(command)] for command in lsd_locked),
            )
            self.assertLess(
                max(command_indexes[id(command)] for command in lsd_locked),
                command_indexes[id(evaluations[0])],
            )
            self.assertLess(
                command_indexes[id(evaluations[0])],
                min(command_indexes[id(command)] for command in ul_search),
            )
            self.assertLess(
                max(command_indexes[id(command)] for command in ul_search),
                min(command_indexes[id(command)] for command in ul_locked),
            )
            self.assertLess(
                max(command_indexes[id(command)] for command in ul_locked),
                command_indexes[id(evaluations[1])],
            )
            self.assertEqual(
                option_value(evaluations[0], '--search_summary'),
                str(outputs['LSD']['locked_summary']),
            )
            self.assertEqual(
                option_value(evaluations[1], '--search_summary'),
                str(outputs['UL-NCA']['locked_summary']),
            )

            lsd_summary = json.loads(
                outputs['LSD']['search_summary'].read_text(encoding='utf-8')
            )
            ul_summary = json.loads(
                outputs['UL-NCA']['search_summary'].read_text(encoding='utf-8')
            )
            self.assertEqual(lsd_summary['search']['candidate_count'], 8)
            self.assertEqual(ul_summary['search']['candidate_count'], 4)
            self.assertEqual(
                outputs['LSD']['search_summary'].parent.name,
                'search',
            )
            self.assertNotIn('_stage1', str(outputs['LSD']['search_summary']))
            self.assertNotIn('_stage2', str(outputs['LSD']['search_summary']))

    def test_load_baseline_config_accepts_merged_winner(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / 'search_summary.json'
            config = {
                'seed': 2025,
                'hidden_dim': 128,
                'learning_rate': 3e-4,
                'num_experts': 5,
                'diverloss': 0.1,
                'soc': 20,
                'baseline_top_k': 2,
                'router_alpha': 10,
                'csr_top_k': 5,
                'curve_channels': 8,
            }
            path.write_text(json.dumps({'winner': {'config': config}}), encoding='utf-8')
            loaded = load_baseline_config(path)
            self.assertEqual(loaded['learning_rate'], 3e-4)
            self.assertEqual(loaded['diverloss'], 0.1)
            self.assertNotIn('csr_top_k', loaded)


if __name__ == '__main__':
    unittest.main()
