import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from experiments.evaluation.evaluate_shared_dual_router_multiseed import (
    SELECTION_ONLY_POLICY,
    aggregate_comparison_metrics,
    run_shared_evaluation,
    validate_shared_summary,
)


LOCKED_SEEDS = (2025, 2026, 2027)


def shared_summary():
    trials = []
    for trial_id, seed in enumerate(LOCKED_SEEDS):
        trials.append({
            'trial_id': trial_id,
            'config': {
                'seed': seed,
                'hidden_dim': 64,
                'learning_rate': 1e-4,
                'num_experts': 5,
                'diverloss': 0.5,
                'soc': 20,
                'baseline_top_k': 2,
                'csr_top_k': 4,
                'router_alpha': 10,
                'curve_channels': 4,
            },
            'validation_mse': 0.01 + trial_id,
            'checkpoint': f'shared_{seed}.pth',
        })
    return {
        'protocol': {'test_policy': SELECTION_ONLY_POLICY},
        'workflow': 'iMOE_SDR',
        'dataset': 'UL-NCA',
        'condition': 'CY25-025_1',
        'search': {'seeds': list(LOCKED_SEEDS)},
        'runtime': {
            'seq_len': 50,
            'pred_len': 2,
            'batch_size': 2,
            'dataaccess': 100,
        },
        'trials': trials,
    }


class EchoSharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.observed_training_modes = []

    def forward(self, prediction, features, charge, discharge, temperature):
        self.observed_training_modes.append(self.training)
        gates = torch.full((prediction.shape[0], 5), 0.2)
        return prediction * self.scale, gates


class TestSharedSummaryValidation(unittest.TestCase):
    def test_accepts_locked_seed_coverage_and_rejects_config_changes(self):
        trials, config = validate_shared_summary(shared_summary())

        self.assertEqual(
            [trial['config']['seed'] for trial in trials],
            list(LOCKED_SEEDS),
        )
        self.assertNotIn('seed', config)

        changed = shared_summary()
        changed['trials'][1]['config']['csr_top_k'] = 5
        with self.assertRaisesRegex(ValueError, 'identical except for seed'):
            validate_shared_summary(changed)


class TestSharedEvaluation(unittest.TestCase):
    def test_aggregate_reports_paired_shared_differences(self):
        seed_results = []
        for seed, offset in zip(LOCKED_SEEDS, range(len(LOCKED_SEEDS))):
            seed_results.append({
                'seed': seed,
                'metrics': {
                    'baseline': {
                        'rmse': 3.0 + offset,
                        'mae': 3.0 + offset,
                        'mape_percent': 3.0 + offset,
                        'r2': 0.1,
                    },
                    'dual_router': {
                        'rmse': 2.0 + offset,
                        'mae': 2.0 + offset,
                        'mape_percent': 2.0 + offset,
                        'r2': 0.2,
                    },
                    'iMOE_SDR': {
                        'rmse': 1.0 + offset,
                        'mae': 1.0 + offset,
                        'mape_percent': 1.0 + offset,
                        'r2': 0.3,
                    },
                },
            })

        aggregate = aggregate_comparison_metrics(seed_results)

        self.assertEqual(
            aggregate['models']['iMOE_SDR']['rmse'],
            {'mean': 2.0, 'std': float(np.std([1.0, 2.0, 3.0]))},
        )
        self.assertEqual(
            aggregate['iMOE_SDR_minus_baseline']['rmse']['by_seed'],
            {
                '2025': -2.0,
                '2026': -2.0,
                '2027': -2.0,
            },
        )
        self.assertEqual(
            aggregate['iMOE_SDR_minus_dual_router']['rmse']['mean'],
            -1.0,
        )

    def test_evaluates_each_shared_checkpoint_once_against_reference_arrays(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            summary = shared_summary()
            for trial in summary['trials']:
                (root / trial['checkpoint']).touch()
            summary_path = root / 'shared_search_summary.json'
            summary_path.write_text(json.dumps(summary), encoding='utf-8')

            prediction = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            reference_seeds = []
            for seed in LOCKED_SEEDS:
                baseline_path = root / f'baseline_{seed}.npy'
                dual_path = root / f'dual_{seed}.npy'
                true_path = root / f'true_{seed}.npy'
                np.save(baseline_path, prediction - 1.0)
                np.save(dual_path, prediction - 0.5)
                np.save(true_path, prediction)
                reference_seeds.append({
                    'seed': seed,
                    'artifacts': {
                        'baseline_pred': str(baseline_path),
                        'ensemble_pred': str(dual_path),
                        'true_values': str(true_path),
                    },
                })
            reference = {
                'dataset': summary['dataset'],
                'condition': summary['condition'],
                'declared_seeds': list(LOCKED_SEEDS),
                'seeds': reference_seeds,
            }
            reference_path = root / 'reference_summary.json'
            reference_path.write_text(json.dumps(reference), encoding='utf-8')

            loader_calls = []

            def fake_loader(args):
                loader_calls.append(args)
                values = torch.from_numpy(prediction)
                features = torch.zeros(2, 12)
                conditions = [torch.zeros_like(values) for _ in range(3)]
                loader = [((values, features, *conditions), values)]
                return None, None, loader, None

            loaded_models = []

            def fake_load_model(*_args, **_kwargs):
                model = EchoSharedModel()
                loaded_models.append(model)
                return model, object()

            with patch.dict(
                'experiments.evaluation.evaluate_shared_dual_router_multiseed.DATA_LOADERS',
                {'UL-NCA': fake_loader},
                clear=True,
            ), patch(
                'experiments.evaluation.evaluate_shared_dual_router_multiseed.load_trained_model',
                side_effect=fake_load_model,
            ) as load_model:
                result = run_shared_evaluation(
                    summary_path,
                    reference_path,
                    root / 'final',
                )

            self.assertEqual(len(loader_calls), 3)
            self.assertEqual(load_model.call_count, 3)
            self.assertTrue(all(
                model.observed_training_modes == [False]
                for model in loaded_models
            ))
            self.assertEqual(
                [seed_result['parameters']['total'] for seed_result in result['seeds']],
                [1, 1, 1],
            )
            self.assertTrue(all(
                seed_result['metrics']['iMOE_SDR']['rmse'] == 0.0
                for seed_result in result['seeds']
            ))
            self.assertTrue((root / 'final' / 'summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
