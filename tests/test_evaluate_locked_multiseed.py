import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.evaluation.evaluate_locked_multiseed import (
    SELECTION_ONLY_POLICY,
    aggregate_seed_metrics,
    run_locked_evaluation,
    validate_locked_summary,
)


def locked_summary(seeds=(2025, 2026)):
    trials = []
    for trial_id, seed in enumerate(seeds):
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
            'alpha': 0.25 + 0.5 * trial_id,
            'baseline_checkpoint': f'baseline_{seed}.pth',
            'curve_checkpoint': f'curve_{seed}.pth',
        })
    return {
        'protocol': {'test_policy': SELECTION_ONLY_POLICY},
        'workflow': 'dual_router',
        'dataset': 'UL-NCA',
        'condition': 'CY25-025_1',
        'search': {'seeds': list(seeds)},
        'runtime': {
            'seq_len': 50,
            'pred_len': 2,
            'batch_size': 2,
            'dataaccess': 100,
        },
        'trials': trials,
        'winner': trials[0],
    }


class TestLockedSummaryValidation(unittest.TestCase):
    def test_accepts_exact_seed_coverage_and_ignores_winner(self):
        trials, config = validate_locked_summary(locked_summary())

        self.assertEqual([trial['config']['seed'] for trial in trials], [2025, 2026])
        self.assertNotIn('seed', config)

    def test_rejects_configuration_changes_other_than_seed(self):
        summary = locked_summary()
        summary['trials'][1]['config']['csr_top_k'] = 5

        with self.assertRaisesRegex(ValueError, 'identical except for seed'):
            validate_locked_summary(summary)

    def test_rejects_missing_declared_seed_or_prior_final_test(self):
        summary = locked_summary()
        summary['trials'].pop()
        with self.assertRaisesRegex(ValueError, 'Trial count'):
            validate_locked_summary(summary)

        summary = locked_summary()
        summary['final_test'] = {'global': {'rmse': 0.0}}
        with self.assertRaisesRegex(ValueError, 'selection-only'):
            validate_locked_summary(summary)


class TestLockedMultiSeedEvaluation(unittest.TestCase):
    def test_aggregate_reports_paired_ensemble_minus_baseline(self):
        seed_results = [
            {
                'seed': 1,
                'metrics': {
                    'baseline': {'rmse': 2, 'mae': 2, 'mape_percent': 2, 'r2': 0.2},
                    'curve': {'rmse': 1, 'mae': 1, 'mape_percent': 1, 'r2': 0.5},
                    'ensemble': {'rmse': 1.5, 'mae': 1.5, 'mape_percent': 1.5, 'r2': 0.4},
                },
            },
            {
                'seed': 2,
                'metrics': {
                    'baseline': {'rmse': 4, 'mae': 4, 'mape_percent': 4, 'r2': 0.0},
                    'curve': {'rmse': 3, 'mae': 3, 'mape_percent': 3, 'r2': 0.3},
                    'ensemble': {'rmse': 3, 'mae': 3, 'mape_percent': 3, 'r2': 0.2},
                },
            },
        ]

        aggregate = aggregate_seed_metrics(seed_results)

        self.assertEqual(aggregate['models']['baseline']['rmse'], {'mean': 3.0, 'std': 1.0})
        self.assertEqual(aggregate['ensemble_minus_baseline']['rmse']['mean'], -0.75)
        self.assertEqual(
            aggregate['ensemble_minus_baseline']['rmse']['by_seed'],
            {'1': -0.5, '2': -1.0},
        )

    def test_evaluates_every_seed_once_with_stored_validation_alpha(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            summary = locked_summary()
            for trial in summary['trials']:
                (root / trial['baseline_checkpoint']).touch()
                (root / trial['curve_checkpoint']).touch()
            summary_path = root / 'search_summary.json'
            summary_path.write_text(json.dumps(summary), encoding='utf-8')

            predictions = [
                (
                    np.array([[0.0, 1.0], [2.0, 3.0]]),
                    np.array([[1.0, 2.0], [3.0, 4.0]]),
                    np.array([[0.25, 1.25], [2.25, 3.25]]),
                ),
                (
                    np.array([[0.0, 1.0], [2.0, 3.0]]),
                    np.array([[2.0, 3.0], [4.0, 5.0]]),
                    np.array([[1.5, 2.5], [3.5, 4.5]]),
                ),
            ]
            loader_calls = []

            def fake_loader(args):
                loader_calls.append(args)
                return None, None, object(), None

            with patch.dict('experiments.evaluation.evaluate_locked_multiseed.DATA_LOADERS', {'UL-NCA': fake_loader}, clear=True), \
                    patch('experiments.evaluation.evaluate_locked_multiseed.load_trained_model', return_value=(object(), object())) as load_model, \
                    patch('experiments.evaluation.evaluate_locked_multiseed.predict_pair', side_effect=predictions) as predict:
                result = run_locked_evaluation(summary_path, root / 'final')

            self.assertEqual(len(loader_calls), 2)
            self.assertEqual(load_model.call_count, 4)
            self.assertEqual(predict.call_count, 2)
            np.testing.assert_allclose(
                np.load(root / 'final' / 'seed_2025' / 'ensemble_pred.npy'),
                predictions[0][2],
            )
            np.testing.assert_allclose(
                np.load(root / 'final' / 'seed_2026' / 'ensemble_pred.npy'),
                predictions[1][2],
            )
            self.assertEqual([item['validation_alpha'] for item in result['seeds']], [0.25, 0.75])
            self.assertNotIn('winner', result)
            self.assertTrue((root / 'final' / 'summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
