import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from experiments.core.exp_forecasting import Exp_Long_Term_Forecast1
from models.iMOE_CSR import Model as CSRModel
from experiments.tuning.tune_validation import (
    baseline_cache_key,
    build_search_space,
    choose_budgeted_trials,
    make_experiment_args,
    model_hparams,
    select_best_trial,
    validate_baseline_reuse,
)


class TestValidationTuning(unittest.TestCase):
    def _space(self, workflow='dual_router'):
        return build_search_space(
            workflow=workflow,
            seeds=[2025, 2026],
            hidden_dims=[32, 64],
            learning_rates=[1e-4],
            num_experts_values=[5],
            baseline_top_k_values=[1, 2],
            csr_top_k_values=[3, 4, 5],
            router_alpha_values=[5, 10],
            curve_channels_values=[2, 4],
            diversity_weights=[0.25],
            soc_values=[20],
        )

    def test_budget_is_exact_and_reproducible(self):
        space = self._space()
        first = choose_budgeted_trials(space, max_trials=5, search_seed=17)
        second = choose_budgeted_trials(space, max_trials=5, search_seed=17)

        self.assertEqual(len(first), 5)
        self.assertEqual(first, second)

    def test_search_space_contains_required_router_and_curve_knobs(self):
        config = self._space()[0]

        self.assertIn('seed', config)
        self.assertIn('hidden_dim', config)
        self.assertIn('learning_rate', config)
        self.assertIn('baseline_top_k', config)
        self.assertIn('router_alpha', config)
        self.assertIn('csr_top_k', config)
        self.assertIn('curve_channels', config)
        self.assertNotIn('fusion_mode', config)
        self.assertNotIn('fusion_gate_bias', config)

    def test_sdr_search_includes_fusion_mode_without_test_evaluation(self):
        configs = build_search_space(
            workflow='iMOE_SDR',
            seeds=[2025],
            hidden_dims=[128],
            learning_rates=[1e-4],
            num_experts_values=[5],
            baseline_top_k_values=[1, 2],
            csr_top_k_values=[4],
            router_alpha_values=[10],
            curve_channels_values=[4],
            diversity_weights=[0.1],
            soc_values=[20],
            fusion_mode_values=['learned', 'fixed'],
            fusion_gate_bias_values=[-1.0, 0.0, 1.0],
        )

        self.assertEqual(len(configs), 12)
        self.assertEqual(
            {config['fusion_mode'] for config in configs},
            {'learned', 'fixed'},
        )
        self.assertEqual(
            {config['fusion_gate_bias'] for config in configs},
            {-1.0, 0.0, 1.0},
        )

    def test_csr_top_k_values_include_sparse_and_dense_options(self):
        configs = self._space(workflow='iMOE_CSR')

        self.assertEqual({config['csr_top_k'] for config in configs}, {3, 4, 5})
        for config in configs:
            values = model_hparams(config, 'iMOE_CSR')
            self.assertEqual(values['top_k'], config['csr_top_k'])
            self.assertEqual(values['curve_channels'], config['curve_channels'])

    def test_curve_channels_reaches_csr_encoder(self):
        model = CSRModel(SimpleNamespace(
            dataset='UL-NCA',
            seq_len=50,
            pred_len=50,
            num_experts=5,
            top_k=5,
            hidden_dim=8,
            curve_channels=2,
        ))

        self.assertEqual(model.curve_encoder.output_dim, 6)

    def test_winner_selection_ignores_test_metrics(self):
        trials = [
            {'trial_id': 0, 'validation_mse': 0.20, 'test_mse': 0.01},
            {'trial_id': 1, 'validation_mse': 0.10, 'test_mse': 100.0},
        ]

        winner = select_best_trial(trials)

        self.assertEqual(winner['trial_id'], 1)

    def test_same_baseline_config_has_same_cache_key(self):
        first = self._space()[0]
        second = dict(first, csr_top_k=5, curve_channels=8)

        self.assertEqual(baseline_cache_key(first), baseline_cache_key(second))

    def test_external_baseline_config_must_match(self):
        config = self._space()[0]
        baseline_config = {
            key: value
            for key, value in baseline_cache_key(config)
        }
        validate_baseline_reuse(config, baseline_config)
        baseline_config['hidden_dim'] += 1

        with self.assertRaises(ValueError):
            validate_baseline_reuse(config, baseline_config)

    def test_tuning_args_select_prediction_mse_and_skip_test(self):
        cli_args = SimpleNamespace(
            dataset='UL-NCA',
            condition='CY25-025_1',
            seq_len=50,
            pred_len=50,
            dataaccess=100,
            batch_size=32,
            train_epochs=2,
            patience=2,
            device='cpu',
            gpu=0,
        )
        args = make_experiment_args(
            cli_args,
            self._space()[0],
            'iMOE',
            'unused',
        )

        self.assertEqual(args.selection_metric, 'prediction_mse')
        self.assertTrue(args.skip_test)
        self.assertEqual(args.fusion_mode, 'learned')

    def test_budget_cannot_exceed_unique_space(self):
        with self.assertRaises(ValueError):
            choose_budgeted_trials(self._space(), max_trials=1000, search_seed=17)


class EchoPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.observed_training_modes = []

    def forward(self, prediction, features, charge, discharge, temperature):
        self.observed_training_modes.append(self.training)
        gates = torch.full((prediction.shape[0], 5), 0.2)
        return prediction, gates


class TestPredictionMSECheckpointMetric(unittest.TestCase):
    def test_validation_metric_is_global_elementwise_prediction_mse(self):
        experiment = object.__new__(Exp_Long_Term_Forecast1)
        experiment.model = EchoPredictionModel()
        experiment.model.train()
        experiment.device = torch.device('cpu')
        experiment.args = SimpleNamespace(
            selection_metric='prediction_mse',
            model='iMOE',
            diverloss=1000.0,
        )

        first_prediction = torch.ones(2, 2)
        second_prediction = torch.full((1, 2), 3.0)
        zeros_first = torch.zeros_like(first_prediction)
        zeros_second = torch.zeros_like(second_prediction)
        validation_loader = [
            ((first_prediction, zeros_first, zeros_first, zeros_first, zeros_first), zeros_first),
            ((second_prediction, zeros_second, zeros_second, zeros_second, zeros_second), zeros_second),
        ]

        value = experiment.vali(None, validation_loader, nn.MSELoss())

        self.assertAlmostEqual(value, 22.0 / 6.0)
        self.assertEqual(experiment.model.observed_training_modes, [False, False])
        self.assertTrue(experiment.model.training)


if __name__ == '__main__':
    unittest.main()
