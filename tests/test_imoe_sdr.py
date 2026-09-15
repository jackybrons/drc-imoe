from types import SimpleNamespace
import json
import os
import tempfile
import unittest

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.loader import _build_sample_metadata
from experiments.core.exp_forecasting import Exp_Long_Term_Forecast1, _global_metrics
from models import iMOE, iMOE_CSR, iMOE_SDR


class TestIMOESDR(unittest.TestCase):
    def _args(self, dataset):
        return SimpleNamespace(
            dataset=dataset,
            seq_len=50,
            pred_len=50,
            num_experts=5,
            top_k=4,
            baseline_top_k=2,
            csr_top_k=4,
            alpha=10,
            hidden_dim=8,
            curve_channels=4,
        )

    def _inputs(self, dataset, batch_size=4):
        feature_dim = 6 if dataset == 'TPSL' else 12
        return (
            torch.randn(batch_size, 50),
            torch.randn(batch_size, feature_dim),
            torch.randn(batch_size, 50),
            torch.randn(batch_size, 50),
            torch.randn(batch_size, 50),
        )

    def _assert_nonzero_finite_gradient(self, module):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) > 0 for gradient in gradients))

    def test_forward_backward_supported_datasets(self):
        torch.manual_seed(17)
        for dataset in ('UL-NCA', 'TPSL', 'LSD'):
            with self.subTest(dataset=dataset):
                model = iMOE_SDR.Model(self._args(dataset))
                inputs = self._inputs(dataset)
                outputs, weights = model(*inputs)

                self.assertEqual(outputs.shape, (4, 50))
                self.assertEqual(weights.shape, (4, 5))
                self.assertTrue(torch.isfinite(outputs).all())
                self.assertTrue(torch.isfinite(weights).all())
                self.assertTrue(
                    torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6)
                )

                target = torch.randn_like(outputs)
                (outputs - target).square().mean().backward()
                for module in (
                    model.baseline_router,
                    model.curve_encoder,
                    model.csr_router,
                    model.fusion_gate,
                    model.capacity_fc,
                    model.lstm,
                    model.fc_final,
                ):
                    self._assert_nonzero_finite_gradient(module)

    def test_shared_experts_are_evaluated_once_and_reduce_parameters(self):
        args = self._args('UL-NCA')
        model = iMOE_SDR.Model(args).eval()
        calls = [0] * args.num_experts
        hooks = []
        for index, expert in enumerate(model.capacity_fc):
            def record_call(_module, _inputs, _output, expert_index=index):
                calls[expert_index] += 1

            hooks.append(expert.register_forward_hook(record_call))

        try:
            with torch.no_grad():
                model(*self._inputs('UL-NCA'))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(len(model.capacity_fc), args.num_experts)
        self.assertEqual(calls, [1] * args.num_experts)

        baseline_args = self._args('UL-NCA')
        baseline_args.top_k = 2
        baseline = iMOE.Model(baseline_args)
        curve = iMOE_CSR.Model(args)
        shared_parameters = sum(parameter.numel() for parameter in model.parameters())
        dual_model_parameters = sum(
            parameter.numel()
            for component in (baseline, curve)
            for parameter in component.parameters()
        )
        self.assertLess(shared_parameters, dual_model_parameters)

    def test_curve_evidence_changes_fused_routing_with_fixed_statistics(self):
        model = iMOE_SDR.Model(self._args('UL-NCA')).eval()
        with torch.no_grad():
            model.baseline_router.weight.zero_()
            model.baseline_router.bias.zero_()
            for branch in model.curve_encoder.branches:
                branch[0].weight.fill_(1.0)
                branch[0].bias.zero_()
            model.csr_router.weight.zero_()
            model.csr_router.bias.copy_(
                torch.tensor([0.0, 0.0, -10.0, -10.0, -10.0])
            )
            model.csr_router.weight[0, 12] = 1.0
            for parameter in model.fusion_gate.parameters():
                parameter.zero_()

        features = torch.zeros(2, 12)
        conditions = [torch.zeros(2, 50) for _ in range(3)]
        _, zero_curve_weights = model(
            torch.zeros(2, 50), features, *conditions
        )
        _, one_curve_weights = model(
            torch.ones(2, 50), features, *conditions
        )

        self.assertFalse(torch.allclose(zero_curve_weights, one_curve_weights))
        for weights in (zero_curve_weights, one_curve_weights):
            self.assertTrue(torch.all(weights >= 0.0))
            self.assertTrue(torch.all(weights <= 1.0))
            self.assertTrue(
                torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
            )

    def test_model_builds_through_experiment_registry(self):
        args = self._args('UL-NCA')
        args.model = 'iMOE_SDR'
        args.use_gpu = False
        args.use_multi_gpu = False
        args.gpu = 0
        args.devices = '0'

        experiment = Exp_Long_Term_Forecast1(args)

        self.assertIsInstance(experiment.model, iMOE_SDR.Model)

    def test_forward_exposes_per_sample_routing_diagnostics(self):
        model = iMOE_SDR.Model(self._args('UL-NCA')).eval()

        with torch.no_grad():
            result = model(*self._inputs('UL-NCA'))
        diagnostics = model.get_routing_diagnostics()

        self.assertEqual(len(result), 2)
        self.assertEqual(set(diagnostics), {
            'fusion_gate',
            'router_weight_l1',
            'statistical_route_entropy',
            'curve_route_entropy',
            'fused_route_entropy',
        })
        for values in diagnostics.values():
            self.assertEqual(values.shape, (4,))
            self.assertTrue(torch.isfinite(values).all())
            self.assertFalse(values.requires_grad)

    def test_fixed_fusion_uses_exact_half_without_a_gate_module(self):
        args = self._args('UL-NCA')
        args.fusion_mode = 'fixed'
        model = iMOE_SDR.Model(args).eval()

        with torch.no_grad():
            model(*self._inputs('UL-NCA'))

        self.assertIsNone(model.fusion_gate)
        self.assertTrue(torch.equal(
            model.get_routing_diagnostics()['fusion_gate'],
            torch.full((4,), 0.5),
        ))

    def test_routing_ablation_switches_disable_noise_and_top_k(self):
        args = self._args('UL-NCA')
        args.disable_noisy_routing = True
        args.disable_top_k = True
        model = iMOE_SDR.Model(args).train()
        inputs = self._inputs('UL-NCA')

        first_outputs, first_weights = model(*inputs)
        second_outputs, second_weights = model(*inputs)

        self.assertIsNone(model.baseline_w_noise)
        self.assertIsNone(model.csr_w_noise)
        self.assertTrue(torch.equal(first_outputs, second_outputs))
        self.assertTrue(torch.equal(first_weights, second_weights))
        logits = torch.tensor([
            [-2.0, -1.0, 0.0, 1.0, 2.0],
        ])
        self.assertTrue(torch.all(model._baseline_weights(logits) > 0.0))
        self.assertTrue(torch.all(model._csr_weights(logits) > 0.0))

    def test_learned_fusion_gate_respects_trainable_initial_bias(self):
        inputs = self._inputs('UL-NCA')
        for bias in (-1.0, 0.0, 1.0):
            with self.subTest(bias=bias):
                args = self._args('UL-NCA')
                args.fusion_gate_bias = bias
                model = iMOE_SDR.Model(args).eval()

                with torch.no_grad():
                    model(*inputs)

                expected = torch.sigmoid(torch.tensor(bias))
                gate = model.get_routing_diagnostics()['fusion_gate']
                self.assertTrue(torch.allclose(
                    gate,
                    torch.full_like(gate, expected),
                ))
                self.assertTrue(model.fusion_gate[0].bias.requires_grad)

    def test_metadata_uses_source_cycle_and_per_battery_thirds(self):
        data = pd.DataFrame({
            'Cycle': [100, 110, 120, 130, 140, 150, 160],
        })

        metadata = _build_sample_metadata(
            data,
            'battery-1',
            range(1, 7),
        )

        self.assertEqual(
            [item['cycle_index'] for item in metadata],
            [110, 120, 130, 140, 150, 160],
        )
        self.assertEqual(
            [item['window_start'] for item in metadata],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            [item['degradation_stage'] for item in metadata],
            ['early', 'early', 'middle', 'middle', 'late', 'late'],
        )
        fallback = _build_sample_metadata(
            pd.DataFrame({'value': range(3)}),
            'battery-2',
            range(3),
        )
        self.assertEqual(
            [item['cycle_index'] for item in fallback],
            [0, 1, 2],
        )


class ConcentratedGateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.observed_training_modes = []

    def forward(self, prediction, features, charge, discharge, temperature):
        self.observed_training_modes.append(self.training)
        gates = torch.zeros((prediction.shape[0], 5))
        gates[:, 0] = 1.0
        return prediction, gates


class TestIMOESDRTrainingContract(unittest.TestCase):
    def test_validation_applies_moe_diversity_loss(self):
        experiment = object.__new__(Exp_Long_Term_Forecast1)
        experiment.model = ConcentratedGateModel()
        experiment.model.train()
        experiment.device = torch.device('cpu')
        experiment.args = SimpleNamespace(
            selection_metric=None,
            model='iMOE_SDR',
            diverloss=0.5,
        )
        prediction = torch.zeros(2, 3)
        validation_loader = [(
            (prediction, prediction, prediction, prediction, prediction),
            prediction,
        )]

        value = experiment.vali(None, validation_loader, nn.MSELoss())

        self.assertGreater(value, 0.0)
        self.assertEqual(experiment.model.observed_training_modes, [False])
        self.assertTrue(experiment.model.training)

    def test_validation_aggregates_and_saves_sdr_diagnostics(self):
        args = SimpleNamespace(
            dataset='UL-NCA',
            seq_len=3,
            pred_len=3,
            num_experts=3,
            baseline_top_k=2,
            csr_top_k=2,
            alpha=10,
            hidden_dim=2,
            curve_channels=2,
            selection_metric='prediction_mse',
            model='iMOE_SDR',
            diverloss=0.5,
        )
        experiment = object.__new__(Exp_Long_Term_Forecast1)
        experiment.model = iMOE_SDR.Model(args)
        experiment.device = torch.device('cpu')
        experiment.args = args
        experiment.validation_diagnostics_history = []
        inputs = (
            torch.randn(2, 3),
            torch.randn(2, 12),
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
        )
        validation_loader = [(inputs, torch.zeros(2, 3))]

        validation_loss = experiment.vali(
            None,
            validation_loader,
            nn.MSELoss(),
        )

        expected_metrics = {
            f'{name}_{statistic}'
            for name in (
                'fusion_gate',
                'router_weight_l1',
                'statistical_route_entropy',
                'curve_route_entropy',
                'fused_route_entropy',
            )
            for statistic in ('mean', 'std')
        }
        self.assertEqual(
            set(experiment.last_validation_diagnostics),
            expected_metrics,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            experiment._save_validation_diagnostics(
                output_dir,
                epoch=1,
                validation_loss=validation_loss,
            )
            diagnostics_path = os.path.join(
                output_dir,
                'validation_routing_diagnostics.json',
            )
            with open(diagnostics_path, encoding='utf-8') as diagnostics_file:
                saved = json.load(diagnostics_file)

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['epoch'], 1)
        self.assertEqual(set(saved[0]), {
            'epoch',
            'validation_loss',
            *expected_metrics,
        })


class ExplainabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diagnostics = None

    def forward(self, prediction, features, charge, discharge, temperature):
        del features, charge, discharge, temperature
        marker = prediction[:, 0]
        gates = torch.stack([
            marker,
            1.0 - marker,
            torch.zeros_like(marker),
        ], dim=1)
        self.diagnostics = {
            'fusion_gate': marker,
            'router_weight_l1': marker + 1.0,
            'statistical_route_entropy': marker + 2.0,
            'curve_route_entropy': marker + 3.0,
            'fused_route_entropy': marker + 4.0,
        }
        return prediction, gates

    def get_routing_diagnostics(self):
        return self.diagnostics


class TestIMOESDRExplainabilityExport(unittest.TestCase):
    def test_global_metrics_match_flattened_prediction_definition(self):
        prediction = [[1.0, 2.0], [3.0, 4.0]]
        true_values = [[1.0, 1.0], [2.0, 4.0]]

        metrics = _global_metrics(prediction, true_values)

        self.assertAlmostEqual(metrics['rmse'], 2 ** -0.5)
        self.assertAlmostEqual(metrics['mae'], 0.5)
        self.assertAlmostEqual(metrics['mape_percent'], 37.5)
        self.assertAlmostEqual(metrics['r2'], 2.0 / 3.0)

    def test_test_export_preserves_loader_metadata_order(self):
        samples = []
        for marker in (0.2, 0.7):
            prediction = torch.full((3,), marker)
            zeros = torch.zeros(3)
            inputs = (
                prediction,
                torch.zeros(12),
                zeros,
                zeros,
                zeros,
            )
            samples.append((inputs, torch.tensor([1.0, 2.0, 3.0])))
        test_loader = DataLoader(samples, batch_size=1, shuffle=False)
        test_loader.sample_metadata = [
            {
                'battery_id': 'cell-a',
                'window_start': 2,
                'cycle_index': 12,
                'degradation_stage': 'early',
            },
            {
                'battery_id': 'cell-b',
                'window_start': 5,
                'cycle_index': 25,
                'degradation_stage': 'late',
            },
        ]
        self.assertEqual(len(next(iter(test_loader))), 2)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            experiment = object.__new__(Exp_Long_Term_Forecast1)
            experiment.model = ExplainabilityModel()
            experiment.device = torch.device('cpu')
            experiment.args = SimpleNamespace(
                model='iMOE_SDR',
                inverse='no',
                checkpoints=checkpoint_dir,
                dataset='UL-NCA',
                condition='example',
            )
            experiment._get_data = lambda: (
                None,
                None,
                test_loader,
                None,
            )

            experiment.test('trial')

            export_path = os.path.join(
                checkpoint_dir,
                'iMOE_SDR',
                'UL-NCA',
                'example',
                'trial',
                'sdr_explainability.csv',
            )
            exported = pd.read_csv(export_path)
            metrics_path = os.path.join(
                checkpoint_dir,
                'iMOE_SDR',
                'UL-NCA',
                'example',
                'trial',
                'metrics.json',
            )
            with open(metrics_path, encoding='utf-8') as metrics_file:
                saved_metrics = json.load(metrics_file)

        self.assertEqual(exported['battery_id'].tolist(), ['cell-a', 'cell-b'])
        self.assertEqual(exported['cycle_index'].tolist(), [12, 25])
        self.assertTrue(torch.allclose(
            torch.tensor(exported['fusion_gate'].to_numpy()),
            torch.tensor([0.2, 0.7], dtype=torch.float64),
        ))
        self.assertEqual(
            [
                column
                for column in exported.columns
                if column.endswith('_fused_weight')
            ],
            [
                'expert_1_fused_weight',
                'expert_2_fused_weight',
                'expert_3_fused_weight',
            ],
        )
        self.assertEqual(saved_metrics['prediction_shape'], [2, 3])
        self.assertEqual(set(saved_metrics['global']), {
            'rmse',
            'mae',
            'mape_percent',
            'r2',
        })
        expected_metrics = _global_metrics(
            [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        )
        for name, expected in expected_metrics.items():
            self.assertAlmostEqual(
                saved_metrics['global'][name],
                expected,
                places=4,
            )


if __name__ == '__main__':
    unittest.main()
