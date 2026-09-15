import unittest

import numpy as np
import torch
import torch.nn as nn

from experiments.evaluation.evaluate_dual_router import (
    blend_predictions,
    global_metrics,
    optimal_curve_alpha,
    predict_pair,
)


class ModeRecordingModel(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        self.observed_training_modes = []

    def forward(self, prediction, features, charge, discharge, temperature):
        self.observed_training_modes.append(self.training)
        gates = torch.ones((prediction.shape[0], 1))
        return prediction + self.offset, gates


class TestDualRouterEvaluation(unittest.TestCase):
    def test_pair_prediction_forces_inference_mode(self):
        baseline = ModeRecordingModel(offset=0.0)
        curve = ModeRecordingModel(offset=1.0)
        baseline.train()
        curve.train()
        prediction = torch.zeros((2, 3))
        loader = [(
            (prediction, prediction, prediction, prediction, prediction),
            prediction,
        )]

        predict_pair(baseline, curve, loader, torch.device('cpu'))

        self.assertEqual(baseline.observed_training_modes, [False])
        self.assertEqual(curve.observed_training_modes, [False])

    def test_closed_form_alpha(self):
        baseline = np.zeros((2, 2))
        curve = np.ones((2, 2))
        true_values = np.full((2, 2), 0.25)

        self.assertAlmostEqual(
            optimal_curve_alpha(baseline, curve, true_values),
            0.25,
        )

    def test_alpha_is_clipped_to_convex_bounds(self):
        baseline = np.zeros(3)
        curve = np.ones(3)

        self.assertEqual(optimal_curve_alpha(baseline, curve, np.full(3, -1.0)), 0.0)
        self.assertEqual(optimal_curve_alpha(baseline, curve, np.full(3, 2.0)), 1.0)

    def test_blend_and_global_metrics(self):
        baseline = np.array([[1.0, 2.0], [3.0, 4.0]])
        curve = np.array([[2.0, 3.0], [4.0, 5.0]])
        true_values = np.array([[1.5, 2.5], [3.5, 4.5]])
        blended = blend_predictions(baseline, curve, 0.5)
        metrics = global_metrics(blended, true_values)

        np.testing.assert_allclose(blended, true_values)
        self.assertEqual(metrics['rmse'], 0.0)
        self.assertEqual(metrics['mae'], 0.0)
        self.assertEqual(metrics['mape_percent'], 0.0)
        self.assertEqual(metrics['r2'], 1.0)


if __name__ == '__main__':
    unittest.main()
