from types import SimpleNamespace
import unittest

import torch

from models.iMOE_CSR import Model


class TestIMOECSR(unittest.TestCase):
    def _args(self, dataset):
        return SimpleNamespace(
            dataset=dataset,
            seq_len=50,
            pred_len=50,
            num_experts=5,
            top_k=2,
            hidden_dim=8,
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

    def _assert_forward_backward(self, dataset):
        model = Model(self._args(dataset))
        inputs = self._inputs(dataset)
        outputs, weights = model(*inputs)

        self.assertEqual(outputs.shape, (4, 50))
        self.assertEqual(weights.shape, (4, 5))
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6))
        self.assertTrue(torch.all(weights.ne(0).sum(dim=1) <= model.top_k))

        loss = outputs.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_ul_forward_backward(self):
        self._assert_forward_backward('UL-NCA')

    def test_tpsl_forward_backward(self):
        self._assert_forward_backward('TPSL')

    def test_curve_changes_routing_with_fixed_features(self):
        model = Model(self._args('UL-NCA')).eval()
        with torch.no_grad():
            for branch in model.curve_encoder.branches:
                branch[0].weight.fill_(1.0)
                branch[0].bias.zero_()
            model.router_fc.weight.zero_()
            model.router_fc.bias.copy_(torch.tensor([0.0, 0.0, -10.0, -10.0, -10.0]))
            model.router_fc.weight[0, 12] = 1.0

        features = torch.zeros(2, 12)
        conditions = [torch.zeros(2, 50) for _ in range(3)]
        zero_curve = torch.zeros(2, 50)
        one_curve = torch.ones(2, 50)

        _, zero_weights = model(zero_curve, features, *conditions)
        _, one_weights = model(one_curve, features, *conditions)

        self.assertFalse(torch.allclose(zero_weights, one_weights))


if __name__ == '__main__':
    unittest.main()
