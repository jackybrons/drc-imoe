from types import SimpleNamespace
import unittest

import torch

from models.iMOE_HD import Model


class TestIMOEHD(unittest.TestCase):
    def _args(self, dataset, top_k=2):
        return SimpleNamespace(
            dataset=dataset,
            seq_len=50,
            pred_len=50,
            num_experts=5,
            top_k=top_k,
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

        outputs.square().mean().backward()
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
        _, zero_weights = model(torch.zeros(2, 50), features, *conditions)
        _, one_weights = model(torch.ones(2, 50), features, *conditions)

        self.assertFalse(torch.allclose(zero_weights, one_weights))

    def test_shared_expert_gradient_and_gate_with_top_one(self):
        model = Model(self._args('UL-NCA', top_k=1))
        inputs = self._inputs('UL-NCA')
        outputs, weights = model(*inputs)

        router_input = torch.cat([
            inputs[1],
            model.curve_encoder(inputs[0]),
        ], dim=1)
        gate = model.shared_gate(router_input)
        self.assertTrue(torch.all((gate >= 0.0) & (gate <= 1.0)))
        self.assertTrue(torch.all(weights.ne(0).sum(dim=1) <= 1))

        target = torch.randn_like(outputs)
        (outputs - target).square().mean().backward()
        shared_gradients = [parameter.grad for parameter in model.shared_expert.parameters()]
        self.assertTrue(all(gradient is not None for gradient in shared_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in shared_gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) > 0 for gradient in shared_gradients))


if __name__ == '__main__':
    unittest.main()
