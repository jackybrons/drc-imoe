import argparse

import pytest
import torch

from experiments.core.exp_basic import Exp_Basic
from models.ConditionedLSTM import Model


def make_args():
    return argparse.Namespace(
        seq_len=50,
        pred_len=50,
        hidden_dim=64,
        use_gpu=False,
    )


@pytest.mark.parametrize('feature_dim', [12, 6])
def test_conditioned_lstm_forward_backward_is_finite(feature_dim):
    torch.manual_seed(2025)
    model = Model(make_args())
    batch_size = 3
    curve = torch.randn(batch_size, 50)
    features = torch.randn(batch_size, feature_dim)
    conditions = [torch.randn(batch_size, 50) for _ in range(3)]

    prediction, trend = model(curve, features, *conditions)

    assert prediction.shape == (batch_size, 50)
    assert trend.shape == (batch_size, 50)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(trend).all()

    prediction.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_conditioned_lstm_uses_future_conditions():
    torch.manual_seed(2025)
    model = Model(make_args()).eval()
    batch_size = 2
    curve = torch.randn(batch_size, 50)
    features = torch.randn(batch_size, 12)
    zeros = torch.zeros(batch_size, 50)

    with torch.no_grad():
        baseline, _ = model(curve, features, zeros, zeros, zeros)
        changed, _ = model(curve, features, zeros, zeros, torch.ones_like(zeros))

    assert not torch.allclose(baseline, changed)


def test_conditioned_lstm_parameter_count_and_registration():
    model = Model(make_args())
    assert sum(parameter.numel() for parameter in model.parameters()) == 38_519
    assert not hasattr(model, 'num_experts')
    assert not hasattr(model, 'top_k')

    class Experiment(Exp_Basic):
        def _build_model(self):
            return self.model_dict['ConditionedLSTM'].Model(self.args).float()

    experiment = Experiment(make_args())
    assert isinstance(experiment.model, Model)
