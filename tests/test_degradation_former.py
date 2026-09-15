import argparse

import pytest
import torch

from models.DegradationFormer import Model


def make_args(dataset, **overrides):
    arguments = dict(
        dataset=dataset,
        seq_len=50,
        pred_len=40,
        hidden_dim=32,
    )
    arguments.update(overrides)
    return argparse.Namespace(**arguments)


@pytest.mark.parametrize(
    ('dataset', 'feature_dim'),
    [('UL-NCA', 12), ('TPSL', 6)],
)
def test_forward_backward_shapes_and_values(dataset, feature_dim):
    torch.manual_seed(2026)
    model = Model(make_args(dataset))
    batch_size = 3
    curve = torch.randn(batch_size, 50)
    features = torch.randn(batch_size, feature_dim)
    future_conditions = [torch.randn(batch_size, 40) for _ in range(3)]

    prediction, trend = model(curve, features, *future_conditions)

    assert prediction.shape == (batch_size, 40)
    assert trend.shape == (batch_size, 40)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(trend).all()

    prediction.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize('condition_index', [0, 1, 2])
def test_each_future_condition_changes_the_prediction(condition_index):
    torch.manual_seed(2026)
    model = Model(make_args('UL-NCA')).eval()
    curve = torch.randn(2, 50)
    features = torch.randn(2, 12)
    conditions = [torch.zeros(2, 40) for _ in range(3)]

    with torch.no_grad():
        baseline, _ = model(curve, features, *conditions)
        changed_conditions = [condition.clone() for condition in conditions]
        changed_conditions[condition_index][:, 10:20] = 1.0
        changed, _ = model(curve, features, *changed_conditions)

    assert not torch.allclose(baseline, changed)


def test_architecture_has_no_routing_or_expert_modules():
    model = Model(make_args('UL-NCA'))
    forbidden_terms = ('moe', 'route', 'router', 'expert', 'gate')

    module_semantics = [
        f'{name}:{module.__class__.__name__}'.lower()
        for name, module in model.named_modules()
    ]

    assert all(
        term not in semantic
        for semantic in module_semantics
        for term in forbidden_terms
    )
    assert not hasattr(model, 'num_experts')
    assert not hasattr(model, 'top_k')


def test_disable_residual_returns_the_explicit_trend():
    torch.manual_seed(2026)
    model = Model(make_args('UL-NCA', disable_residual=True)).eval()
    curve = torch.randn(2, 50)
    features = torch.randn(2, 12)
    conditions = [torch.randn(2, 40) for _ in range(3)]

    with torch.no_grad():
        prediction, trend = model(curve, features, *conditions)

    torch.testing.assert_close(prediction, trend)
