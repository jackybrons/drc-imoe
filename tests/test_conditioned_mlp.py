import argparse

import pytest
import torch

from experiments.core.exp_basic import Exp_Basic
from models.ConditionedMLP import Model
from models.iMOE import Model as IMOModel


def make_args(dataset):
    return argparse.Namespace(
        dataset=dataset,
        seq_len=50,
        pred_len=50,
        hidden_dim=64,
        num_experts=5,
        top_k=2,
        alpha=10,
        use_gpu=False,
    )


@pytest.mark.parametrize(
    ('dataset', 'feature_dim'),
    [('UL-NCA', 12), ('TPSL', 6)],
)
def test_conditioned_mlp_forward_backward_is_finite(dataset, feature_dim):
    torch.manual_seed(2025)
    model = Model(make_args(dataset))
    batch_size = 3
    curve = torch.randn(batch_size, 50)
    features = torch.randn(batch_size, feature_dim)
    conditions = [torch.randn(batch_size, 50) for _ in range(3)]

    prediction, auxiliary = model(curve, features, *conditions)

    assert prediction.shape == (batch_size, 50)
    assert auxiliary.shape == (batch_size, 64)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(auxiliary).all()

    prediction.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize('condition_index', [0, 1, 2])
def test_conditioned_mlp_uses_each_future_condition(condition_index):
    torch.manual_seed(2025)
    model = Model(make_args('UL-NCA')).eval()
    batch_size = 2
    curve = torch.randn(batch_size, 50)
    features = torch.randn(batch_size, 12)
    conditions = [torch.zeros(batch_size, 50) for _ in range(3)]

    with torch.no_grad():
        baseline, _ = model(curve, features, *conditions)
        changed_conditions = [condition.clone() for condition in conditions]
        changed_conditions[condition_index].fill_(1.0)
        changed, _ = model(curve, features, *changed_conditions)

    assert not torch.allclose(baseline, changed)


@pytest.mark.parametrize('dataset', ['UL-NCA', 'TPSL'])
def test_conditioned_mlp_parameter_count_is_comparable_to_imoe(dataset):
    args = make_args(dataset)
    baseline_count = sum(parameter.numel() for parameter in Model(args).parameters())
    imoe_count = sum(parameter.numel() for parameter in IMOModel(args).parameters())

    assert 0.5 * imoe_count <= baseline_count <= 2.0 * imoe_count


def test_conditioned_mlp_is_registered_without_diversity_routing():
    class Experiment(Exp_Basic):
        def _build_model(self):
            return self.model_dict['ConditionedMLP'].Model(self.args).float()

    experiment = Experiment(make_args('UL-NCA'))

    assert isinstance(experiment.model, Model)
    assert not hasattr(experiment.model, 'num_experts')
    assert not hasattr(experiment.model, 'top_k')
