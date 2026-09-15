import argparse
import random

import numpy as np
import torch

from experiments.core.exp_forecasting import Exp_Long_Term_Forecast1
from models.DegradationFormer import Model
from experiments.main.run import build_parser, build_setting, seed_everything


def make_args(**overrides):
    arguments = dict(
        model='DegradationFormer',
        dataset='UL-NCA',
        seq_len=50,
        pred_len=20,
        hidden_dim=16,
        use_gpu=False,
        seed=7,
        dataaccess=100,
        des='test',
        disable_trend=False,
        disable_residual=False,
        test_condition=None,
    )
    arguments.update(overrides)
    return argparse.Namespace(**arguments)


def test_exp_builds_degradation_former_on_cpu_and_runs_forward():
    args = make_args()
    experiment = Exp_Long_Term_Forecast1(args)

    assert isinstance(experiment.model, Model)
    assert experiment.device.type == 'cpu'

    curve = torch.randn(2, args.seq_len)
    features = torch.randn(2, 12)
    conditions = [torch.randn(2, args.pred_len) for _ in range(3)]
    prediction, trend = experiment.model(curve, features, *conditions)

    assert prediction.shape == (2, args.pred_len)
    assert trend.shape == (2, args.pred_len)
    assert torch.isfinite(prediction).all()


def test_setting_separates_seed_ablation_and_cross_condition():
    baseline = build_setting(make_args(), 0)
    changed = build_setting(
        make_args(seed=11, disable_trend=True, test_condition='Fixed'),
        0,
    )

    assert baseline != changed
    assert '_seed7_' in baseline
    assert '_seed11_' in changed
    assert '_trendoff_' in changed
    assert '_testFixed_' in changed
    assert '_ex' not in baseline
    assert '_tk' not in baseline


def test_setting_separates_data_access_and_non_default_descriptions():
    baseline = build_setting(make_args(), 0)
    reduced_data = build_setting(make_args(dataaccess=20), 0)
    labeled = build_setting(make_args(des='main_seed'), 0)

    assert baseline != reduced_data
    assert baseline != labeled
    assert '_da100' in baseline
    assert '_da20' in reduced_data
    assert '_desmain_seed' in labeled
    assert '_destest' not in baseline


def test_parser_accepts_seed_ablations_and_test_condition():
    args = build_parser().parse_args(
        [
            '--model', 'DegradationFormer',
            '--seed', '19',
            '--disable_residual',
            '--test_condition', 'Fixed',
        ]
    )

    assert args.seed == 19
    assert args.disable_residual is True
    assert args.disable_trend is False
    assert args.test_condition == 'Fixed'


def test_seed_everything_controls_all_three_random_generators():
    seed_everything(23)
    first = (random.random(), np.random.rand(), torch.rand(1))
    seed_everything(23)
    second = (random.random(), np.random.rand(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])
