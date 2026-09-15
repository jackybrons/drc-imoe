import argparse

import torch

from models.PATCHTST import Model as PatchTST
from models.Informer import Model as Informer


def make_args():
    return argparse.Namespace(
        seq_len=50,
        pred_len=50,
        patch_size=2,
        d_model=64,
        d_ff=64,
        dropout=0.0,
        e_layers=2,
    )


def model_inputs(curve):
    batch_size = curve.shape[0]
    features = torch.zeros(batch_size, 12)
    conditions = [torch.zeros(batch_size, 50) for _ in range(3)]
    return curve, features, *conditions


def test_patchtst_does_not_mix_samples_within_a_batch():
    torch.manual_seed(2025)
    model = PatchTST(make_args()).eval()
    focal_curve = torch.randn(1, 50)
    companion_a = torch.zeros(1, 50)
    companion_b = torch.ones(1, 50)

    with torch.no_grad():
        prediction_a, _ = model(*model_inputs(torch.cat([focal_curve, companion_a])))
        prediction_b, _ = model(*model_inputs(torch.cat([focal_curve, companion_b])))

    torch.testing.assert_close(prediction_a[0], prediction_b[0])


def test_informer_evaluation_is_deterministic():
    torch.manual_seed(2025)
    model = Informer(make_args()).eval()
    curve = torch.randn(2, 50)

    with torch.no_grad():
        prediction_a, _ = model(*model_inputs(curve))
        prediction_b, _ = model(*model_inputs(curve))

    torch.testing.assert_close(prediction_a, prediction_b)
