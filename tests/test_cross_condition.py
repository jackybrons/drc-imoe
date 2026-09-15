import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from datasets import loader as dataloader
from experiments.transfer.evaluate_cross_condition import run_cross_condition_evaluation


class FakeBatteryDataset:
    constructions = []

    def __init__(self, data, window_size, scaler_features=None, soc=None):
        del window_size, soc
        self.path = data.attrs["path"]
        self.scaler_features = scaler_features
        self.constructions.append(self)

    def __len__(self):
        return 19

    def __getitem__(self, index):
        feature = torch.arange(6, dtype=torch.float32) + index
        sequence = torch.zeros(50, dtype=torch.float32)
        inputs = (sequence, feature, sequence, sequence, sequence)
        return inputs, sequence


def loader_args(**overrides):
    values = {
        "condition": "Arbitrary",
        "test_condition": "Fixed",
        "pred_len": 50,
        "soc": 20,
        "dataaccess": 100,
        "batch_size": 32,
        "skip_test": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_read_csv(path):
    data = pd.DataFrame({"QV_Curve": ["[0.0]"]})
    data.attrs["path"] = str(path).replace("\\", "/")
    return data


def test_tpsl_loader_fits_source_and_transforms_only_target_test(monkeypatch):
    FakeBatteryDataset.constructions = []
    monkeypatch.setattr(dataloader, "BatteryDataset1", FakeBatteryDataset)
    monkeypatch.setattr(dataloader.pd, "read_csv", fake_read_csv)

    train_loader, val_loader, test_loader, scaler = dataloader.TPSL_trainloader(
        loader_args()
    )

    self_paths = [item.path for item in FakeBatteryDataset.constructions]
    test_items = FakeBatteryDataset.constructions[-4:]
    assert len(train_loader.dataset) == 38 * 19
    assert len(val_loader.dataset) == 2 * 19
    assert len(test_loader.dataset) == 4 * 19
    assert all("datasets/processed/TPSL-Arbitrary/" in path for path in self_paths[:-4])
    assert [item.path.rsplit("/", 2)[-2] for item in test_items] == [
        "#23",
        "#35",
        "#48",
        "#57",
    ]
    assert all(item.scaler_features is scaler for item in test_items)


def locked_tpsl_summary(root):
    trials = []
    for trial_id, seed in enumerate((2025, 2026)):
        baseline = root / f"baseline_{seed}.pth"
        curve = root / f"curve_{seed}.pth"
        baseline.touch()
        curve.touch()
        trials.append(
            {
                "trial_id": trial_id,
                "config": {
                    "seed": seed,
                    "hidden_dim": 128,
                    "learning_rate": 1e-4,
                    "num_experts": 5,
                    "diverloss": 0.5,
                    "soc": 20,
                    "baseline_top_k": 2,
                    "csr_top_k": 4,
                    "router_alpha": 10,
                    "curve_channels": 4,
                },
                "validation_mse": 0.1 + trial_id,
                "alpha": 0.25 + trial_id * 0.5,
                "baseline_checkpoint": str(baseline),
                "curve_checkpoint": str(curve),
            }
        )
    return {
        "protocol": {
            "test_policy": "not loaded or evaluated in selection-only mode"
        },
        "workflow": "dual_router",
        "dataset": "TPSL",
        "condition": "Arbitrary",
        "search": {"seeds": [2025, 2026]},
        "runtime": {
            "seq_len": 50,
            "pred_len": 2,
            "batch_size": 32,
            "dataaccess": 100,
        },
        "trials": trials,
    }


def test_cross_condition_evaluator_reuses_source_alpha_and_target_test(
    tmp_path, monkeypatch
):
    from experiments.transfer import evaluate_cross_condition

    source_summary = locked_tpsl_summary(tmp_path)
    summary_path = tmp_path / "search_summary.json"
    summary_path.write_text(json.dumps(source_summary), encoding="utf-8")
    loader_calls = []

    def fake_loader(args):
        loader_calls.append(args)
        return object(), object(), object(), object()

    predictions = [
        (
            torch.tensor([[0.0, 1.0]]).numpy(),
            torch.tensor([[2.0, 3.0]]).numpy(),
            torch.tensor([[0.5, 1.5]]).numpy(),
        ),
        (
            torch.tensor([[1.0, 2.0]]).numpy(),
            torch.tensor([[3.0, 4.0]]).numpy(),
            torch.tensor([[2.5, 3.5]]).numpy(),
        ),
    ]
    monkeypatch.setattr(evaluate_cross_condition, "DATA_LOADERS", {"TPSL": fake_loader})
    monkeypatch.setattr(
        evaluate_cross_condition,
        "load_trained_model",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        evaluate_cross_condition,
        "predict_pair",
        lambda *args, **kwargs: predictions.pop(0),
    )

    output_dir = tmp_path / "cross_condition"
    result = run_cross_condition_evaluation(
        summary_path, "Fixed", output_dir, device="cpu"
    )

    assert len(loader_calls) == 1
    assert loader_calls[0].condition == "Arbitrary"
    assert loader_calls[0].test_condition == "Fixed"
    assert result["source_condition"] == "Arbitrary"
    assert result["target_condition"] == "Fixed"
    assert result["target_test_cells"] == ["#23", "#35", "#48", "#57"]
    assert [item["validation_alpha"] for item in result["seeds"]] == [0.25, 0.75]
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "seed_2025" / "baseline_pred.npy").is_file()
    assert (output_dir / "seed_2025" / "curve_pred.npy").is_file()
    assert (output_dir / "seed_2025" / "ensemble_pred.npy").is_file()
    assert (output_dir / "seed_2025" / "true_values.npy").is_file()


def test_cross_condition_evaluator_rejects_same_source_and_target(tmp_path):
    source_summary = locked_tpsl_summary(tmp_path)
    summary_path = tmp_path / "search_summary.json"
    summary_path.write_text(json.dumps(source_summary), encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        run_cross_condition_evaluation(
            summary_path, "Arbitrary", tmp_path / "same_condition"
        )


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (np.zeros((1, 1)), np.zeros((1, 2)), "prediction shape"),
        (np.array([[np.nan, 0.0]]), np.zeros((1, 2)), "non-finite"),
        (np.zeros((1, 2)), np.array([[0.0, np.inf]]), "non-finite"),
        (np.zeros((1, 2)), np.ones((1, 2)), "target values differ"),
    ],
)
def test_additional_baseline_prediction_validation(
    prediction, target, message
):
    from experiments.transfer import evaluate_cross_condition

    with pytest.raises(ValueError, match=message):
        evaluate_cross_condition._validate_baseline_prediction(
            "Informer", prediction, target, np.zeros((1, 2))
        )


def test_cross_condition_evaluator_can_join_conditioned_lstm_checkpoints(
    tmp_path, monkeypatch
):
    from experiments.transfer import evaluate_cross_condition

    source_summary = locked_tpsl_summary(tmp_path)
    summary_path = tmp_path / "search_summary.json"
    summary_path.write_text(json.dumps(source_summary), encoding="utf-8")
    conditioned_paths = {}
    for seed in (2025, 2026):
        checkpoint = tmp_path / f"conditioned_{seed}.pth"
        checkpoint.touch()
        conditioned_paths[seed] = checkpoint

    monkeypatch.setattr(
        evaluate_cross_condition,
        "DATA_LOADERS",
        {"TPSL": lambda args: (object(), object(), object(), object())},
    )
    monkeypatch.setattr(
        evaluate_cross_condition,
        "load_trained_model",
        lambda *args, **kwargs: (object(), object()),
    )
    dual_predictions = [
        (
            torch.tensor([[0.0, 1.0]]).numpy(),
            torch.tensor([[2.0, 3.0]]).numpy(),
            torch.tensor([[1.0, 2.0]]).numpy(),
        ),
        (
            torch.tensor([[0.0, 1.0]]).numpy(),
            torch.tensor([[2.0, 3.0]]).numpy(),
            torch.tensor([[1.0, 2.0]]).numpy(),
        ),
    ]
    monkeypatch.setattr(
        evaluate_cross_condition,
        "predict_pair",
        lambda *args, **kwargs: dual_predictions.pop(0),
    )
    monkeypatch.setattr(
        evaluate_cross_condition,
        "load_conditioned_lstm",
        lambda *args, **kwargs: object(),
    )
    conditioned_predictions = [
        (torch.tensor([[0.5, 1.5]]).numpy(), torch.tensor([[1.0, 2.0]]).numpy()),
        (torch.tensor([[1.5, 2.5]]).numpy(), torch.tensor([[1.0, 2.0]]).numpy()),
    ]
    monkeypatch.setattr(
        evaluate_cross_condition,
        "predict_single",
        lambda *args, **kwargs: conditioned_predictions.pop(0),
    )

    output_dir = tmp_path / "cross_condition"
    result = run_cross_condition_evaluation(
        summary_path,
        "Fixed",
        output_dir,
        conditioned_lstm_checkpoints=conditioned_paths,
    )

    assert "conditioned_lstm" in result["aggregate"]["models"]
    assert "conditioned_lstm" in result["seeds"][0]["metrics"]
    assert (
        output_dir / "seed_2025" / "conditioned_lstm_pred.npy"
    ).is_file()


def test_cross_condition_evaluator_can_join_multiple_baseline_summaries(
    tmp_path, monkeypatch
):
    from experiments.transfer import evaluate_cross_condition

    source_summary = locked_tpsl_summary(tmp_path)
    source_path = tmp_path / "search_summary.json"
    source_path.write_text(json.dumps(source_summary), encoding="utf-8")
    baseline_summaries = {}
    for model in ("Informer", "PATCHTST"):
        seeds = []
        for seed in (2025, 2026):
            checkpoint = tmp_path / f"{model}_{seed}.pth"
            checkpoint.touch()
            seeds.append({"seed": seed, "checkpoint": str(checkpoint)})
        path = tmp_path / f"{model}_summary.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": "TPSL",
                    "condition": "Arbitrary",
                    "model": model,
                    "declared_seeds": [2025, 2026],
                    "seeds": seeds,
                }
            ),
            encoding="utf-8",
        )
        baseline_summaries[model] = path

    monkeypatch.setattr(
        evaluate_cross_condition,
        "DATA_LOADERS",
        {"TPSL": lambda args: (object(), object(), object(), object())},
    )
    monkeypatch.setattr(
        evaluate_cross_condition,
        "load_trained_model",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        evaluate_cross_condition,
        "predict_pair",
        lambda *args, **kwargs: (
            torch.tensor([[0.0, 1.0]]).numpy(),
            torch.tensor([[2.0, 3.0]]).numpy(),
            torch.tensor([[1.0, 2.0]]).numpy(),
        ),
    )
    loaded_models = []

    def fake_load(model, *args, **kwargs):
        loaded_models.append(model)
        return object()

    monkeypatch.setattr(evaluate_cross_condition, "load_baseline_model", fake_load)
    monkeypatch.setattr(
        evaluate_cross_condition,
        "predict_single",
        lambda *args, **kwargs: (
            torch.tensor([[1.0, 2.0]]).numpy(),
            torch.tensor([[1.0, 2.0]]).numpy(),
        ),
    )

    result = run_cross_condition_evaluation(
        source_path,
        "Fixed",
        tmp_path / "cross_condition_multiple",
        baseline_checkpoints=baseline_summaries,
    )

    assert loaded_models == ["Informer", "PATCHTST", "Informer", "PATCHTST"]
    assert result["additional_baselines"] == ["informer", "patchtst"]
    assert set(result["aggregate"]["models"]) == {
        "baseline",
        "curve",
        "ensemble",
        "informer",
        "patchtst",
    }
