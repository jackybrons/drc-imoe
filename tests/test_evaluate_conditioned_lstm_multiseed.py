from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments.evaluation import evaluate_conditioned_lstm_multiseed as baseline_eval
from experiments.evaluation import evaluate_baselines_multiseed as unified_eval


def test_locked_configuration():
    args = baseline_eval.make_experiment_args(
        'TPSL', 'Arbitrary', Path('output'), 2025, 'cpu', 0, skip_test=True
    )

    assert args.model == 'ConditionedLSTM'
    assert args.hidden_dim == 128
    assert args.learning_rate == 1e-4
    assert args.batch_size == 32
    assert args.dataaccess == 100
    assert args.soc == 20
    assert args.seq_len == args.pred_len == 50
    assert args.train_epochs == 1500
    assert args.patience == 250
    assert args.eval_batch_size == 64
    assert args.selection_metric == 'prediction_mse'
    assert args.skip_test is True
    with pytest.raises(ValueError, match='Unsupported dataset/condition'):
        baseline_eval.locked_hidden_dim('TPSL', 'Fixed')


@pytest.mark.parametrize('model', baseline_eval.SUPPORTED_MODELS)
def test_unified_entry_builds_every_supported_baseline_configuration(model):
    args = unified_eval.make_experiment_args(
        'TPSL',
        'Arbitrary',
        Path('output'),
        2025,
        'cpu',
        0,
        skip_test=True,
        model=model,
    )

    assert args.model == model
    assert args.hidden_dim == 128
    assert args.d_model == 64
    assert args.e_layers == 2
    assert args.d_ff == 64
    assert args.dropout == 0.2
    assert args.patch_size == 2


def test_resume_existing_cli_flag_is_explicit(monkeypatch, tmp_path):
    required_args = [
        'evaluate_baselines_multiseed.py',
        '--dataset',
        'UL-NCA',
        '--condition',
        'CY25-025_1',
        '--output_dir',
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, 'argv', required_args)
    assert baseline_eval.parse_args().resume_existing is False

    monkeypatch.setattr(sys, 'argv', [*required_args, '--resume_existing'])
    assert baseline_eval.parse_args().resume_existing is True


def test_two_stage_protocol_locks_all_checkpoints_before_test(monkeypatch, tmp_path):
    events = []
    target = torch.linspace(1.0, 2.0, 50).repeat(2, 1)
    inputs = (
        target.clone(),
        torch.zeros(2, 12),
        torch.zeros(2, 50),
        torch.zeros(2, 50),
        torch.zeros(2, 50),
    )
    loader = [(inputs, target)]

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.9))

        def forward(self, curve, features, charge, discharge, temperature):
            del features, charge, discharge, temperature
            prediction = curve * self.scale
            return prediction, prediction

    class FakeExperiment:
        def __init__(self, args):
            self.args = args
            self.device = torch.device('cpu')
            self.model = FakeModel()

        def train(self, setting):
            events.append(('train', self.args.seed, self.args.skip_test))
            checkpoint = (
                Path(self.args.checkpoints)
                / self.args.model
                / self.args.dataset
                / self.args.condition
                / setting
                / 'checkpoint.pth'
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), checkpoint)

        def _load_checkpoint(self, checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    def fake_loader(args):
        events.append(('loader', args.seed, args.skip_test))
        return loader, loader, None if args.skip_test else loader, None

    monkeypatch.setattr(baseline_eval, 'Exp_Long_Term_Forecast1', FakeExperiment)
    monkeypatch.setattr(baseline_eval, 'DATA_LOADERS', {'UL-NCA': fake_loader})
    monkeypatch.setattr(baseline_eval, 'set_seed', lambda seed: None)

    summary = baseline_eval.run_locked_multiseed(
        'UL-NCA',
        'CY25-025_1',
        tmp_path,
        device='cpu',
        checkpoint_dir=tmp_path / 'selection',
    )

    train_positions = [i for i, event in enumerate(events) if event[0] == 'train']
    test_positions = [
        i for i, event in enumerate(events)
        if event[0] == 'loader' and event[2] is False
    ]
    assert len(train_positions) == 3
    assert all(events[position][2] is True for position in train_positions)
    assert len(test_positions) == 3
    assert max(train_positions) < min(test_positions)

    assert summary['declared_seeds'] == [2025, 2026, 2027]
    assert summary['runtime']['hidden_dim'] == 64
    assert summary['parameters'] == {'total': 1, 'trainable': 1}
    assert summary['protocol']['selection_test_policy'] == (
        'test split not loaded during checkpoint selection'
    )
    assert (tmp_path / 'selection_summary.json').is_file()
    assert (tmp_path / 'aggregate.json').is_file()
    assert (tmp_path / 'summary.json').is_file()
    for seed in baseline_eval.LOCKED_SEEDS:
        seed_dir = tmp_path / f'seed_{seed}'
        assert (seed_dir / 'pred_values.npy').is_file()
        assert (seed_dir / 'true_values.npy').is_file()
        assert (seed_dir / 'metrics.json').is_file()
        assert np.load(seed_dir / 'pred_values.npy').shape == (2, 50)


def test_resume_existing_skips_only_completed_seed_training(
    monkeypatch, tmp_path, capsys
):
    trained_seeds = []
    loaded_checkpoints = []
    target = torch.linspace(1.0, 2.0, 50).repeat(2, 1)
    inputs = (
        target.clone(),
        torch.zeros(2, 12),
        torch.zeros(2, 50),
        torch.zeros(2, 50),
        torch.zeros(2, 50),
    )
    loader = [(inputs, target)]

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.9))

        def forward(self, curve, features, charge, discharge, temperature):
            del features, charge, discharge, temperature
            prediction = curve * self.scale
            return prediction, prediction

    class FakeExperiment:
        def __init__(self, args):
            self.args = args
            self.device = torch.device('cpu')
            self.model = FakeModel()

        def train(self, setting):
            trained_seeds.append(self.args.seed)
            checkpoint = (
                Path(self.args.checkpoints)
                / self.args.model
                / self.args.dataset
                / self.args.condition
                / setting
                / 'checkpoint.pth'
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), checkpoint)

        def _load_checkpoint(self, checkpoint):
            loaded_checkpoints.append((self.args.seed, self.args.skip_test))
            self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    existing_checkpoint = (
        tmp_path
        / 'selection'
        / 'ConditionedLSTM'
        / 'UL-NCA'
        / 'CY25-025_1'
        / 'ConditionedLSTM_seed_2025'
        / 'checkpoint.pth'
    )
    existing_checkpoint.parent.mkdir(parents=True)
    torch.save(FakeModel().state_dict(), existing_checkpoint)

    monkeypatch.setattr(baseline_eval, 'Exp_Long_Term_Forecast1', FakeExperiment)
    monkeypatch.setattr(
        baseline_eval,
        'DATA_LOADERS',
        {'UL-NCA': lambda args: (loader, loader, None if args.skip_test else loader, None)},
    )
    monkeypatch.setattr(baseline_eval, 'set_seed', lambda seed: None)

    summary = baseline_eval.run_locked_multiseed(
        'UL-NCA',
        'CY25-025_1',
        tmp_path,
        device='cpu',
        resume_existing=True,
        checkpoint_dir=tmp_path / 'selection',
    )

    assert trained_seeds == [2026, 2027]
    assert (2025, True) in loaded_checkpoints
    assert (2026, True) not in loaded_checkpoints
    assert (2027, True) not in loaded_checkpoints
    assert len(summary['seeds']) == 3
    output = capsys.readouterr().out
    assert '[seed 2025] resuming existing checkpoint' in output
    assert '[seed 2026] training' in output
    assert '[seed 2027] training' in output
