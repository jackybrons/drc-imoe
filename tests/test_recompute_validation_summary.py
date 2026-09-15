import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.tuning.recompute_validation_summary import recompute_validation_summary


class TestRecomputeValidationSummary(unittest.TestCase):
    def test_recomputes_from_checkpoints_without_loading_test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = []
            for trial_id, top_k in enumerate((3, 4)):
                baseline = root / f'baseline_{trial_id}.pth'
                curve = root / f'curve_{trial_id}.pth'
                baseline.touch()
                curve.touch()
                trials.append({
                    'trial_id': trial_id,
                    'config': {
                        'seed': 2025,
                        'hidden_dim': 8,
                        'learning_rate': 1e-4,
                        'num_experts': 5,
                        'diverloss': 0.5,
                        'soc': 20,
                        'baseline_top_k': 2,
                        'csr_top_k': top_k,
                        'router_alpha': 10,
                        'curve_channels': 4,
                    },
                    'validation_mse': 999.0,
                    'baseline_validation_mse': 999.0,
                    'curve_validation_mse': 999.0,
                    'alpha': 0.0,
                    'baseline_checkpoint': str(baseline),
                    'curve_checkpoint': str(curve),
                })
            source = {
                'protocol': {
                    'selection_split': 'validation',
                    'selection_metric': 'global_mse',
                    'test_policy': 'not loaded or evaluated in selection-only mode',
                },
                'runtime': {
                    'seq_len': 50,
                    'pred_len': 50,
                    'batch_size': 32,
                    'dataaccess': 100,
                },
                'workflow': 'dual_router',
                'dataset': 'UL-NCA',
                'condition': 'CY25-025_1',
                'trials': trials,
            }
            source_path = root / 'source.json'
            output_path = root / 'evalfix' / 'search_summary.json'
            source_path.write_text(json.dumps(source), encoding='utf-8')
            loader_calls = []

            def fake_loader(args):
                loader_calls.append(args.skip_test)
                return [], 'validation-only', None, None

            def fake_load(cli_args, config, model_name, checkpoint, device):
                return Path(checkpoint).stem, None

            def fake_predict(baseline_model, curve_model, loader, device):
                curve_value = 1.0 if curve_model == 'curve_0' else 2.0
                return (
                    np.zeros((1, 1), dtype=np.float32),
                    np.full((1, 1), curve_value, dtype=np.float32),
                    np.full((1, 1), 1.5, dtype=np.float32),
                )

            with (
                patch(
                    'experiments.tuning.recompute_validation_summary.DATA_LOADERS',
                    {'UL-NCA': fake_loader},
                ),
                patch(
                    'experiments.tuning.recompute_validation_summary.load_trained_model',
                    side_effect=fake_load,
                ),
                patch(
                    'experiments.tuning.recompute_validation_summary.predict_pair',
                    side_effect=fake_predict,
                ),
            ):
                result = recompute_validation_summary(source_path, output_path)

            self.assertEqual(loader_calls, [True, True])
            self.assertEqual(result['recompute']['training_jobs'], 0)
            self.assertEqual(result['winner']['trial_id'], 1)
            self.assertEqual(result['winner']['alpha'], 0.75)
            self.assertEqual(result['winner']['validation_mse'], 0.0)
            self.assertTrue(output_path.is_file())

    def test_refuses_to_overwrite_source_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / 'summary.json'
            summary_path.write_text('{}', encoding='utf-8')

            with self.assertRaises(ValueError):
                recompute_validation_summary(summary_path, summary_path)


if __name__ == '__main__':
    unittest.main()
