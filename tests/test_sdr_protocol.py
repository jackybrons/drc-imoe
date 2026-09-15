from types import SimpleNamespace
import unittest

from experiments.main.run import build_parser, build_setting
from experiments.tuning.tune_validation import build_search_space, validate_sdr_locked_args


class TestSDRLockedProtocol(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            workflow='iMOE_SDR',
            selection_only=True,
            seeds=[2025, 2026, 2027],
            max_trials=3,
            search_seed=2025,
            dataset='UL-NCA',
            condition='CY25-025_1',
            hidden_dims=[64],
            learning_rates=[1e-4],
            num_experts_values=[5],
            baseline_top_k_values=[2],
            csr_top_k_values=[4],
            router_alpha_values=[10],
            curve_channels_values=[4],
            diversity_weights=[0.5],
            soc_values=[20],
            seq_len=50,
            pred_len=50,
            dataaccess=100,
            batch_size=32,
            train_epochs=1500,
            patience=250,
            device='cuda',
            gpu=0,
        )

    def test_fixed_three_seed_configs_vary_only_by_seed(self):
        args = self._args()
        validate_sdr_locked_args(args)

        configs = build_search_space(
            args.workflow,
            args.seeds,
            args.hidden_dims,
            args.learning_rates,
            args.num_experts_values,
            args.baseline_top_k_values,
            args.csr_top_k_values,
            args.router_alpha_values,
            args.curve_channels_values,
            args.diversity_weights,
            args.soc_values,
        )

        self.assertEqual([config['seed'] for config in configs], args.seeds)
        self.assertEqual(len(configs), 3)
        locked = [
            {key: value for key, value in config.items() if key != 'seed'}
            for config in configs
        ]
        self.assertEqual(locked, [locked[0]] * 3)
        self.assertEqual({config['router_alpha'] for config in configs}, {10})
        self.assertFalse(
            any('gate' in key or 'fusion' in key for key in configs[0])
        )

    def test_rejects_alpha_candidates_instead_of_searching_them(self):
        args = self._args()
        args.router_alpha_values = [10, 20]

        with self.assertRaisesRegex(ValueError, 'router_alpha_values.*locked'):
            validate_sdr_locked_args(args)

    def test_run_cli_exposes_sdr_ablation_switches(self):
        args = build_parser().parse_args([
            '--model', 'iMOE_SDR',
            '--fusion_mode', 'fixed',
            '--fusion_gate_bias', '-1.5',
            '--disable_noisy_routing',
            '--disable_top_k',
        ])

        self.assertEqual(args.fusion_mode, 'fixed')
        self.assertEqual(args.fusion_gate_bias, -1.5)
        self.assertTrue(args.disable_noisy_routing)
        self.assertTrue(args.disable_top_k)

    def test_sdr_settings_separate_all_routing_ablations(self):
        base = build_parser().parse_args(['--model', 'iMOE_SDR'])
        settings = set()
        for fusion_mode, disable_noise, disable_top_k in (
            ('learned', False, False),
            ('fixed', False, False),
            ('learned', True, False),
            ('learned', False, True),
        ):
            args = SimpleNamespace(**vars(base))
            args.fusion_mode = fusion_mode
            args.disable_noisy_routing = disable_noise
            args.disable_top_k = disable_top_k
            settings.add(build_setting(args, 0))

        self.assertEqual(len(settings), 4)
        self.assertTrue(any('_fusionfixed_fgb0.0_noiseon_topkon' in value for value in settings))
        self.assertTrue(any('_fusionlearned_fgb0.0_noiseoff_topkon' in value for value in settings))
        self.assertTrue(any('_fusionlearned_fgb0.0_noiseon_topkoff' in value for value in settings))
        self.assertTrue(all('_btk2_ctk4_cc4_' in value for value in settings))

        biased = SimpleNamespace(**vars(base))
        biased.fusion_gate_bias = 1.25
        self.assertNotEqual(build_setting(base, 0), build_setting(biased, 0))
        self.assertIn('_fgb1.25_', build_setting(biased, 0))


if __name__ == '__main__':
    unittest.main()
