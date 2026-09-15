import unittest

import torch
import torch.nn as nn

from experiments.evaluation.benchmark_compression import (
    BENCHMARK_SYSTEMS,
    LOCKED_SEEDS,
    _aggregate_seed_benchmarks,
    benchmark_inference,
    parameter_counts,
)


class RecordingInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.observed_training_modes = []
        self.observed_grad_modes = []

    def forward(self, inputs):
        self.observed_training_modes.append(self.training)
        self.observed_grad_modes.append(torch.is_grad_enabled())
        return inputs[0]


class TestCompressionBenchmark(unittest.TestCase):
    def test_main_protocol_uses_three_common_seeds(self):
        self.assertEqual(LOCKED_SEEDS, (2025, 2026, 2027))

    def test_eval_inference_mode_and_call_count_match_protocol(self):
        inference = RecordingInference().eval()
        batches = [
            (torch.zeros(2, 3),),
            (torch.zeros(3, 3),),
        ]

        result = benchmark_inference(
            inference,
            batches,
            torch.device('cpu'),
            warmup_runs=1,
            timed_runs=2,
        )

        expected_calls = (1 + 2) * len(batches)
        self.assertEqual(len(inference.observed_training_modes), expected_calls)
        self.assertEqual(inference.observed_training_modes, [False] * expected_calls)
        self.assertEqual(inference.observed_grad_modes, [False] * expected_calls)
        self.assertEqual(result['num_samples'], 5)
        self.assertEqual(result['warmup_runs'], 1)
        self.assertEqual(result['timed_runs'], 2)
        self.assertEqual(len(result['timings_ms']), 2)
        self.assertGreaterEqual(result['median_ms'], 0.0)
        self.assertGreaterEqual(result['ms_per_sample'], 0.0)
        self.assertGreater(result['samples_per_second'], 0.0)
        self.assertIsNone(result['peak_allocated_memory_bytes'])

    def test_four_system_aggregate_has_stable_efficiency_fields(self):
        seed_results = []
        for seed_index, seed in enumerate(LOCKED_SEEDS):
            timing = {
                system: {
                    'median_ms': float(seed_index + system_index + 1),
                    'ms_per_sample': 0.1 * (system_index + 1),
                    'samples_per_second': 100.0 / (system_index + 1),
                    'peak_allocated_memory_bytes': None,
                }
                for system_index, system in enumerate(BENCHMARK_SYSTEMS)
            }
            parameters = {
                system: {'total': 100 + index, 'trainable': 100 + index}
                for index, system in enumerate(BENCHMARK_SYSTEMS)
            }
            seed_results.append({
                'seed': seed,
                'timing': timing,
                'parameters': parameters,
                'speedup_dual_router_over_iMOE_SDR': 2.0,
            })

        aggregate = _aggregate_seed_benchmarks(seed_results)

        self.assertEqual(
            set(aggregate) - {'speedup_dual_router_over_iMOE_SDR'},
            set(BENCHMARK_SYSTEMS),
        )
        for system in BENCHMARK_SYSTEMS:
            self.assertIn('parameters', aggregate[system])
            self.assertIn('median_ms', aggregate[system])
            self.assertIn('ms_per_sample', aggregate[system])
            self.assertIn('samples_per_second', aggregate[system])
            self.assertEqual(
                aggregate[system]['peak_allocated_memory_bytes'],
                {'mean': None, 'std': None},
            )

    def test_parameter_counts_cover_total_and_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
        model[1].weight.requires_grad_(False)

        counts = parameter_counts(model)

        self.assertEqual(counts['total'], 21)
        self.assertEqual(counts['trainable'], 17)


if __name__ == '__main__':
    unittest.main()
