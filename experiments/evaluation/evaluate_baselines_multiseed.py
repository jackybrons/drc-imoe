"""Unified entry point for locked multi-seed baseline evaluation.

The implementation remains in the original ConditionedLSTM module so existing
commands and imports continue to work.
"""

from experiments.evaluation.evaluate_conditioned_lstm_multiseed import *  # noqa: F401,F403
from experiments.evaluation.evaluate_conditioned_lstm_multiseed import main


if __name__ == '__main__':
    main()
