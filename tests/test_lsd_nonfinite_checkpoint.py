from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.loader import BatteryDataset2
from models.iMOE import Model


class LsdCheckpointFiniteOutputTest(unittest.TestCase):
    def test_battery_85_window_508_produces_only_finite_predictions(self):
        torch.manual_seed(2025)
        torch.set_num_threads(1)

        source_path = REPO_ROOT / "datasets" / "processed" / "LSD" / "85.csv"
        checkpoint_path = (
            REPO_ROOT
            / "checkpoints"
            / "imported"
            / "local"
            / "locked_lsd_top4_multiseed"
            / "trials"
            / "trial_000"
            / "iMOE"
            / "LSD"
            / "LSD"
            / "iMOE_trial_000"
            / "checkpoint.pth"
        )

        required_columns = [
            "Capacity_Increment",
            "Relaxation_Voltage",
            "Charge_Current",
            "Discharge_Current",
            "Temperature",
            "Discharge_Capacity",
        ]
        data = pd.read_csv(source_path, usecols=required_columns).iloc[508:558]
        data = data.reset_index(drop=True)
        inputs, _ = BatteryDataset2(data, window_size=50, soc=20)[0]

        args = SimpleNamespace(
            num_experts=5,
            seq_len=50,
            pred_len=50,
            dataset="LSD",
            top_k=2,
            alpha=10,
            hidden_dim=64,
        )
        model = Model(args)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("capacity_fcs."):
                key = "capacity_fc." + key[len("capacity_fcs.") :]
            elif key.startswith("relaxation_fc."):
                key = "features_fc." + key[len("relaxation_fc.") :]
            remapped_state_dict[key] = value
        model.load_state_dict(remapped_state_dict)
        model.eval()

        with torch.no_grad():
            output, _ = model(*(tensor.unsqueeze(0) for tensor in inputs))

        nonfinite_count = int((~torch.isfinite(output)).sum().item())
        self.assertEqual((1, 50), tuple(output.shape))
        self.assertEqual(
            0,
            nonfinite_count,
            "dataset/LSD/85.csv window 508 produced "
            f"{nonfinite_count} non-finite predictions",
        )


if __name__ == "__main__":
    unittest.main()
