import torch
import torch.nn as nn


class Model(nn.Module):
    """Condition-aware linear trend encoder followed by FORNN."""

    def __init__(self, args):
        super().__init__()
        self.trend_projection = nn.Linear(args.seq_len, args.pred_len)
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=args.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_final = nn.Linear(2 * args.hidden_dim, 1)

    def forward(
        self,
        capacity_increment,
        features,
        charge_current,
        discharge_current,
        temperature,
    ):
        del features
        degradation_trend = self.trend_projection(capacity_increment)
        recurrent_input = torch.stack(
            (
                degradation_trend,
                charge_current,
                discharge_current,
                temperature,
            ),
            dim=2,
        )
        recurrent_output, _ = self.lstm(recurrent_input)
        prediction = self.fc_final(recurrent_output).squeeze(-1)
        return prediction, degradation_trend
