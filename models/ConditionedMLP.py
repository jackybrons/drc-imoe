import torch
import torch.nn as nn


class Model(nn.Module):
    """MLP baseline conditioned on battery features and future operation."""

    def __init__(self, args):
        super().__init__()
        hidden_dim = args.hidden_dim
        feature_dim = 6 if args.dataset == 'TPSL' else 12

        self.curve_encoder = nn.Linear(args.seq_len, hidden_dim)
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        self.charge_encoder = nn.Linear(args.pred_len, hidden_dim)
        self.discharge_encoder = nn.Linear(args.pred_len, hidden_dim)
        self.temperature_encoder = nn.Linear(args.pred_len, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_projection = nn.Linear(hidden_dim, args.pred_len)
        self.activation = nn.ReLU()

    def forward(
        self,
        capacity_increment,
        features,
        charge_current,
        discharge_current,
        temperature,
    ):
        encoded_inputs = [
            self.activation(self.curve_encoder(capacity_increment)),
            self.activation(self.feature_encoder(features)),
            self.activation(self.charge_encoder(charge_current)),
            self.activation(self.discharge_encoder(discharge_current)),
            self.activation(self.temperature_encoder(temperature)),
        ]
        fused_features = torch.cat(encoded_inputs, dim=1)
        hidden = self.fusion(fused_features)
        prediction = self.output_projection(hidden)
        return prediction, hidden
