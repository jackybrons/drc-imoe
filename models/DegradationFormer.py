import torch
import torch.nn as nn


class Model(nn.Module):
    """Condition-modulated trend-residual forecaster for battery degradation."""

    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.disable_trend = getattr(args, 'disable_trend', False)
        self.disable_residual = getattr(args, 'disable_residual', False)
        hidden_dim = args.hidden_dim
        feature_dim = 6 if args.dataset == 'TPSL' else 12

        self.curve_encoder = nn.Sequential(
            nn.Linear(args.seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.modulation = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)

        self.condition_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, args.pred_len, hidden_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.trend_head = nn.Linear(hidden_dim, 3)
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.residual_scale = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        forecast_position = torch.linspace(0.0, 1.0, args.pred_len)
        self.register_buffer('forecast_position', forecast_position.view(1, -1))

    def forward(
        self,
        capacity_increment,
        features,
        charge_current,
        discharge_current,
        temperature,
    ):
        curve_state = self.curve_encoder(capacity_increment)
        feature_state = self.feature_encoder(features)
        scale, shift = self.modulation(feature_state).chunk(2, dim=-1)
        global_state = self.global_norm(
            curve_state * (1.0 + torch.tanh(scale)) + shift
        )

        trend_parameters = self.trend_head(global_state)
        position = self.forecast_position
        level = trend_parameters[:, 0:1]
        slope = trend_parameters[:, 1:2]
        curvature = trend_parameters[:, 2:3]
        trend = level + slope * position + curvature * position * (1.0 - position)

        future_conditions = torch.stack(
            (charge_current, discharge_current, temperature), dim=-1
        )
        temporal_state = (
            self.condition_encoder(future_conditions)
            + self.position_embedding
            + global_state.unsqueeze(1)
        )
        temporal_state = self.temporal_mixer(
            temporal_state.transpose(1, 2)
        ).transpose(1, 2)
        residual = self.residual_head(temporal_state).squeeze(-1)
        residual = residual - residual.mean(dim=1, keepdim=True)

        trend_component = torch.zeros_like(trend) if self.disable_trend else trend
        residual_component = (
            torch.zeros_like(residual)
            if self.disable_residual
            else self.residual_scale(global_state) * residual
        )
        prediction = trend_component + residual_component
        return prediction, trend
