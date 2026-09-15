import torch
import torch.nn as nn


class CurveEncoder(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            for kernel_size in (3, 5, 7)
        ])

    @property
    def output_dim(self):
        return sum(branch[0].out_channels for branch in self.branches)

    def forward(self, curve):
        curve = curve.unsqueeze(1)
        return torch.cat([branch(curve).squeeze(-1) for branch in self.branches], dim=1)


class SharedLocalExpert(nn.Module):
    def __init__(self, seq_len, pred_len, channels=4):
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, curve):
        local_features = self.local_encoder(curve.unsqueeze(1)).squeeze(1)
        return self.projection(local_features)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError('top_k must be between 1 and num_experts')

        self.capacity_fc = nn.ModuleList([
            nn.Linear(args.seq_len, args.pred_len)
            for _ in range(self.num_experts)
        ])

        feature_dim = 6 if args.dataset == 'TPSL' else 12
        self.curve_encoder = CurveEncoder(channels=4)
        router_input_dim = feature_dim + self.curve_encoder.output_dim
        self.router_fc = nn.Linear(router_input_dim, self.num_experts)
        self.shared_expert = SharedLocalExpert(args.seq_len, args.pred_len, channels=4)
        self.shared_gate = nn.Sequential(
            nn.Linear(router_input_dim, 1),
            nn.Sigmoid(),
        )

        self.w_noise = nn.Parameter(torch.zeros(self.num_experts, self.num_experts))
        self.softplus = nn.Softplus()
        self.noise_epsilon = 1e-5

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=args.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_final = nn.Linear(2 * args.hidden_dim, 1)

    def sparse_top_k(self, logits):
        top_values, top_indices = torch.topk(logits, self.top_k, dim=1)
        top_weights = torch.softmax(top_values, dim=1)
        return torch.zeros_like(logits).scatter(1, top_indices, top_weights)

    def forward(self, capacity_increment, features, charge_current, discharge_current, Temperature):
        routed_outs = [fc(capacity_increment) for fc in self.capacity_fc]

        curve_features = self.curve_encoder(capacity_increment)
        router_input = torch.cat([features, curve_features], dim=1)
        clean_logits = self.router_fc(router_input)
        if self.training:
            raw_noise_stddev = clean_logits @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        weights = self.sparse_top_k(logits)

        routed_trend = torch.zeros_like(routed_outs[0])
        for i in range(self.num_experts):
            routed_trend += weights[:, i].unsqueeze(1) * routed_outs[i]

        shared_trend = self.shared_expert(capacity_increment)
        shared_weight = self.shared_gate(router_input)
        degradation_trend = shared_weight * shared_trend + (1.0 - shared_weight) * routed_trend

        lstm_input = torch.cat([
            degradation_trend.unsqueeze(2),
            charge_current.unsqueeze(2),
            discharge_current.unsqueeze(2),
            Temperature.unsqueeze(2),
        ], dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        final_output = self.fc_final(lstm_out)
        return final_output.squeeze(-1), weights
