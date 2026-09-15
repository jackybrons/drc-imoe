import torch
import torch.nn as nn

from .iMOE_CSR import CurveEncoder


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_experts = args.num_experts
        self.baseline_top_k = getattr(args, 'baseline_top_k', self.num_experts)
        self.csr_top_k = getattr(args, 'csr_top_k', self.num_experts)
        self.use_top_k = not getattr(args, 'disable_top_k', False)
        self.use_noisy_routing = not getattr(args, 'disable_noisy_routing', False)
        self.fusion_mode = getattr(args, 'fusion_mode', 'learned')
        self.fusion_gate_bias = getattr(args, 'fusion_gate_bias', 0.0)
        if self.fusion_mode not in ('learned', 'fixed'):
            raise ValueError("fusion_mode must be either 'learned' or 'fixed'")
        if self.use_top_k:
            if not 1 <= self.baseline_top_k <= self.num_experts:
                raise ValueError('baseline_top_k must be between 1 and num_experts')
            if not 1 <= self.csr_top_k <= self.num_experts:
                raise ValueError('csr_top_k must be between 1 and num_experts')

        self.router_alpha = args.alpha
        self.capacity_fc = nn.ModuleList([
            nn.Linear(args.seq_len, args.pred_len)
            for _ in range(self.num_experts)
        ])

        feature_dim = 6 if args.dataset == 'TPSL' else 12
        self.baseline_router = nn.Linear(feature_dim, self.num_experts)
        self.curve_encoder = CurveEncoder(channels=args.curve_channels)
        self.csr_router = nn.Linear(
            feature_dim + self.curve_encoder.output_dim,
            self.num_experts,
        )
        if self.fusion_mode == 'learned':
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * self.num_experts, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.fusion_gate[0].weight)
            nn.init.constant_(
                self.fusion_gate[0].bias,
                self.fusion_gate_bias,
            )
        else:
            self.fusion_gate = None

        if self.use_noisy_routing:
            self.baseline_w_noise = nn.Parameter(
                torch.zeros(self.num_experts, self.num_experts)
            )
            self.csr_w_noise = nn.Parameter(
                torch.zeros(self.num_experts, self.num_experts)
            )
            self.softplus = nn.Softplus()
            self.noise_epsilon = 1e-5
        else:
            self.register_parameter('baseline_w_noise', None)
            self.register_parameter('csr_w_noise', None)

        self._routing_diagnostics = None

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=args.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_final = nn.Linear(2 * args.hidden_dim, 1)

    def _training_logits(self, clean_logits, noise_weights):
        if not self.training or not self.use_noisy_routing:
            return clean_logits
        raw_noise_stddev = clean_logits @ noise_weights
        noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
        return clean_logits + torch.randn_like(clean_logits) * noise_stddev

    def _baseline_weights(self, logits):
        if not self.use_top_k:
            return torch.softmax(logits, dim=1)
        kth_largest, _ = torch.kthvalue(
            logits,
            self.num_experts - self.baseline_top_k + 1,
        )
        mask = logits < kth_largest.unsqueeze(1)
        probabilities = torch.softmax(logits, dim=1)
        transformed = torch.zeros_like(logits)
        transformed[mask] = self.router_alpha * torch.log(probabilities[mask] + 1)
        transformed[~mask] = self.router_alpha * (torch.exp(probabilities[~mask]) - 1)
        return torch.softmax(transformed, dim=1)

    def _csr_weights(self, logits):
        if not self.use_top_k:
            return torch.softmax(logits, dim=1)
        top_values, top_indices = torch.topk(logits, self.csr_top_k, dim=1)
        top_weights = torch.softmax(top_values, dim=1)
        return torch.zeros_like(logits).scatter(1, top_indices, top_weights)

    def forward(self, capacity_increment, features, charge_current, discharge_current, Temperature):
        expert_outputs = [fc(capacity_increment) for fc in self.capacity_fc]

        baseline_clean_logits = self.baseline_router(features)
        curve_features = self.curve_encoder(capacity_increment)
        csr_input = torch.cat([features, curve_features], dim=1)
        csr_clean_logits = self.csr_router(csr_input)

        baseline_logits = self._training_logits(
            baseline_clean_logits,
            self.baseline_w_noise,
        )
        csr_logits = self._training_logits(csr_clean_logits, self.csr_w_noise)
        baseline_weights = self._baseline_weights(baseline_logits)
        csr_weights = self._csr_weights(csr_logits)

        if self.fusion_mode == 'learned':
            fusion_weight = self.fusion_gate(torch.cat([
                baseline_clean_logits,
                csr_clean_logits,
            ], dim=1))
        else:
            fusion_weight = baseline_weights.new_full(
                (baseline_weights.shape[0], 1),
                0.5,
            )
        fused_weights = (
            (1.0 - fusion_weight) * baseline_weights
            + fusion_weight * csr_weights
        )

        self._routing_diagnostics = {
            'fusion_gate': fusion_weight.detach().flatten(),
            'router_weight_l1': (
                baseline_weights - csr_weights
            ).abs().sum(dim=1).detach(),
            'statistical_route_entropy': self._route_entropy(
                baseline_weights
            ).detach(),
            'curve_route_entropy': self._route_entropy(csr_weights).detach(),
            'fused_route_entropy': self._route_entropy(fused_weights).detach(),
        }

        degradation_trend = torch.zeros_like(expert_outputs[0])
        for index, expert_output in enumerate(expert_outputs):
            degradation_trend += fused_weights[:, index].unsqueeze(1) * expert_output

        lstm_input = torch.cat([
            degradation_trend.unsqueeze(2),
            charge_current.unsqueeze(2),
            discharge_current.unsqueeze(2),
            Temperature.unsqueeze(2),
        ], dim=2)
        lstm_output, _ = self.lstm(lstm_input)
        prediction = self.fc_final(lstm_output)
        return prediction.squeeze(-1), fused_weights

    @staticmethod
    def _route_entropy(weights):
        positive_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
        return -(weights * positive_weights.log()).sum(dim=1)

    def get_routing_diagnostics(self):
        """Return detached per-sample diagnostics from the latest forward pass."""
        return self._routing_diagnostics
