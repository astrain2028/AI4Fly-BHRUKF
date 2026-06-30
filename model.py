"""
Heteroscedastic surrogate measurement model.

Input  (16,): [vehicle state (12), sensor health (4)]
Output (30,): [mu (15) | log_var (15)]
"""

import torch
import torch.nn as nn


class SensorHead(nn.Module):
    def __init__(self, in_features, out_dim, dropout_p=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features // 2, out_dim * 2),
        )
        nn.init.zeros_(self.net[-1].bias[out_dim:])  # init log_var near 0

    def forward(self, x):
        out = self.net(x)
        return out[:, :self.out_dim], out[:, self.out_dim:]


class HetRegModel(nn.Module):
    SENSOR_DIMS = {'gps': 6, 'imu': 6, 'pitot': 1, 'nineh': 2}

    def __init__(self, in_dim=16, trunk_dims=(128, 128, 64), dropout_p=0.1):
        super().__init__()

        # Raw inputs span wildly different scales (position up to ~150 m,
        # angles in radians, health flags in [0, 1]); normalize so the trunk
        # sees zero-mean/unit-variance features. Defaults to a no-op until
        # set_input_stats() is called with statistics fit on the training set.
        self.register_buffer('x_mean', torch.zeros(in_dim))
        self.register_buffer('x_std',  torch.ones(in_dim))

        layers, prev = [], in_dim
        for h in trunk_dims:
            layers += [nn.Linear(prev, h), nn.SiLU(), nn.Dropout(p=dropout_p)]
            prev = h
        self.trunk = nn.Sequential(*layers)

        self.head_gps   = SensorHead(prev, self.SENSOR_DIMS['gps'],   dropout_p)
        self.head_imu   = SensorHead(prev, self.SENSOR_DIMS['imu'],   dropout_p)
        self.head_pitot = SensorHead(prev, self.SENSOR_DIMS['pitot'], dropout_p)
        self.head_nineh = SensorHead(prev, self.SENSOR_DIMS['nineh'], dropout_p)

    def set_input_stats(self, mean, std):
        """Fit input normalization to training-set statistics (call before training)."""
        self.x_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-6))

    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        z = self.trunk(x)
        mu_gps,   lv_gps   = self.head_gps(z)
        mu_imu,   lv_imu   = self.head_imu(z)
        mu_pitot, lv_pitot = self.head_pitot(z)
        mu_nineh, lv_nineh = self.head_nineh(z)

        mu      = torch.cat([mu_gps, mu_imu, mu_pitot, mu_nineh], dim=1)
        log_var = torch.cat([lv_gps, lv_imu, lv_pitot, lv_nineh], dim=1)
        log_var = log_var.clamp(min=-10.0, max=10.0)  # bound predicted variance at inference too
        return torch.cat([mu, log_var], dim=1)

    def predict_mean(self, x):
        """Deterministic forward pass (dropout off): returns mu (batch, 15)."""
        self.eval()
        with torch.no_grad():
            out = self(x)
        return out[:, :15]

    def predict_with_uncertainty(self, x, n_samples=30):
        """MC-dropout: returns (mu, total_var) each (batch, 15)."""
        self.eval()
        _enable_dropout(self)
        with torch.no_grad():
            samples = torch.stack([self(x) for _ in range(n_samples)], dim=1)
        mu_s      = samples[:, :, :15]
        log_var_s = samples[:, :, 15:]
        epistemic = mu_s.var(dim=1)
        aleatoric = log_var_s.exp().mean(dim=1)
        return mu_s.mean(dim=1), epistemic + aleatoric


def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def gaussian_nll_loss(f, y, beta=0.5):
    """Beta-weighted heteroscedastic Gaussian NLL."""
    D       = y.shape[1]
    mu      = f[:, :D]
    log_var = f[:, D:].clamp(min=-10.0, max=10.0)  # prevent var collapse -> exploding NLL
    var     = log_var.exp()
    return (var.detach().pow(beta) * (mu - y).pow(2) / var + log_var).mean()
