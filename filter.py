"""
UKF for UAS state estimation with a BHR surrogate measurement model.

Composite state x = [x_vehicle (12), x_health (4)] = 16-dim.
Wraps ukf_core.UnscentedKalmanFilter and injects a state-dependent R(x)
from the BHR model before each measurement update.
"""

import numpy as np
import torch
from dynamics import N_VEHICLE, rk4_step, hover_trim
from sensors import N_MEAS
from ukf_core import UnscentedKalmanFilter

N_HEALTH = 4
N_STATE  = N_VEHICLE + N_HEALTH

_, _U_TRIM = hover_trim()


def _vehicle_health_transition(x, u=None, dt=0.01):
    if u is None:
        u = _U_TRIM
    x_veh = rk4_step(x[:N_VEHICLE], u, dt)
    x_h   = np.clip(x[N_VEHICLE:], 0.0, 1.0)
    return np.concatenate([x_veh, x_h])


def _nn_predict_mean(model, x_composite, device):
    """Deterministic (dropout-off) forward pass, used inside the unscented transform."""
    x_t = torch.from_numpy(x_composite.astype(np.float32)).unsqueeze(0).to(device)
    mu_z = model.predict_mean(x_t)
    return mu_z.squeeze(0).cpu().numpy()


def _nn_predict_uncertainty(model, x_composite, device, n_samples=20):
    """MC-dropout pass, used only to set R at the prior mean (one draw per update)."""
    x_t = torch.from_numpy(x_composite.astype(np.float32)).unsqueeze(0).to(device)
    _, var_z = model.predict_with_uncertainty(x_t, n_samples=n_samples)
    return var_z.squeeze(0).cpu().numpy()


class BHRUKF(UnscentedKalmanFilter):
    """UKF whose measurement function and R both come from the BHR model."""

    def __init__(self, model, device, dt=0.01,
                 sigma_health=0.005, n_mc=20):
        self.model  = model
        self.device = device
        self.n_mc   = n_mc

        Q = np.zeros((N_STATE, N_STATE))
        from dynamics import process_noise_cov
        Q[:N_VEHICLE, :N_VEHICLE] = process_noise_cov(dt)
        Q[N_VEHICLE:, N_VEHICLE:] = np.eye(N_HEALTH) * sigma_health**2

        f = lambda x, u=None: _vehicle_health_transition(x, u, dt)
        h = lambda x: _nn_predict_mean(model, x, device)

        # alpha=1e-3 (the UKF default) makes n+lambda ~1e-5 for this 16-dim state,
        # producing a huge central sigma-point weight. With a linear f this cancels
        # exactly; with the true nonlinear dynamics it amplifies float round-off
        # into NaNs. alpha=1 keeps n+lambda = n+kappa, a well-scaled spread.
        super().__init__(f=f, h=h, Q=Q, R=np.eye(N_MEAS),
                         dim_x=N_STATE, dim_z=N_MEAS, alpha=1.0, kappa=0.0)

    def initialize(self, x_vehicle_init, P_vehicle_init,
                   h_init=None, P_health_init=None):
        if h_init is None:
            h_init = np.ones(N_HEALTH)
        if P_health_init is None:
            P_health_init = np.eye(N_HEALTH) * 0.1

        x0 = np.concatenate([x_vehicle_init, h_init])
        P0 = np.zeros((N_STATE, N_STATE))
        P0[:N_VEHICLE, :N_VEHICLE] = P_vehicle_init
        P0[N_VEHICLE:, N_VEHICLE:] = P_health_init
        super().initialize(x0, P0)

    def update(self, z):
        var_z = _nn_predict_uncertainty(self.model, self.x, self.device, n_samples=self.n_mc)
        self.R = np.diag(np.clip(var_z, 1e-6, None))
        super().update(z)
        self.x[N_VEHICLE:] = np.clip(self.x[N_VEHICLE:], 0.0, 1.0)
        return self.x.copy(), self.P.copy()

    def step(self, z, u=None):
        self.predict(u)
        return self.update(z)
