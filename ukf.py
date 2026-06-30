"""
Unscented Kalman Filter (UKF) Implementation
=============================================
A general-purpose Unscented Kalman Filter for nonlinear systems with Gaussian noise.

System Model (nonlinear):
    x_k = f(x_{k-1}, u_k) + w_k,   w_k ~ N(0, Q)
    z_k = h(x_k) + v_k,             v_k ~ N(0, R)

Instead of linearizing with Jacobians (like the EKF), the UKF uses a deterministic
sampling technique known as the *Unscented Transform*. A set of carefully chosen
"sigma points" are propagated through the true nonlinear functions, and the
statistics (mean and covariance) are recovered from these transformed points.

Usage:
    from ukf_core import UnscentedKalmanFilter

    ukf = UnscentedKalmanFilter(f=f_func, h=h_func, Q=Q, R=R, dim_x=4, dim_z=2)
    ukf.initialize(x0, P0)
    for z in measurements:
        ukf.predict()
        ukf.update(z)
"""

import numpy as np
from numpy.linalg import inv


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear systems.

    Parameters
    ----------
    f : callable
        State transition function. Signature: f(x, u=None) -> x_predicted
    h : callable
        Measurement function. Signature: h(x) -> z_predicted
    Q : np.ndarray (n, n)
        Process noise covariance.
    R : np.ndarray (m, m)
        Measurement noise covariance.
    dim_x : int
        Dimension of the state vector.
    dim_z : int
        Dimension of the measurement vector.
    alpha : float, optional
        Spread of sigma points around the mean (default: 1e-3).
        NOTE: for high-dimensional states with a *nonlinear* f, a very small
        alpha makes n+lambda tiny, producing a huge central sigma-point
        weight that amplifies floating-point round-off into NaNs. Use
        alpha close to 1 unless f is linear (see filter.py).
    beta : float, optional
        Incorporates prior knowledge of the distribution (default: 2.0).
        beta = 2 is optimal for Gaussian distributions.
    kappa : float, optional
        Secondary scaling parameter (default: 0).
        Common choice: 3 - n.
    """

    def __init__(self, f, h, Q, R, dim_x, dim_z, alpha=1e-3, beta=2.0, kappa=0.0):
        self.f = f
        self.h = h
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Unscented transform parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        n = dim_x
        self.lam = alpha**2 * (n + kappa) - n  # lambda

        # State and covariance
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

        # Diagnostics
        self.innovation = None
        self.innovation_cov = None
        self.K = None

    def initialize(self, x0, P0):
        """
        Set the initial state estimate and covariance.

        Parameters
        ----------
        x0 : np.ndarray (n,)
            Initial state estimate.
        P0 : np.ndarray (n, n)
            Initial state covariance.
        """
        self.x = np.array(x0, dtype=float).flatten()
        self.P = np.array(P0, dtype=float)

    # ------------------------------------------------------------------ #
    #  Sigma Point Generation & Weights
    # ------------------------------------------------------------------ #
    def _compute_sigma_points(self, x, cov):
        """
        Generate 2n+1 sigma points and their associated weights.

        Parameters
        ----------
        x : np.ndarray (n,)
            Mean of the distribution.
        cov : np.ndarray (n, n)
            Covariance of the distribution.

        Returns
        -------
        sigma_pts : np.ndarray (2n+1, n)
            Sigma points (each row is a point).
        weights_mean : np.ndarray (2n+1,)
            Weights for computing the mean.
        weights_cov : np.ndarray (2n+1,)
            Weights for computing the covariance.
        """
        n = self.dim_x
        lam = self.lam
        num_sigma = 2 * n + 1

        # Ensure covariance is symmetric and positive-definite
        cov = (cov + cov.T) / 2.0
        cov = cov + 1e-8 * np.eye(n)

        # Compute matrix square root using eigendecomposition
        # sqrt((n + lambda) * P) via eigendecomposition is more robust than Cholesky
        eigvals, eigvecs = np.linalg.eigh((n + lam) * cov)
        eigvals = np.maximum(eigvals, 1e-6)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        # Generate sigma points
        sigma_pts = np.zeros((num_sigma, n))
        sigma_pts[0] = x
        for i in range(n):
            sigma_pts[i + 1] = x + sqrt_cov[:, i]
            sigma_pts[n + i + 1] = x - sqrt_cov[:, i]

        # Compute weights
        weights_mean = np.zeros(num_sigma)
        weights_cov = np.zeros(num_sigma)
        weights_mean[0] = lam / (n + lam)
        weights_cov[0] = lam / (n + lam) + (1 - self.alpha**2 + self.beta)
        for i in range(1, num_sigma):
            weights_mean[i] = 1.0 / (2.0 * (n + lam))
            weights_cov[i] = 1.0 / (2.0 * (n + lam))

        return sigma_pts, weights_mean, weights_cov

    # ------------------------------------------------------------------ #
    #  Predict & Update
    # ------------------------------------------------------------------ #
    def predict(self, u=None):
        """
        Prediction (time-update) step via the Unscented Transform.

        1. Generate sigma points from (x, P)
        2. Propagate each sigma point through f
        3. Recover predicted mean and covariance

        Parameters
        ----------
        u : optional
            Control input (passed to f).
        """
        n = self.dim_x

        # 1. Generate sigma points
        sigma_pts, Wm, Wc = self._compute_sigma_points(self.x, self.P)

        # 2. Propagate sigma points through dynamics
        sigma_pts_pred = np.zeros_like(sigma_pts)
        for i in range(2 * n + 1):
            sigma_pts_pred[i] = self.f(sigma_pts[i], u)

        # 3. Recover predicted mean
        x_pred = np.zeros(n)
        for i in range(2 * n + 1):
            x_pred += Wm[i] * sigma_pts_pred[i]

        # 4. Recover predicted covariance
        P_pred = np.zeros((n, n))
        for i in range(2 * n + 1):
            diff = sigma_pts_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)
        P_pred += self.Q

        self.x = x_pred
        self.P = (P_pred + P_pred.T) / 2.0  # enforce symmetry

    def update(self, z):
        """
        Measurement-update (correction) step via the Unscented Transform.

        1. Generate sigma points from predicted (x^-, P^-)
        2. Transform sigma points through h to get predicted measurements
        3. Compute innovation covariance S and cross-covariance Pxz
        4. Compute Kalman gain and update state + covariance

        Parameters
        ----------
        z : np.ndarray (m,)
            Measurement vector.
        """
        z = np.array(z, dtype=float).flatten()
        n = self.dim_x
        m = self.dim_z

        # 1. Generate sigma points from predicted state
        sigma_pts, Wm, Wc = self._compute_sigma_points(self.x, self.P)

        # 2. Transform through measurement function
        gamma = np.zeros((2 * n + 1, m))
        for i in range(2 * n + 1):
            gamma[i] = self.h(sigma_pts[i])

        # Predicted measurement mean
        z_pred = np.zeros(m)
        for i in range(2 * n + 1):
            z_pred += Wm[i] * gamma[i]

        # 3. Innovation covariance S = Pyy + R
        Pyy = np.zeros((m, m))
        for i in range(2 * n + 1):
            diff_z = gamma[i] - z_pred
            Pyy += Wc[i] * np.outer(diff_z, diff_z)
        S = Pyy + self.R

        # Cross-covariance Pxz
        Pxz = np.zeros((n, m))
        for i in range(2 * n + 1):
            diff_x = sigma_pts[i] - self.x
            diff_z = gamma[i] - z_pred
            Pxz += Wc[i] * np.outer(diff_x, diff_z)

        # 4. Kalman gain
        K = Pxz @ inv(S)

        # Innovation
        y = z - z_pred

        # State and covariance update
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.P = (self.P + self.P.T) / 2.0  # enforce symmetry

        # Store diagnostics
        self.innovation = y
        self.innovation_cov = S
        self.K = K

    def run(self, measurements, x0, P0, u_list=None):
        """
        Convenience method: run the full filter over a sequence of measurements.

        Parameters
        ----------
        measurements : np.ndarray (T, m)
            Array of measurements, one per time step.
        x0 : np.ndarray (n,)
            Initial state estimate.
        P0 : np.ndarray (n, n)
            Initial covariance.
        u_list : list, optional
            Control inputs for each time step.

        Returns
        -------
        x_est : np.ndarray (T, n)
            State estimates at each time step.
        P_est : np.ndarray (T, n, n)
            Covariance estimates at each time step.
        innovations : np.ndarray (T, m)
            Innovations at each time step.
        """
        T = len(measurements)
        x_est = np.zeros((T, self.dim_x))
        P_est = np.zeros((T, self.dim_x, self.dim_x))
        innovations = np.zeros((T, self.dim_z))

        self.initialize(x0, P0)

        for k in range(T):
            u = u_list[k] if u_list is not None else None
            self.predict(u)
            self.update(measurements[k])

            x_est[k] = self.x.copy()
            P_est[k] = self.P.copy()
            innovations[k] = self.innovation.copy()

        return x_est, P_est, innovations
