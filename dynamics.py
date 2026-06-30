"""
6DOF quadrotor dynamics.

State X (12,): [x, y, z, phi, theta, psi, u, v, w, p, q, r]
Control U (4,): [T, L, M, N]
"""

import numpy as np

m   = 1.0
g   = 9.81
Ixx = 8.1e-3
Iyy = 8.1e-3
Izz = 14.2e-3

N_VEHICLE  = 12
DT_DEFAULT = 0.01


def R_body_to_inertial(phi, theta, psi):
    c, s = np.cos, np.sin
    return np.array([
        [c(psi)*c(theta),  c(psi)*s(theta)*s(phi) - s(psi)*c(phi),  c(psi)*s(theta)*c(phi) + s(psi)*s(phi)],
        [s(psi)*c(theta),  s(psi)*s(theta)*s(phi) + c(psi)*c(phi),  s(psi)*s(theta)*c(phi) - c(psi)*s(phi)],
        [-s(theta),        c(theta)*s(phi),                          c(theta)*c(phi)],
    ])


def R_inertial_to_body(phi, theta, psi):
    return R_body_to_inertial(phi, theta, psi).T


def euler_kinematics(phi, theta):
    c, s, t = np.cos, np.sin, np.tan
    return np.array([
        [1,  s(phi)*t(theta),   c(phi)*t(theta)],
        [0,  c(phi),           -s(phi)],
        [0,  s(phi)/c(theta),   c(phi)/c(theta)],
    ])


def quad_6dof(X, U):
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)

    phi, theta, psi = X[3], X[4], X[5]
    u_b, v_b, w_b   = X[6], X[7], X[8]
    p, q, r         = X[9], X[10], X[11]
    T, L, M, N      = U[0], U[1], U[2], U[3]

    pos_dot = R_body_to_inertial(phi, theta, psi) @ np.array([u_b, v_b, w_b])
    eul_dot = euler_kinematics(phi, theta) @ np.array([p, q, r])

    gravity_body = R_inertial_to_body(phi, theta, psi) @ np.array([0.0, 0.0, g])
    vel_dot = np.array([r*v_b - q*w_b,
                        p*w_b - r*u_b,
                        q*u_b - p*v_b]) + np.array([0.0, 0.0, -T/m]) + gravity_body

    att_dot = np.array([
        L/Ixx + ((Iyy - Izz)/Ixx)*q*r,
        M/Iyy + ((Izz - Ixx)/Iyy)*p*r,
        N/Izz + ((Ixx - Iyy)/Izz)*p*r,
    ])

    return np.concatenate([pos_dot, eul_dot, vel_dot, att_dot])


def rk4_step(X, U, dt):
    k1 = quad_6dof(X,           U)
    k2 = quad_6dof(X + dt/2*k1, U)
    k3 = quad_6dof(X + dt/2*k2, U)
    k4 = quad_6dof(X + dt*k3,   U)
    return X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def simulate(X0, U_seq, dt=0.01):
    N = len(U_seq)
    X_hist = np.zeros((N + 1, 12))
    t_hist = np.arange(N + 1) * dt
    X_hist[0] = X0
    for k in range(N):
        X_hist[k + 1] = rk4_step(X_hist[k], U_seq[k], dt)
    return X_hist, t_hist


def hover_trim():
    return np.zeros(12), np.array([m * g, 0.0, 0.0, 0.0])


def state_transition(x_vehicle, dt=DT_DEFAULT):
    """Linear constant-velocity propagation."""
    F = np.eye(N_VEHICLE)
    F[0, 6] = dt; F[1, 7] = dt; F[2, 8] = dt  # pos += vel * dt
    return F @ x_vehicle


def process_noise_cov(dt=DT_DEFAULT, sigma_accel=0.3):
    """Discrete process noise covariance Q (12x12) for constant-velocity model."""
    q = sigma_accel ** 2
    Q = np.zeros((N_VEHICLE, N_VEHICLE))
    Q_block = np.array([[dt**3/3, dt**2/2],
                         [dt**2/2, dt     ]]) * q
    for i in range(3):
        ix = np.ix_([i, i+6], [i, i+6])
        Q[ix] = Q_block
    return Q
