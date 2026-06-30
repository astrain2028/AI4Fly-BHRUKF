"""
Sensor models with fault injection.

z (15,): [gps(6), imu(6), pitot(1), 9hole(2)]
h (4,):  [h_gps, h_imu, h_pitot, h_9hole]  in [0, 1]
"""

import numpy as np
from dynamics import R_body_to_inertial

N_MEAS = 15

# Nominal noise std devs
SIGMA = {
    'gps_pos': 2.0,
    'gps_vel': 0.1,
    'imu_acc': 0.05,
    'imu_gyr': 0.005,
    'pitot':   0.3,
    'nineh':   0.02,
}

# How much noise inflates when health -> 0
FAULT_NOISE_SCALE = {
    'gps':   20.0,
    'imu':   15.0,
    'pitot':  8.0,
    'nineh': 10.0,
}


def measure(X, h, fault_cfg=None, rng=None):
    """
    Returns noisy z (15,) given state X and health h.
    fault_cfg keys: 'gps_bias', 'imu_bias', 'pitot_stuck', 'nineh_bias'
    """
    if rng is None:
        rng = np.random.default_rng()
    if fault_cfg is None:
        fault_cfg = {}

    phi, theta, psi = X[3], X[4], X[5]
    u_b, v_b, w_b   = X[6], X[7], X[8]
    p, q, r         = X[9], X[10], X[11]
    h_gps, h_imu, h_pit, h_9h = h

    vel_i = R_body_to_inertial(phi, theta, psi) @ np.array([u_b, v_b, w_b])

    # GPS
    s_p = SIGMA['gps_pos'] * (1 + (1 - h_gps) * FAULT_NOISE_SCALE['gps'])
    s_v = SIGMA['gps_vel'] * (1 + (1 - h_gps) * FAULT_NOISE_SCALE['gps'])
    z_gps = np.array([X[0], X[1], X[2], vel_i[0], vel_i[1], vel_i[2]])
    z_gps += rng.standard_normal(6) * np.array([s_p]*3 + [s_v]*3)
    if 'gps_bias' in fault_cfg:
        z_gps += np.asarray(fault_cfg['gps_bias'])

    # IMU  (accel truth ≈ 0 near hover)
    s_a = SIGMA['imu_acc'] * (1 + (1 - h_imu) * FAULT_NOISE_SCALE['imu'])
    s_g = SIGMA['imu_gyr'] * (1 + (1 - h_imu) * FAULT_NOISE_SCALE['imu'])
    z_imu = np.array([0.0, 0.0, 0.0, p, q, r])
    z_imu[:3] += rng.standard_normal(3) * s_a
    z_imu[3:]  += rng.standard_normal(3) * s_g
    if 'imu_bias' in fault_cfg:
        z_imu += np.asarray(fault_cfg['imu_bias'])

    # Pitot
    airspeed = np.sqrt(u_b**2 + v_b**2 + w_b**2)
    s_pit = SIGMA['pitot'] * (1 + (1 - h_pit) * FAULT_NOISE_SCALE['pitot'])
    if 'pitot_stuck' in fault_cfg:
        z_pitot = np.array([float(fault_cfg['pitot_stuck'])])
    else:
        z_pitot = np.array([airspeed]) + rng.standard_normal(1) * s_pit

    # 9-hole
    alpha  = np.arctan2(w_b, np.sqrt(u_b**2 + v_b**2))
    beta   = np.arctan2(v_b, u_b)
    s_9h   = SIGMA['nineh'] * (1 + (1 - h_9h) * FAULT_NOISE_SCALE['nineh'])
    z_9hole = np.array([alpha, beta]) + rng.standard_normal(2) * s_9h
    if 'nineh_bias' in fault_cfg:
        z_9hole += np.asarray(fault_cfg['nineh_bias'])

    return np.concatenate([z_gps, z_imu, z_pitot, z_9hole])


def true_measurement(X):
    """Noiseless measurement."""
    phi, theta, psi = X[3], X[4], X[5]
    u_b, v_b, w_b   = X[6], X[7], X[8]
    p, q, r         = X[9], X[10], X[11]

    vel_i    = R_body_to_inertial(phi, theta, psi) @ np.array([u_b, v_b, w_b])
    airspeed = np.sqrt(u_b**2 + v_b**2 + w_b**2)
    alpha    = np.arctan2(w_b, np.sqrt(u_b**2 + v_b**2))
    beta     = np.arctan2(v_b, u_b)

    return np.concatenate([
        [X[0], X[1], X[2], vel_i[0], vel_i[1], vel_i[2]],
        [0.0, 0.0, 0.0, p, q, r],
        [airspeed],
        [alpha, beta],
    ])


IDX = {
    'gps':   slice(0,  6),
    'imu':   slice(6,  12),
    'pitot': slice(12, 13),
    'nineh': slice(13, 15),
}
