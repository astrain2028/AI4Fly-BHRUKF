"""
Generate synthetic training data and save to a compressed .npz file.

Usage:
    python generate_data.py --episodes 200 --T 20 --out data/synthetic.npz
"""

import argparse
import os
import numpy as np
from dynamics import rk4_step, hover_trim, m, g, Ixx, Iyy, Izz
from sensors import measure

INPUT_COLS = [
    'x', 'y', 'z', 'phi', 'theta', 'psi',
    'u', 'v', 'w', 'p', 'q', 'r',
    'h_gps', 'h_imu', 'h_pitot', 'h_9hole',
]
OUTPUT_COLS = [
    'z_gps_x', 'z_gps_y', 'z_gps_z', 'z_gps_vx', 'z_gps_vy', 'z_gps_vz',
    'z_imu_ax', 'z_imu_ay', 'z_imu_az', 'z_imu_p', 'z_imu_q', 'z_imu_r',
    'z_pitot',
    'z_9h_alpha', 'z_9h_beta',
]
ALL_COLS = INPUT_COLS + OUTPUT_COLS


def _sample_health(rng, fault_prob=0.3):
    h = np.ones(4)
    for i in range(4):
        if rng.random() < fault_prob:
            h[i] = rng.beta(2, 5)
    return h


def _build_fault_cfg(h, rng):
    cfg = {}
    h_gps, h_imu, h_pit, h_9h = h

    if h_gps < 0.5:
        mag = (1 - h_gps) * 80.0
        cfg['gps_bias'] = rng.standard_normal(6) * mag * np.array([1,1,1,0.1,0.1,0.1])

    if h_imu < 0.5:
        bias = np.zeros(6)
        bias[:3] = rng.standard_normal(3) * (1 - h_imu) * 2.0
        cfg['imu_bias'] = bias

    if h_pit < 0.5:
        cfg['pitot_stuck'] = rng.uniform(0.0, 2.0)

    if h_9h < 0.5:
        cfg['nineh_bias'] = rng.standard_normal(2) * (1 - h_9h) * 0.5

    return cfg


# PD gains for a simple attitude/altitude stabilizer (keeps the sim near hover
# instead of letting random torques tumble the vehicle)
KP_ATT, KD_ATT = 6.0, 2.0
KP_ALT, KD_ALT = 4.0, 3.0


def _stabilizing_control(X, z_setpoint, rng, dt):
    """PD control toward level attitude + a target altitude, with small
    random setpoint jitter so trajectories still vary between episodes."""
    phi, theta, psi = X[3], X[4], X[5]
    z, w            = X[2], X[8]
    p, q, r         = X[9], X[10], X[11]

    phi_sp   = rng.normal(0, 0.05)
    theta_sp = rng.normal(0, 0.05)

    L = Ixx * (KP_ATT * (phi_sp - phi)   - KD_ATT * p)
    M = Iyy * (KP_ATT * (theta_sp - theta) - KD_ATT * q)
    N = Izz * (-KD_ATT * r)

    T_cmd = m * g + m * (KP_ALT * (z - z_setpoint) - KD_ALT * w)
    T_cmd = np.clip(T_cmd, 0.1, 2 * m * g)

    return np.array([T_cmd, L, M, N])


def generate_episode(T=20.0, dt=0.01, fault_prob=0.3, seed=0):
    rng = np.random.default_rng(seed)

    X0, _ = hover_trim()
    X0[0] = rng.uniform(-50, 50)
    X0[1] = rng.uniform(-50, 50)
    X0[2] = rng.uniform(-150, -30)
    X0[6] = rng.uniform(-3, 3)
    X0[7] = rng.uniform(-1, 1)
    z_setpoint = X0[2]

    N_steps = int(T / dt)
    X_hist = np.zeros((N_steps, 12))
    X_k = X0.copy()
    for k in range(N_steps):
        X_hist[k] = X_k
        U_k = _stabilizing_control(X_k, z_setpoint, rng, dt)
        X_k = rk4_step(X_k, U_k, dt)

    change_interval = max(1, int(2.0 / dt))
    h         = _sample_health(rng, fault_prob)
    fault_cfg = _build_fault_cfg(h, rng)

    rows = []
    for k, X_k in enumerate(X_hist):
        if k % change_interval == 0:
            h         = _sample_health(rng, fault_prob)
            fault_cfg = _build_fault_cfg(h, rng)
        z = measure(X_k, h, fault_cfg, rng=rng)
        rows.append(np.concatenate([X_k, h, z]))

    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',   type=int,   default=200)
    parser.add_argument('--T',          type=float, default=20.0)
    parser.add_argument('--dt',         type=float, default=0.01)
    parser.add_argument('--fault_prob', type=float, default=0.3)
    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--out',        type=str,   default='data/synthetic.npz')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_rows = []
    for ep in range(args.episodes):
        rows = generate_episode(T=args.T, dt=args.dt,
                                fault_prob=args.fault_prob,
                                seed=args.seed + ep)
        all_rows.append(rows)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{args.episodes} | rows: {sum(len(r) for r in all_rows):,}")

    data = np.concatenate(all_rows, axis=0).astype(np.float32)
    X = data[:, :len(INPUT_COLS)]
    y = data[:, len(INPUT_COLS):]
    np.savez(args.out, X=X, y=y,
              input_cols=np.array(INPUT_COLS), output_cols=np.array(OUTPUT_COLS))
    print(f"\nSaved {len(data):,} rows -> {args.out}")


if __name__ == '__main__':
    main()
