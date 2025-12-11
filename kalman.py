from typing import Tuple
import numpy as np


def run_kf_altitude(t: np.ndarray, z: np.ndarray, sigma_meas: float = 1.5, q_hdot: float = 0.2, q_vdot: float = 0.5, h0: float = 0.0, v0: float = 0.0, P0_diag=(50.0**2, 10.0**2)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a time-varying discrete Kalman filter for altitude and vertical speed

    :param t: Time stamps (s)
    :type t: numpy.ndarray
    :param z: Altimeter measurements (m)
    :type z: numpy.ndarray
    :param sigma_meas: Measurement noise standard deviation (m)
    :type sigma_meas: float
    :param q_hdot: Process noise PSD for altitude rate (m^2/s^3); scaled by dt in Q
    :type q_hdot: float
    :param q_vdot: Process noise PSD for vertical-speed rate (m^2/s^3); scaled by dt in Q
    :type q_vdot: float
    :param h0: Initial altitude estimate (m)
    :type h0: float
    :param v0: Initial vertical speed estimate (m/s, down-positive)
    :type v0: float
    :param P0_diag: Initial covariance diagonal (h, v)
    :type P0_diag: tuple[float, float]

    :returns: (xhat_hist, Pdiag_hist) where
              * xhat_hist is the filtered state estimates [h, v] over time with shape (N, 2),
              * Pdiag_hist is the time history of the covariance diagonal entries with shape (N, 2).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """

    N = len(t)
    xhat = np.array([h0, v0], dtype=float)
    P = np.diag(P0_diag).astype(float)

    R = np.array([[sigma_meas**2]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)

    xhat_hist = np.zeros((N, 2), dtype=float)
    Pdiag_hist = np.zeros((N, 2), dtype=float)

    for k in range(N):
        if k == 0:
            dt = max(1e-9, t[1] - t[0]) if N > 1 else 0.01
        else:
            dt = max(1e-9, t[k] - t[k - 1])

        # State transition (depends on dt)
        F = np.array([[1.0, -dt],
                      [0.0,  1.0]], dtype=float)

        # continuous white-noise -> discrete Q scaling
        Q = np.array([[q_hdot * dt, 0.0], [0.0, q_vdot * dt]], dtype=float)

        # Predict
        xhat = F @ xhat
        P = F @ P @ F.T + Q

        # Update
        y = z[k] - (H @ xhat)[0]
        S = H @ P @ H.T + R
        K = (P @ H.T) @ np.linalg.inv(S)
        xhat = xhat + (K @ np.array([y]))
        P = (np.eye(2) - K @ H) @ P

        xhat_hist[k, :] = xhat
        Pdiag_hist[k, :] = np.diag(P)

    return xhat_hist, Pdiag_hist
