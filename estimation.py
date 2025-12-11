from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class KF2CV:
    """
    2-state (h, v) constant-velocity Kalman Filter.

    State vector
        x = [h, v]^T

    with discrete-time process model

        x_{k+1} = F x_k + Γ a_k,      a_k ~ N(0, sigma_accel^2)

    and measurement model

        z_k = H x_k + w_k,            w_k ~ N(0, sigma_meas^2).

    where

        F = [[1, dt],
             [0, 1]]

        Γ = [[0.5 * dt^2],
             [dt]]

        H = [[1, 0]]

    and the process / measurement noise covariances are

        Q = sigma_accel^2 * Γ Γ^T
        R = sigma_meas^2.

    The filter tracks altitude (up-positive) and vertical velocity using a
    scalar altitude measurement.

    :param dt: Discrete-time step (s).
    :type dt: float
    :param sigma_meas: Standard deviation of the altitude measurement noise (m).
    :type sigma_meas: float
    :param sigma_accel: Standard deviation of the (unmodeled) vertical
                        acceleration process noise (m/s^2).
    :type sigma_accel: float
    :param x: Initial state vector ``[h, v]``. If ``None``, initialized to zeros.
    :type x: numpy.ndarray or None, optional
    :param P: Initial state covariance (2x2). If ``None``, initialized to
              a large diagonal matrix ``diag(10^2, 10^2)``.
    :type P: numpy.ndarray or None, optional
    """
    dt: float
    sigma_meas: float
    sigma_accel: float

    x: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # State and covariance initialization
        if self.x is None:
            self.x = np.zeros(2)  # [h, v]
        if self.P is None:
            # Large initial uncertainty
            self.P = np.diag([10.0**2, 10.0**2])

        # State transition and process-noise gain
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        self.Gamma = np.array([0.5 * self.dt**2, self.dt]).reshape(2, 1)

        # Measurement matrix
        self.H = np.array([[1.0, 0.0]])

        # Process and measurement noise covariances
        q_a = self.sigma_accel**2
        self.Q = q_a * (self.Gamma @ self.Gamma.T)
        self.R = np.array([[self.sigma_meas**2]])

    def predict(self) -> None:
        """
        Perform the time-update step of the Kalman Filter.

        Propagates the state and covariance forward one time step using the
        constant-velocity model and process noise covariance ``Q``:

            x_{k+1|k} = F x_{k|k}
            P_{k+1|k} = F P_{k|k} F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float) -> Tuple[float, float, np.ndarray, float]:
        """
        Perform the measurement-update step with a scalar altitude measurement.

        Given measurement

            z_k = h_k + w_k,

        this step computes the innovation, Kalman gain, and updated state
        and covariance.

        :param z: Altitude measurement (m, up-positive).
        :type z: float

        :returns: ``(h_est, v_est, Pk, sigma_innov)`` where
                  * ``h_est`` is the updated altitude estimate (m),
                  * ``v_est`` is the updated vertical-velocity estimate (m/s),
                  * ``Pk`` is the updated 2x2 covariance matrix,
                  * ``sigma_innov`` is the standard deviation of the innovation
                    (scalar), i.e. ``sqrt(S)``, where ``S`` is the innovation
                    covariance.
        :rtype: tuple[float, float, numpy.ndarray, float]
        """
        z_vec = np.array([[z]])
        y = z_vec - self.H @ self.x             # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State and covariance update
        self.x = self.x + (K @ y).ravel()
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        h_est = float(self.x[0])
        v_est = float(self.x[1])
        sigma_innov = float(np.sqrt(S[0, 0]))

        return h_est, v_est, self.P.copy(), sigma_innov

    def step(self, z: float) -> Tuple[float, float, np.ndarray, float, float]:
        """
        Perform one full predict–update cycle of the Kalman Filter.

        This convenience method calls :meth:`predict` followed by
        :meth:`update`, and additionally returns the scalar residual
        (measurement minus updated altitude estimate).

        :param z: Altitude measurement (m, up-positive).
        :type z: float

        :returns: ``(h_est, v_est, Pk, sigma_innov, residual)`` where
                  * ``h_est`` is the updated altitude estimate (m),
                  * ``v_est`` is the updated vertical-velocity estimate (m/s),
                  * ``Pk`` is the updated 2x2 covariance matrix,
                  * ``sigma_innov`` is the standard deviation of the innovation,
                  * ``residual`` is the measurement residual ``z - h_est`` (m).
        :rtype: tuple[float, float, numpy.ndarray, float, float]
        """
        self.predict()
        h_est, v_est, Pk, sigma_innov = self.update(z)
        residual = z - h_est
        return h_est, v_est, Pk, sigma_innov, residual
