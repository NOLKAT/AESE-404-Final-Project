from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from dynamics import f_rhs, rk4_step, enforce_ground_plane, kT
from estimation import KF2CV
from copy import deepcopy


# ----------------------------------------------------------------------
# Altitude PID design
# ----------------------------------------------------------------------

def design_pid_altitude_heave(
    P: dict,
    dn_fixed_deg: float = 90.0,
    zeta: float = 0.9,
    wn: float = 0.8,
    pole_ratio: float = 3.0,
) -> Dict[str, float]:
    """
    Design PID gains for altitude based on the approximate heave dynamics

        h¨ = (kT/m) * Δu

    where Δu is the increment around hover collective. The plant is
        G(s) = k / s^2
    with k = kT/m.

    The closed-loop characteristic polynomial for unity feedback with
    PID C(s) = Kp + Ki/s + Kd s is

        s^3 + k Kd s^2 + k Kp s + k Ki = 0.

    We choose a desired third-order polynomial as

        (s^2 + 2 ζ ω_n s + ω_n^2) (s + ω_3),

    with ω_3 = pole_ratio * ω_n, and match coefficients.

    Parameters
    ----------
    P : dict
        Parameter dictionary (mass and rotor efficiencies used).
    dn_fixed_deg : float
        Nacelle angle used for hover (deg).
    zeta : float
        Desired damping ratio of the dominant complex pair.
    wn : float
        Desired natural frequency [rad/s] of the dominant poles.
    pole_ratio : float
        Ratio ω_3 / ω_n for the real pole (typically 2–5).

    Returns
    -------
    dict
        {"kp": Kp, "ki": Ki, "kd": Kd}
    """
    m = P["massprops"]["m"]
    kT_hover = kT(P, dn_fixed_deg)
    k = kT_hover / max(m, 1e-9)  # heave gain [ (m/s^2) / unit collective ]

    omega_n = wn
    omega_3 = pole_ratio * omega_n

    a2 = 2.0 * zeta * omega_n + omega_3
    a1 = omega_n**2 + 2.0 * zeta * omega_n * omega_3
    a0 = omega_n**2 * omega_3

    kp = a1 / k
    ki = a0 / k
    kd = a2 / k

    return {"kp": kp, "ki": ki, "kd": kd}


# ----------------------------------------------------------------------
# Sensor and Kalman filter
# ----------------------------------------------------------------------

def baro_measurement(h_true: float, rng: np.random.Generator, sigma: float, bias: float) -> float:
    """
    Barometric altitude measurement: z = h + bias + N(0, sigma^2).
    """
    return float(h_true + bias + rng.normal(0.0, sigma))


# ----------------------------------------------------------------------
# PID controller
# ----------------------------------------------------------------------

@dataclass
class PID:
    """
    PID controller that operates on a bounded *increment* around a
    feedforward baseline u_ff (hover collective).

    u_total = clamp( u_ff + Δu, [umin, umax] ),
    with Δu produced by PID and limited to [du_min, du_max].

    This avoids large excursions far from hover and greatly reduces
    the overshoot / ringing that occurred when the full range [0,1]
    was available to the PID.
    """
    kp: float
    ki: float
    kd: float
    umin: float = 0.0
    umax: float = 1.0

    # bounds on the *increment* Δu around u_ff
    du_min: float = -0.15
    du_max: float = +0.15

    # internal state
    integrator: float = 0.0
    prev_e: float = 0.0
    initialized: bool = False

    def reset(self):
        self.integrator = 0.0
        self.prev_e = 0.0
        self.initialized = False

    def step(
        self,
        r: float,
        y: float,
        dt: float,
        u_ff: float,
        de_meas: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute saturated control u and unsaturated increment Δu_unsat.

        Parameters
        ----------
        r : float
            Reference value.
        y : float
            Measured/estimated output.
        dt : float
            Time step [s].
        u_ff : float
            Feedforward baseline command (hover collective).
        de_meas : float or None
            Optional externally supplied derivative of the error e = r - y.
            If provided, it is used instead of the finite-difference estimate.

        Returns
        -------
        u : float
            Saturated total command in [umin, umax].
        du_unsat : float
            Increment before the Δu saturation (for diagnostics).
        """
        e = r - y

        if not self.initialized:
            self.prev_e = e
            self.initialized = True

        if de_meas is None:
            de = (e - self.prev_e) / max(dt, 1e-6)
        else:
            de = de_meas

        self.prev_e = e

        # Tentative integrator update
        i_tent = self.integrator + self.ki * e * dt

        # Unsaturated increment around u_ff
        du_unsat = self.kp * e + i_tent + self.kd * de

        # Limit the increment
        du = float(np.clip(du_unsat, self.du_min, self.du_max))

        # Total command before hard saturation
        u_cmd = u_ff + du

        # Hard saturation on the total command
        u_sat = float(np.clip(u_cmd, self.umin, self.umax))

        if (du != du_unsat) and np.sign(du_unsat - du) != 0:
            pass
        else:
            self.integrator = i_tent

        return u_sat, du_unsat


# ----------------------------------------------------------------------
# Closed-loop simulator
# ----------------------------------------------------------------------

def run_pid_altitude_hold(
    P_nominal: dict,
    x0: np.ndarray,
    Tfinal: float,
    dt: float,
    h_ref: float,
    pid_gains: Dict[str, float],
    use_kf: bool = True,
    baro_sigma: float = 1.5,
    baro_bias: float = 0.5,
    baro_seed: int = 7,
    dyn_mismatch: Dict[str, float] | None = None,
    dn_fixed_deg: float = 90.0,
) -> Dict[str, np.ndarray]:
    """
    Closed-loop altitude hold around hover using collective-only PID.
    Other controls are held at zero; nacelles fixed to dn_fixed_deg.

    Parameters
    ----------
    P_nominal : dict
        Nominal dynamics parameters used for hover feedforward.
    x0 : ndarray(12,)
        Initial state.
    Tfinal : float
        Simulation horizon [s].
    dt : float
        Time step [s].
    h_ref : float
        Desired altitude (up-positive) [m].
    pid_gains : dict
        Gains {"kp": ..., "ki": ..., "kd": ...}.
    use_kf : bool
        If True, use KF altitude estimate as feedback; else baro.
    dyn_mismatch : dict or None
        Optional plant parameter multipliers applied to a deep copy
        of P_nominal (mass and rotor gains).
    dn_fixed_deg : float
        Fixed nacelle angle [deg].

    Returns
    -------
    dict with logs: t, X, U, h_true, z_baro, h_hat, Pdiag, dcol_hover_ff,
    dcol_unsat (increment), residuals, sigma_innov.
    """

    P_dyn = deepcopy(P_nominal)
    if dyn_mismatch:
        if "mass_mult" in dyn_mismatch:
            P_dyn["massprops"]["m"] *= float(dyn_mismatch["mass_mult"])
        if "kT_hover_mult" in dyn_mismatch:
            if P_dyn["rotor_eff"].get("kT_hover", None) is not None:
                P_dyn["rotor_eff"]["kT_hover"] *= float(dyn_mismatch["kT_hover_mult"])
        if "kT_air_scale_mult" in dyn_mismatch:
            P_dyn["rotor_eff"]["kT_airplane_scale"] *= float(dyn_mismatch["kT_air_scale_mult"])

    # Hover feedforward
    m_nom = P_nominal["massprops"]["m"]
    g = P_nominal["env"]["gravity"]
    dcol_hover_ff = (m_nom * g) / max(1e-6, kT(P_nominal, dn_fixed_deg))

    # PID controller
    pid = PID(
        kp=pid_gains.get("kp", 0.02),
        ki=pid_gains.get("ki", 0.002),
        kd=pid_gains.get("kd", 0.01),
        umin=0.0,
        umax=1.0,
        du_min=-0.15,
        du_max=+0.15,
    )

    # Time base
    N = int(np.floor(Tfinal / dt)) + 1
    t = np.linspace(0.0, Tfinal, N)

    # Logs
    X = np.zeros((N, 12))
    U = np.zeros((N, 5))
    Htrue = np.zeros(N)
    Z = np.zeros(N)
    Hhat = np.zeros(N)
    Pdiag = np.zeros((N, 2))
    dcol_unsat_hist = np.zeros(N)
    residuals = np.zeros(N)
    sigma_innov = np.zeros(N)

    # Initial state
    X[0, :] = x0.copy()
    Htrue[0] = -X[0, 2]
    rng = np.random.default_rng(baro_seed)

    # KF
    if use_kf:
        kf = KF2CV(dt=dt, sigma_meas=baro_sigma, sigma_accel=0.8)
        kf.x = np.array([Htrue[0], 0.0])
        kf.P = np.diag([5.0**2, 5.0**2])
        Hhat[0] = Htrue[0]
        Pdiag[0, :] = np.diag(kf.P)
        residuals[0] = 0.0
        sigma_innov[0] = baro_sigma

    # Simulation loop
    x = X[0, :].copy()
    for k in range(1, N):
        h_true = -x[2]
        Htrue[k - 1] = h_true

        # Barometric measurement
        z = baro_measurement(h_true, rng, sigma=baro_sigma, bias=baro_bias)
        Z[k - 1] = z

        # Estimation
        if use_kf:
            h_est, v_est, Pk, sig_y, res = kf.step(z)
            Hhat[k - 1] = h_est
            Pdiag[k - 1, :] = np.diag(Pk)
            residuals[k - 1] = res
            sigma_innov[k - 1] = sig_y

            y = h_est
            # e = h_ref - h_est, so e_dot = -v_est
            de_meas = -v_est
        else:
            y = z
            de_meas = None

        d_col_cmd, du_unsat = pid.step(h_ref, y, dt, u_ff=dcol_hover_ff, de_meas=de_meas)
        dcol_unsat_hist[k - 1] = du_unsat

        # Control vector (collective + fixed nacelle)
        u_vec = np.array([d_col_cmd, 0.0, 0.0, 0.0, dn_fixed_deg], dtype=float)
        U[k - 1, :] = u_vec

        # Integrate one step
        ufun = (lambda uv: (lambda tt, xx: uv))(u_vec)
        x = rk4_step(f_rhs, t[k - 1], x, dt, ufun, P_dyn)
        x = enforce_ground_plane(x)
        X[k, :] = x

    # Logs
    Htrue[-1] = -X[-1, 2]
    Z[-1] = baro_measurement(Htrue[-1], rng, sigma=baro_sigma, bias=baro_bias)
    if use_kf:
        h_est, _, Pk, sig_y, res = kf.step(Z[-1])
        Hhat[-1] = h_est
        Pdiag[-1, :] = np.diag(Pk)
        residuals[-1] = res
        sigma_innov[-1] = sig_y

    return {
        "t": t,
        "X": X,
        "U": U,
        "h_true": Htrue,
        "z_baro": Z,
        "h_hat": Hhat if use_kf else None,
        "Pdiag": Pdiag if use_kf else None,
        "dcol_hover_ff": dcol_hover_ff,
        "dcol_unsat": dcol_unsat_hist,
        "residuals": residuals if use_kf else None,
        "sigma_innov": sigma_innov if use_kf else None,
    }
