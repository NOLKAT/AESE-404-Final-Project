from typing import Callable
import numpy as np
from dynamics import clamp, kT, aero_longitudinal_forces_moments

ScheduleFunc = Callable[[float, np.ndarray], np.ndarray]


def _smoothstep(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """
    Interpolate smoothly between two values on a finite time interval.


    :param t: Query time (s).
    :type t: float
    :param t0: Start time of the transition (s).
    :type t0: float
    :param t1: End time of the transition (s).
    :type t1: float
    :param v0: Value at t <= t0
    :type v0: float
    :param v1: Value at t >= t1
    :type v1: float
    :returns: Interpolated value at time t
    :rtype: float
    """
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    u = (t - t0) / max(1e-9, (t1 - t0))
    s = u * u * (3.0 - 2.0 * u)
    return v0 + (v1 - v0) * s


def hover_simulation(P: dict) -> ScheduleFunc:
    """
    Generate a hover simulation with a smooth longitudinal cyclic ramp.

    :param P: uses massprops.m, env.gravity, and the thrust scheduler via function dynamics.kT
    :type P: dict
    :returns: Function u(t, x) -> ndarray([d_col, d_lon, d_lat, d_ped, dn_deg])
    :rtype: ScheduleFunc
    """
    m, g = P["massprops"]["m"], P["env"]["gravity"]
    dcol_hover = (m * g) / kT(P, 90.0)

    def u_of_t(t: float, x: np.ndarray) -> np.ndarray:
        d_col = dcol_hover
        d_lon = _smoothstep(t, 5.0, 5.5, 0.0, 0.03)
        d_lat = 0.0
        d_ped = 0.0
        dn = 90.0
        return np.array([d_col, d_lon, d_lat, d_ped, dn], dtype=float)

    return u_of_t


def conversion_ramp_simulation(P: dict) -> ScheduleFunc:
    """
    Generate a constant-collective conversion ramp simulation with nacelle rate limiting.

    :param P: uses massprops.m, env.gravity, nacelle.dn_rate_max_degps, and function dynamics.kT
    :type P: dict
    :returns: Function u(t, x) -> ndarray([d_col, d_lon, d_lat, d_ped, dn_deg])
    :rtype: ScheduleFunc
    """
    m, g = P["massprops"]["m"], P["env"]["gravity"]
    dcol_hover = (m * g) / kT(P, 90.0)
    dn_rate = P["nacelle"]["dn_rate_max_degps"]
    last = {"t": 0.0, "dn": 90.0}

    def u_of_t(t: float, x: np.ndarray) -> np.ndarray:
        dt = t - last["t"]
        dn_des = max(30.0, 90.0 - 60.0 * min(max(t, 0.0), 12.0) / 12.0)
        step = clamp(dn_des - last["dn"], -dn_rate * max(dt, 0.0), dn_rate * max(dt, 0.0))
        dn = clamp(last["dn"] + step, P["nacelle"]["dn_min"], P["nacelle"]["dn_max"])
        last["t"] = t
        last["dn"] = dn
        return np.array([dcol_hover, 0.0, 0.0, 0.0, dn], dtype=float)

    return u_of_t


def Simulation(P: dict) -> ScheduleFunc:
    """
    Generate a 20-second simulation for liftoff, hover, conversion, brief cruise,
    and reconversion/landing for V-22 Helicotper

    Timeline (s) and behavior:
      * ``0–3``: liftoff to ~10 m
      * ``3–6``: hover
      * ``6–12``: conversion and acceleration
      * ``12–16``: short airplane mode
      * ``16–20``: reconversion and landing

    :param P: uses massprops.m, env.gravity, aero.*, and function dynamics.aero_longitudinal_forces_moments
    :type P: dict
    :returns: Function u(t, x) -> ndarray([d_col, d_lon, d_lat, d_ped, dn_deg])
    :rtype: ScheduleFunc
    """
    m, g = P["massprops"]["m"], P["env"]["gravity"]
    last = {"t": 0.0, "dn": 90.0}

    def u_of_t(t: float, x: np.ndarray) -> np.ndarray:
        pn, pe, pd, u, v, w, phi, th, psi, pr, qr, rr = x
        d_lat = 0.0
        d_ped = 0.0

        if t < 3.0:
            dn_des = 90.0
            T_des = 1.10 * m * g if t < 1.5 else 1.00 * m * g
            d_lon = 0.0

        elif t < 6.0:
            dn_des = 90.0
            T_des = 1.00 * m * g
            d_lon = 0.0

        elif t < 12.0:
            dn_des = 90.0 - (t - 6.0) * (80.0 / 6.0)  # 90 -> 10 deg
            d_lon = 0.02
            delta_e = P["aero"]["ke_lon_to_elev"] * d_lon
            Fa_b, _, V, alpha = aero_longitudinal_forces_moments(u, w, qr, P, delta_e)
            L_est = -Fa_b[2] * np.cos(alpha) + Fa_b[0] * np.sin(alpha)
            L_est = max(0.0, L_est)
            sdn = np.sin(np.deg2rad(max(1.0, min(89.0, dn_des))))
            T_des = max(0.0, m * g - L_est) / sdn

        elif t < 16.0:
            dn_des = 10.0
            d_lon = 0.015
            delta_e = P["aero"]["ke_lon_to_elev"] * d_lon
            Fa_b, _, V, alpha = aero_longitudinal_forces_moments(u, w, qr, P, delta_e)
            L_est = -Fa_b[2] * np.cos(alpha) + Fa_b[0] * np.sin(alpha)
            L_est = max(0.0, L_est)
            sdn = np.sin(np.deg2rad(max(1.0, dn_des)))
            T_des = max(0.0, (m * g - L_est)) / sdn if (m * g - L_est) > 0 else 0.2 * m * g

        else:
            dn_des = 10.0 + (t - 16.0) * (80.0 / 4.0)  # 10 -> 90 over 4 s
            d_lon = -0.01
            delta_e = P["aero"]["ke_lon_to_elev"] * d_lon
            Fa_b, _, V, alpha = aero_longitudinal_forces_moments(u, w, qr, P, delta_e)
            L_est = -Fa_b[2] * np.cos(alpha) + Fa_b[0] * np.sin(alpha)
            L_est = max(0.0, L_est)
            sdn = np.sin(np.deg2rad(max(1.0, min(89.0, dn_des))))
            vert_need = 0.95 * m * g - L_est if (t > 18.0) else (m * g - L_est)
            T_des = max(0.0, vert_need) / sdn if vert_need > 0 else 0.2 * m * g

        # Nacelle rate limiting
        dt_local = max(1e-9, t - last["t"])
        step_max = P["nacelle"]["dn_rate_max_degps"] * dt_local
        dn_cmd = clamp(dn_des - last["dn"], -step_max, step_max) + last["dn"]
        dn_cmd = clamp(dn_cmd, P["nacelle"]["dn_min"], P["nacelle"]["dn_max"])
        last["t"] = t
        last["dn"] = dn_cmd

        # Collective from desired thrust
        d_col = T_des / max(1e-3, kT(P, dn_cmd))
        d_col = clamp(d_col, 0.0, 1.0)

        return np.array([d_col, d_lon, d_lat, d_ped, dn_cmd], dtype=float)

    return u_of_t
