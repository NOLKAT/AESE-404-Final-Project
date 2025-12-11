from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from dynamics import aero_longitudinal_forces_moments, kT


def ensure_outdir(path: str) -> None:
    """
    Ensure that the given directory exists.

    :param path: Directory path to create if needed.
    :type path: str
    """
    os.makedirs(path, exist_ok=True)


def pretty_name(key: str) -> str:
    """
    Map a scenario key to a human-friendly name.

    :param key: Scenario key string.
    :type key: str
    :returns: Scenario name.
    :rtype: str
    """
    return {
        "Hover": "Hover",
        "Conversion_Ramp": "Conversion Ramp",
        "Simulation": "20-Second Simulation",
    }.get(key, key)


def _derived_series(t: np.ndarray, X: np.ndarray, U: np.ndarray, P: dict) -> tuple[np.ndarray, ...]:
    """
    Compute time histories of derived flight quantities.

    For each time index i, the following items are computed:
      * V: airspeed magnitude (m/s)
      * alpha: angle of attack (rad)
      * qbar: dynamic pressure (Pa)
      * L: lift estimate from minimal aerodynamics (N)
      * Tmag: thrust magnitude (N)
      * VS: vertical support (N) = T*sin(dn) + L*cos(alpha)
      * mg: weight (N), returned as a scalar (constant in time)

    :param t: Time vector (s)
    :type t: numpy.ndarray
    :param X: State history with columns: [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    :type X: numpy.ndarray
    :param U: Input history with columns: [d_col, d_lon, d_lat, d_ped, dn_deg]
    :type U: numpy.ndarray
    :param P: Parameter dictionary (env, aero, mass properties)
    :type P: dict
    :returns: Tuple (V, alpha, qbar, L, Tmag, VS, mg)
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                  numpy.ndarray, numpy.ndarray, float]
    """
    m = P["massprops"]["m"]
    g = P["env"]["gravity"]

    V = np.zeros_like(t)
    alpha = np.zeros_like(t)
    qbar = np.zeros_like(t)
    L = np.zeros_like(t)
    Tmag = np.zeros_like(t)
    VS = np.zeros_like(t)

    for i in range(len(t)):
        u, w, q = X[i, 3], X[i, 5], X[i, 10]
        d_col, d_lon, _, _, dn_deg = U[i, :]
        delta_e = P["aero"]["ke_lon_to_elev"] * d_lon

        Fa_b, _, Vi, alphai = aero_longitudinal_forces_moments(u, w, q, P, delta_e)
        Li = -Fa_b[2] * np.cos(alphai) + Fa_b[0] * np.sin(alphai)
        Li = max(0.0, Li)

        Ti = kT(P, dn_deg) * d_col
        sdn = np.sin(np.deg2rad(dn_deg))

        V[i] = Vi
        alpha[i] = alphai
        qbar[i] = 0.5 * P["env"]["air_density"] * Vi * Vi
        L[i] = Li
        Tmag[i] = Ti
        VS[i] = Ti * sdn + Li * np.cos(alphai)

    mg = m * g
    return V, alpha, qbar, L, Tmag, VS, mg


def save_plots(t: np.ndarray, X: np.ndarray, U: np.ndarray, scenario_key: str, P: dict, outdir: str = "plots") -> None:
    """
    Create and save graphs for a single open-loop simulation scenario.

    Graphs are saved as PNG files under ``outdir``.

    :param t: Time vector (s)
    :type t: numpy.ndarray
    :param X: State history with columns: [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    :type X: numpy.ndarray
    :param U: Input history with columns: [d_col, d_lon, d_lat, d_ped, dn_deg]
    :type U: numpy.ndarray
    :param scenario_key: Scenario key string
    :type scenario_key: str
    :param P: Parameter dictionary
    :type P: dict
    :param outdir: Output directory for graphs. Defaults to "plots".
    :type outdir: str
    :returns: None
    :rtype: None
    """
    ensure_outdir(outdir)
    title_prefix = pretty_name(scenario_key)

    # States
    pn, pe, pd = X[:, 0], X[:, 1], X[:, 2]
    u, v, w = X[:, 3], X[:, 4], X[:, 5]
    phi, th, psi = np.rad2deg(X[:, 6]), np.rad2deg(X[:, 7]), np.rad2deg(X[:, 8])
    p, q, r = np.rad2deg(X[:, 9]), np.rad2deg(X[:, 10]), np.rad2deg(X[:, 11])
    altitude = -pd  # up-positive

    V, alpha, qbar, L, Tmag, VS, mg = _derived_series(t, X, U, P)

    mark = max(1, len(t) // 40)

    # Positions vs. time
    plt.figure()
    plt.plot(t, pn, linestyle="-", marker="o", markevery=mark, label="North position N (m)")
    plt.plot(t, pe, linestyle="--", marker="x", markevery=mark, label="East position E (m)")
    plt.plot(t, altitude, linestyle=":", marker="^", markevery=mark, label="Altitude (m)")
    plt.axhline(0.0, linestyle="--", linewidth=1, label="Ground (Altitude = 0 m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position / Altitude (m)")
    plt.title(f"{title_prefix}: Positions vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{scenario_key}_Positions.png"),
                dpi=140, bbox_inches="tight")

    # Body-axis speeds vs. time
    plt.figure()
    plt.plot(t, u, linestyle="-", marker="o", markevery=mark, label="Body-forward speed u (m/s)")
    plt.plot(t, v, linestyle="--", marker="x", markevery=mark, label="Body-right speed v (m/s)")
    plt.plot(t, w, linestyle=":", marker="^", markevery=mark, label="Body-down speed w (m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(f"{title_prefix}: Body Speeds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{scenario_key}_Speeds.png"),
                dpi=140, bbox_inches="tight")

    # Attitude vs. time
    plt.figure()
    plt.plot(t, phi, linestyle="-", marker="o", markevery=mark, label="Roll (deg)")
    plt.plot(t, th, linestyle="--", marker="x", markevery=mark, label="Pitch (deg)")
    plt.plot(t, psi, linestyle=":", marker="^", markevery=mark, label="Yaw (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(f"{title_prefix}: Attitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{scenario_key}_Attitude.png"),
                dpi=140, bbox_inches="tight")

    # Body rates vs. time
    plt.figure()
    plt.plot(t, p, linestyle="-", marker="o", markevery=mark, label="Roll rate p (deg/s)")
    plt.plot(t, q, linestyle="--", marker="x", markevery=mark, label="Pitch rate q (deg/s)")
    plt.plot(t, r, linestyle=":", marker="^", markevery=mark, label="Yaw rate r (deg/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (deg/s)")
    plt.title(f"{title_prefix}: Body Rates")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{scenario_key}_Rates.png"),
                dpi=140, bbox_inches="tight")

    # 3-D flight path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pe, pn, altitude, linestyle="-", marker="o", markevery=mark)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"{title_prefix}: 3D Flight Path")
    plt.savefig(os.path.join(outdir, f"{scenario_key}_3D_Path.png"),
                dpi=140, bbox_inches="tight")

    # Flight quantities vs. time
    plt.figure()
    plt.plot(t, V, linestyle="-", marker="o", markevery=mark, label="Airspeed V (m/s)")
    plt.plot(t, np.rad2deg(alpha), linestyle="--", marker="x", markevery=mark,
             label="Angle of attack α (deg)")
    plt.plot(t, qbar, linestyle=":", marker="^", markevery=mark, label="Dynamic pressure q (Pa)")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(f"{title_prefix}: Flight Dynamics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{scenario_key}_Flight_Characteristics.png"),
                dpi=140, bbox_inches="tight")

    # Vertical support budget vs. time
    plt.figure()
    plt.plot(t, VS, linestyle="-", marker="o", markevery=mark,
             label="Available vertical support (N)")
    plt.axhline(mg, linestyle="--", label="Weight mg (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(f"{title_prefix}: Vertical Support Budget")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{scenario_key}_Vertical_Support.png"),
                dpi=140, bbox_inches="tight")

    plt.close("all")


# ----------------------------------------------------------------------
# Closed-loop
# ----------------------------------------------------------------------
def plot_altitude_vs_time(results: dict, outdir: str = "plots") -> None:
    """
    Plot true altitude, barometric measurement, KF estimate, and reference vs time

    :param results: Dictionary returned by simulate_altitude_control_20s
    :type results: dict
    :param outdir: Output directory for the PNG file
    :type outdir: str
    """
    ensure_outdir(outdir)
    t = results["t"]
    h_true = results["h_true"]
    z_baro = results["z_baro"]
    h_hat = results["h_hat"]
    href = results["href"]

    plt.figure()
    plt.plot(t, h_true, label="True altitude", linewidth=1.8)
    plt.plot(t, z_baro, label="Barometric altitude", alpha=0.4)
    plt.plot(t, h_hat, "--", label="Kalman filter altitude", linewidth=1.6)
    plt.plot(t, href, "k:", label="Reference altitude", linewidth=1.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Time ")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outdir, "Altitude_vs_Time.png"),
                dpi=140, bbox_inches="tight")
    plt.close()


def plot_control_inputs_vs_time(results: dict, outdir: str = "plots") -> None:
    """
    Plot collective command time history against the nominal hover baseline

    :param results: Dictionary returned by simulate_altitude_control_20s
    :type results: dict
    :param outdir: Output directory for the PNG file
    :type outdir: str
    """
    ensure_outdir(outdir)
    t = results["t"]
    U = results["U"]
    dcol_hover = results["dcol_hover"]

    plt.figure()
    plt.plot(t, U[:, 0], label="Collective command $d_{col}$", linewidth=1.8)
    plt.axhline(dcol_hover, color="k", linestyle=":", label="Hover collective baseline")
    plt.xlabel("Time (s)")
    plt.ylabel("Collective (unitless)")
    plt.title("Collective Command vs Time ")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outdir, "Control_Inputs_vs_Time.png"),
                dpi=140, bbox_inches="tight")
    plt.close()


def plot_kalman_covariance_vs_time(results: dict, outdir: str = "plots") -> None:
    """
    Plot diagonal entries of the 2x2 KF covariance P over time

    :param results: Dictionary returned by simulate_altitude_control_20s
    :type results: dict
    :param outdir: Output directory for the PNG file
    :type outdir: str
    """
    ensure_outdir(outdir)
    t = results["t"]
    Pdiag = results["Pdiag"]

    plt.figure()
    plt.plot(t, Pdiag[:, 0], label="Variance of altitude", linewidth=1.8)
    plt.plot(t, Pdiag[:, 1], "--", label="Variance of vertical velocity", linewidth=1.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Variance")
    plt.title("Kalman Filter Covariance vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outdir, "Kalman_Covariance_vs_Time.png"),
                dpi=140, bbox_inches="tight")
    plt.close()


def plot_orientation_vs_time(results: dict, outdir: str = "plots") -> None:
    """
    Plot Euler angle orientation vs time (roll, pitch, yaw), in degrees

    :param results: Dictionary returned by simulate_altitude_control_20s
    :type results: dict
    :param outdir: Output directory for the PNG file
    :type outdir: str
    """
    ensure_outdir(outdir)
    t = results["t"]
    X = results["X"]

    phi_deg = np.rad2deg(X[:, 6])
    th_deg = np.rad2deg(X[:, 7])
    psi_deg = np.rad2deg(X[:, 8])

    plt.figure()
    plt.plot(t, phi_deg, label="Roll (deg)", linewidth=1.6)
    plt.plot(t, th_deg, label="Pitch (deg)", linewidth=1.6)
    plt.plot(t, psi_deg, label="Yaw (deg)", linewidth=1.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Orientation vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outdir, "Orientation_vs_Time.png"),
                dpi=140, bbox_inches="tight")
    plt.close()


def plot_altitude_ic_scenarios(results_list, h_ref, outdir: str, fname: str = "altitude_ic_scenarios.png"):
    """
    Plot altitude responses for multiple initial conditions on one figure.
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 4))

    for res in results_list:
        t = res["t"]
        h_true = res["h_true"]
        label = res.get("label", "scenario")
        plt.plot(t, h_true, label=f"{label}: h_true")

    # Reference line
    t_ref = results_list[0]["t"]
    plt.plot(t_ref, h_ref * np.ones_like(t_ref), "k--", label="h_ref")

    plt.xlabel("Time [s]")
    plt.ylabel("Altitude h [m]")
    plt.title("Altitude responses for different initial conditions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(outdir, fname)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_altitude_mismatch_scenarios(results_list, h_ref, outdir: str, fname: str = "altitude_mismatch_scenarios.png"):
    """
    Plot altitude responses for different parameter-mismatch cases.
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 4))

    for res in results_list:
        t = res["t"]
        h_true = res["h_true"]
        label = res.get("label", "scenario")
        plt.plot(t, h_true, label=f"{label}: h_true")

    t_ref = results_list[0]["t"]
    plt.plot(t_ref, h_ref * np.ones_like(t_ref), "k--", label="h_ref")

    plt.xlabel("Time [s]")
    plt.ylabel("Altitude h [m]")
    plt.title("Altitude responses under model parameter mismatch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(outdir, fname)
    plt.savefig(outpath, dpi=150)
    plt.close()