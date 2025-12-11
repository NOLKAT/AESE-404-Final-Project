from __future__ import annotations
import numpy as np

from params import Params as P, X0
from linearizaton import linearization
from dynamics import f_rhs, rk4_step, enforce_ground_plane, kT
from control import (
    KF2CV,
    PID,
    baro_measurement,
    run_pid_altitude_hold,
    design_pid_altitude_heave,
)
from plotting import (
    ensure_outdir,
    plot_altitude_vs_time,
    plot_control_inputs_vs_time,
    plot_kalman_covariance_vs_time,
    plot_orientation_vs_time,
)

PLOT_DIR = "plots"


# ----------------------------------------------------------------------
# 20-second mission-like reference: altitude + nacelle angle
# ----------------------------------------------------------------------
def mission_profile_20s(
    t: float,
    hover_alt: float,
    cruise_alt: float,
    dn_hover: float = 90.0,
    dn_air: float = 10.0,
) -> tuple[float, float]:
    """
    Generate a 20-second mission-like reference profile for altitude and nacelle angle.

    Phases (seconds):
      0–3   : liftoff, climb 0 -> hover_alt, nacelles at dn_hover
      3–6   : out-of-ground-effect hover at hover_alt, nacelles at dn_hover
      6–12  : conversion and climb to cruise_alt, nacelles tilt dn_hover -> dn_air
      12–16 : short airplane-like segment at cruise_alt, nacelles at dn_air
      16–20 : reconversion and descent cruise_alt -> 0, nacelles tilt dn_air -> dn_hover

    :param t: Time (s).
    :type t: float
    :param hover_alt: Hover altitude (m, up-positive).
    :type hover_alt: float
    :param cruise_alt: Cruise altitude (m, up-positive).
    :type cruise_alt: float
    :param dn_hover: Hover nacelle angle (deg), typically 90 deg.
    :type dn_hover: float
    :param dn_air: Airplane-mode nacelle angle (deg), e.g., 10 deg.
    :type dn_air: float

    :returns: (h_ref, dn_ref), where
              * h_ref is the altitude reference (m, up-positive),
              * dn_ref is the nacelle angle reference (deg).
    :rtype: tuple[float, float]
    """
    if t < 0.0:
        return 0.0, dn_hover

    if t < 3.0:
        # 0–3: ramp 0 -> hover_alt, dn fixed at hover
        h_ref = hover_alt * (t / 3.0)
        dn_ref = dn_hover
    elif t < 6.0:
        # 3–6: hold hover_alt, dn hover
        h_ref = hover_alt
        dn_ref = dn_hover
    elif t < 12.0:
        # 6–12: ramp hover_alt -> cruise_alt, dn hover -> air
        s = (t - 6.0) / 6.0  # in [0, 1]
        h_ref = hover_alt + (cruise_alt - hover_alt) * s
        dn_ref = dn_hover + (dn_air - dn_hover) * s
    elif t < 16.0:
        # 12–16: hold cruise_alt, dn air
        h_ref = cruise_alt
        dn_ref = dn_air
    elif t <= 20.0:
        # 16–20: ramp cruise_alt -> 0, dn air -> hover
        s = (t - 16.0) / 4.0  # in [0, 1]
        h_ref = cruise_alt * max(0.0, 1.0 - s)
        dn_ref = dn_air + (dn_hover - dn_air) * s
    else:
        # after 20 s, assume back on ground in hover configuration
        h_ref = 0.0
        dn_ref = dn_hover

    return float(h_ref), float(dn_ref)


# ----------------------------------------------------------------------
# 20-second closed-loop mission simulation (PID + KF)
# ----------------------------------------------------------------------
def simulate_mission_20s_closed_loop(
    P: dict,
    x0: np.ndarray,
    Tfinal: float = 20.0,
    hover_alt: float = 10.0,
    cruise_alt: float = 25.0,
    dt: float | None = None,
    baro_sigma: float = 1.5,
    baro_bias: float = 0.5,
    baro_seed: int = 7,
    pid_gains: dict | None = None,
) -> dict:
    """
    Simulate a 20-second closed-loop mission-like maneuver with PID altitude control.

    The reference is the 5-phase mission_profile_20s(t, hover_alt, cruise_alt),
    which prescribes both altitude and nacelle angle. The PID acts on altitude
    (using a 2-state Kalman Filter estimate) by adjusting collective around a
    time-varying feedforward baseline that balances weight at the current
    nacelle angle.

    Other controls (lon/lat/ped) are set to zero.

    :param P: Global parameter dictionary containing mass properties, environment,
              rotor efficiency, and simulation settings.
    :type P: dict
    :param x0: Initial 12-state vector
               ``[pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]``.
    :type x0: numpy.ndarray
    :param Tfinal: Final simulation time (s). Default is 20.0.
    :type Tfinal: float, optional
    :param hover_alt: Hover altitude level (m, up-positive).
    :type hover_alt: float, optional
    :param cruise_alt: Cruise altitude level (m, up-positive).
    :type cruise_alt: float, optional
    :param dt: Simulation time step (s). If ``None``, uses ``P["sim"]["dt"]``.
    :type dt: float or None, optional
    :param baro_sigma: Standard deviation of the barometric altitude measurement
                       noise (m). Default is 1.5.
    :type baro_sigma: float, optional
    :param baro_bias: Constant bias added to the barometric altitude measurement (m).
                      Default is 0.5.
    :type baro_bias: float, optional
    :param baro_seed: Random seed for the barometric measurement noise generator.
                      Default is 7.
    :type baro_seed: int, optional
    :param pid_gains: PID gains as a dictionary with keys ``"kp"``, ``"ki"``,
                      and ``"kd"``. If ``None``, default gains are used.
    :type pid_gains: dict or None, optional

    :returns: Dictionary containing simulation logs:

              * ``"t"``: time vector (s), shape ``(N,)``,
              * ``"X"``: state history, shape ``(N, 12)``,
              * ``"U"``: control input history
                ``[d_col, d_lon, d_lat, d_ped, dn_deg]``, shape ``(N, 5)``,
              * ``"h_true"``: true altitude (m, up-positive), shape ``(N,)``,
              * ``"z_baro"``: barometric altitude measurements (m), shape ``(N,)``,
              * ``"h_hat"``: KF altitude estimates (m), shape ``(N,)``,
              * ``"href"``: altitude reference trajectory (m), shape ``(N,)``,
              * ``"dn_ref"``: nacelle-angle reference (deg), shape ``(N,)``,
              * ``"Pdiag"``: diagonal elements of the KF covariance matrix
                (altitude and vertical-velocity variances), shape ``(N, 2)``,
              * ``"dcol_hover"``: hover collective baseline at dn = 90 deg
                (unitless scalar).
    :rtype: dict
    """
    if dt is None:
        dt = P["sim"]["dt"]

    if pid_gains is None:
        pid_gains = dict(kp=0.015, ki=0.003, kd=0.005)

    # Time base
    N = int(np.floor(Tfinal / dt)) + 1
    t = np.linspace(0.0, Tfinal, N)

    # State and logs
    X = np.zeros((N, 12))
    X[0, :] = x0.copy()

    U = np.zeros((N, 5))
    h_true = np.zeros(N)
    z_baro = np.zeros(N)
    h_hat = np.zeros(N)
    href = np.zeros(N)
    dn_ref_hist = np.zeros(N)
    Pdiag = np.zeros((N, 2))

    # Mass and gravity for feedforward
    m = P["massprops"]["m"]
    g = P["env"]["gravity"]

    # Simple hover baseline for plotting (dn = 90 deg)
    dcol_hover = (m * g) / max(1e-6, kT(P, 90.0))

    # PID (incremental around time-varying feedforward)
    pid = PID(
        kp=pid_gains["kp"],
        ki=pid_gains["ki"],
        kd=pid_gains["kd"],
        umin=0.0,
        umax=1.0,
    )

    # KF for altitude + vertical velocity
    rng = np.random.default_rng(baro_seed)
    h0 = -X[0, 2]  # altitude up-positive
    kf = KF2CV(dt=dt, sigma_meas=baro_sigma, sigma_accel=0.8)
    kf.x = np.array([h0, 0.0])
    kf.P = np.diag([5.0**2, 5.0**2])

    # Initial logging
    h_true[0] = h0
    z_baro[0] = baro_measurement(h_true[0], rng, sigma=baro_sigma, bias=baro_bias)
    h_est, v_est, Pk, _, _ = kf.step(z_baro[0])
    h_hat[0] = h_est
    Pdiag[0, :] = np.diag(Pk)

    h_ref0, dn_ref0 = mission_profile_20s(0.0, hover_alt, cruise_alt)
    href[0] = h_ref0
    dn_ref_hist[0] = dn_ref0

    x = X[0, :].copy()
    for k in range(N - 1):
        tk = t[k]

        # Mission-style reference: altitude + nacelle angle
        h_ref, dn_ref = mission_profile_20s(tk, hover_alt, cruise_alt)
        href[k] = h_ref
        dn_ref_hist[k] = dn_ref

        # True altitude from current state
        h_true[k] = -x[2]

        # Barometric measurement
        z = baro_measurement(h_true[k], rng, sigma=baro_sigma, bias=baro_bias)
        z_baro[k] = z

        # KF update
        h_est, v_est, Pk, _, _ = kf.step(z)
        h_hat[k] = h_est
        Pdiag[k, :] = np.diag(Pk)

        # PID on KF altitude, using KF velocity as derivative of error
        # e = h_ref - h_est, so de/dt ≈ -v_est (assuming slow reference variation)
        de_meas = -v_est
        # Time-varying feedforward collective for current nacelle angle
        dcol_ff = (m * g) / max(1e-6, kT(P, dn_ref))
        d_col_cmd, _ = pid.step(h_ref, h_est, dt, u_ff=dcol_ff, de_meas=de_meas)

        # Control vector
        u_vec = np.array([d_col_cmd, 0.0, 0.0, 0.0, dn_ref], dtype=float)
        U[k, :] = u_vec

        # Integrate one step
        ufun = (lambda uv: (lambda tt, xx: uv))(u_vec)
        x = rk4_step(f_rhs, tk, x, dt, ufun, P)
        x = enforce_ground_plane(x)
        X[k + 1, :] = x

    # Final sample logs
    h_true[-1] = -X[-1, 2]
    z_baro[-1] = baro_measurement(h_true[-1], rng, sigma=baro_sigma, bias=baro_bias)
    h_est, v_est, Pk, _, _ = kf.step(z_baro[-1])
    h_hat[-1] = h_est
    Pdiag[-1, :] = np.diag(Pk)
    h_ref_last, dn_ref_last = mission_profile_20s(t[-1], hover_alt, cruise_alt)
    href[-1] = h_ref_last
    dn_ref_hist[-1] = dn_ref_last

    return {
        "t": t,
        "X": X,
        "U": U,
        "h_true": h_true,
        "z_baro": z_baro,
        "h_hat": h_hat,
        "href": href,
        "dn_ref": dn_ref_hist,
        "Pdiag": Pdiag,
        "dcol_hover": dcol_hover,
    }


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    """
    Run linearization and closed-loop altitude control simulations, then generate plots.

    This driver function:

      1. Ensures the plot output directory exists.
      2. Linearizes the nonlinear dynamics about a 10 m hover equilibrium.
      3. Designs altitude PID gains from the simplified heave model.
      4. Runs a 20-second closed-loop mission-like maneuver with a 5-phase
         altitude and nacelle-angle reference profile.
      5. Generates standard plots for altitude, control inputs, Kalman
         covariance, and attitude for this mission run.
      6. Evaluates closed-loop performance for multiple initial conditions
         (robustness to different starting altitudes).
      7. Evaluates closed-loop performance under parameter mismatch
         (robustness to mass and thrust-gain variations).
      8. Saves all plots into ``PLOT_DIR``.

    The function uses global configuration/initial-condition objects:
    ``P`` (parameter dictionary), ``X0`` (initial state), and ``PLOT_DIR``.
    It does not return anything; results are written to disk as image files.
    """
    # ------------------------------------------------------------------
    # Setup and linearization
    # ------------------------------------------------------------------
    ensure_outdir(PLOT_DIR)

    # Linearization about hover at 10 m
    linearization()

    # ------------------------------------------------------------------
    # Nominal simulation setup
    # ------------------------------------------------------------------
    # Initial condition: on the ground, near-level (h = 0 m)
    x0 = X0.copy()
    x0[2] = 0.0  # p_d (down), so altitude h = -p_d = 0 m at t = 0

    dt = P["sim"]["dt"]

    # PID gains from heave model
    gains = design_pid_altitude_heave(
        P,
        dn_fixed_deg=90.0,
        zeta=0.9,
        wn=0.8,
        pole_ratio=3.0,
    )
    print("Designed altitude PID gains from heave model:", gains)

    # Simulation parameters
    Tfinal = 20.0
    hover_alt = 10.0
    cruise_alt = 25.0
    baro_sigma = 1.5
    baro_bias = 0.5
    baro_seed = 7

    # ------------------------------------------------------------------
    # 20-second closed-loop mission run (used for the main 4 plots)
    # ------------------------------------------------------------------
    results = simulate_mission_20s_closed_loop(
        P=P,
        x0=x0,
        Tfinal=Tfinal,
        hover_alt=hover_alt,
        cruise_alt=cruise_alt,
        dt=dt,
        baro_sigma=baro_sigma,
        baro_bias=baro_bias,
        baro_seed=baro_seed,
        pid_gains=gains,
    )

    # Plots for the nominal mission scenario
    plot_altitude_vs_time(results, outdir=PLOT_DIR)
    plot_control_inputs_vs_time(results, outdir=PLOT_DIR)
    plot_kalman_covariance_vs_time(results, outdir=PLOT_DIR)
    plot_orientation_vs_time(results, outdir=PLOT_DIR)

    # ------------------------------------------------------------------
    # Multiple initial conditions plots (hover altitude = 10 m)
    # ------------------------------------------------------------------
    Tfinal_ic = 20.0
    h_ref_ic = 10.0

    # 1) Start on ground at 0 m
    x0_ground = X0.copy()
    x0_ground[2] = 0.0  # p_d = 0 -> h = 0

    # 2) Start above reference (15 m)
    x0_above = X0.copy()
    x0_above[2] = -15.0  # p_d = -h

    # 3) Start below reference (2 m)
    x0_below = X0.copy()
    x0_below[2] = -2.0  # h = 2 m, reference = 10 m

    ic_scenarios = [
        ("start_ground_0m", x0_ground),
        ("start_above_15m", x0_above),
        ("start_below_2m", x0_below),
    ]

    ic_results: list[dict] = []
    for label, x0_ic in ic_scenarios:
        res_ic = run_pid_altitude_hold(
            P_nominal=P,
            x0=x0_ic,
            Tfinal=Tfinal_ic,
            dt=dt,
            h_ref=h_ref_ic,
            pid_gains=gains,
            use_kf=True,
            baro_sigma=baro_sigma,
            baro_bias=baro_bias,
            baro_seed=baro_seed,
            dyn_mismatch=None,
            dn_fixed_deg=90.0,
        )
        res_ic["label"] = label
        ic_results.append(res_ic)

    from plotting import plot_altitude_ic_scenarios
    plot_altitude_ic_scenarios(ic_results, h_ref_ic, outdir=PLOT_DIR)

    # ------------------------------------------------------------------
    # Parameter mismatch plots
    # ------------------------------------------------------------------
    Tfinal_mm = 20.0
    x0_mm = x0_ground.copy()

    mismatch_cases = [
        ("nominal", {}),
        ("mass+30%", {"mass_mult": 1.3}),
        ("mass-30%", {"mass_mult": 0.7}),
        ("kT-20%", {"kT_hover_mult": 0.8}),
    ]

    mm_results: list[dict] = []
    for label, mm in mismatch_cases:
        res_mm = run_pid_altitude_hold(
            P_nominal=P,
            x0=x0_mm,
            Tfinal=Tfinal_mm,
            dt=dt,
            h_ref=h_ref_ic,
            pid_gains=gains,
            use_kf=True,
            baro_sigma=baro_sigma,
            baro_bias=baro_bias,
            baro_seed=baro_seed,
            dyn_mismatch=mm if mm else None,
            dn_fixed_deg=90.0,
        )
        res_mm["label"] = label
        mm_results.append(res_mm)

    from plotting import plot_altitude_mismatch_scenarios
    plot_altitude_mismatch_scenarios(mm_results, h_ref_ic, outdir=PLOT_DIR)

    print(f"Done. Plots saved to ./{PLOT_DIR}/")


if __name__ == "__main__":
    main()
