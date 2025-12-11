import numpy as np


def clamp(v: float, lo: float, hi: float) -> float:
    """
    Clamp a scalar to a closed interval [lo, hi]

    :param v: Value to clamp.
    :type v: float
    :param lo: Lower bound.
    :type lo: float
    :param hi: Upper bound.
    :type hi: float
    :returns: The clamped value.
    :rtype: float
    """
    return max(lo, min(hi, v))


def Rb2n(phi: float, th: float, psi: float) -> np.ndarray:
    """
    Compute the body-to-NED direction-cosine matrix (DCM) using ZYX
    (yaw–pitch–roll) Euler angles

    :param phi: Roll angle
    :type phi: float
    :param th: Pitch angle
    :type th: float
    :param psi: Yaw angle
    :type psi: float
    :returns: Rotation matrix from body to NED
    :rtype: numpy.ndarray (3, 3)
    """
    c, s = np.cos, np.sin
    cphi, cth, cpsi = c(phi), c(th), c(psi)
    sphi, sth, spsi = s(phi), s(th), s(psi)
    return np.array([
        [cth * cpsi,                      cth * spsi,                     -sth],
        [sphi * sth * cpsi - cphi * spsi, sphi * sth * spsi + cphi * cpsi, sphi * cth],
        [cphi * sth * cpsi + sphi * spsi, cphi * sth * spsi - sphi * cpsi, cphi * cth],
    ])


def T_euler(phi: float, th: float) -> np.ndarray:
    """
    Map body angular rates (p, q, r) to Euler angle rates (phi_dot, theta_dot, psi_dot)

    :param phi: Roll angle
    :type phi: float
    :param th: Pitch angle
    :type th: float
    :returns: Transformation matrix T such that [phi_dot, theta_dot, psi_dot]^T = T @ [p, q, r]^T
    :rtype: numpy.ndarray (3, 3)
    """
    c, s = np.cos, np.sin
    return np.array([
        [1.0, s(phi) * np.tan(th),  c(phi) * np.tan(th)],
        [0.0, c(phi),              -s(phi)],
        [0.0, s(phi) / c(th),       c(phi) / c(th)],
    ])


def sched01_from_dn_deg(dn_deg: float) -> float:
    """
    Compute a linear schedule in [0, 1] from nacelle angle (degrees),
    where 0 corresponds to hover (90°) and 1 to airplane mode (0°)

    :param dn_deg: Nacelle angle in degrees
    :type dn_deg: float
    :returns: Schedule value in [0, 1]
    :rtype: float
    """
    dn = np.clip(dn_deg, 0.0, 90.0)
    return (90.0 - dn) / 90.0


def kT(P: dict, dn_deg: float) -> float:
    """
    Compute the thrust-per-collective gain scheduled by nacelle angle

    If P["rotor_eff"]["kT_hover"] is None, it is computed on-the-fly so
    that a collective of approximately 0.6` balances weight at
    dn = 90° (hover). The airplane-mode gain is scaled by P["rotor_eff"]["kT_airplane_scale"]

    :param P: Parameter dictionary
    :type P: dict
    :param dn_deg: Nacelle angle in degrees
    :type dn_deg: float
    :returns: Thrust gain (N per unit collective)
    :rtype: float
    """
    kT_hover = P["rotor_eff"]["kT_hover"]
    if kT_hover is None:
        m = P["massprops"]["m"]
        g = P["env"]["gravity"]
        kT_hover = (m * g) / 0.6

    kT_air = P["rotor_eff"]["kT_airplane_scale"] * kT_hover
    s = sched01_from_dn_deg(dn_deg)
    return (1.0 - s) * kT_hover + s * kT_air


def k_eff(dn_deg: float, k_hover: float, k_air: float) -> float:
    """
    Interpolate an effectiveness gain between hover (90°) and
    airplane mode (0°)

    :param dn_deg: Nacelle angle in degrees
    :type dn_deg: float
    :param k_hover: Gain at 90°
    :type k_hover: float
    :param k_air: Gain at 0°
    :type k_air: float
    :returns: Interpolated gain
    :rtype: float
    """
    s = sched01_from_dn_deg(dn_deg)
    return (1.0 - s) * k_hover + s * k_air


def aero_longitudinal_forces_moments(u: float, w: float, q: float, P: dict, delta_e: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute minimal longitudinal wing/tail aerodynamics

    :param u: Body-forward velocity (m/s)
    :type u: float
    :param w: Body-down velocity (m/s)
    :type w: float
    :param q: Pitch rate (rad/s)
    :type q: float
    :param P: Parameter dictionary (uses env, geometry, aero)
    :type P: dict
    :param delta_e: Elevator-like deflection (rad)
    :type delta_e: float
    :returns:
        * Fa_b: Body forces [X, Y, Z] (N)
        * Ma_b: Body moments [L, M, N] (N·m), only M is nonzero
        * V: Airspeed magnitude (m/s)
        * alpha: Angle of attack (rad)
    :rtype: tuple[numpy.ndarray, numpy.ndarray, float, float]
    """
    rho = P["env"]["air_density"]
    S = P["geometry"]["S"]
    b = P["geometry"]["b"]
    cbar = P["geometry"]["cbar"]
    cf = P["aero"]

    V = np.sqrt(max(u * u + w * w, 1e-8))
    alpha = np.arctan2(w, u)

    CLalpha = cf["CLalpha"]
    CLde = cf["CLde"]
    alpha0 = np.deg2rad(cf["alpha0_deg"])
    CD0 = cf["CD0"]
    e = cf["e"]
    AR = b * b / S
    k_ind = 1.0 / (np.pi * e * AR)
    Cm0, Cma, Cmq, Cmde = cf["Cm0"], cf["Cma"], cf["Cmq"], cf["Cmde"]

    qbar = 0.5 * rho * V * V
    CL = CLalpha * (alpha - alpha0) + CLde * delta_e
    CD = CD0 + k_ind * (CL * CL)
    L = qbar * S * CL
    D = qbar * S * CD

    ca, sa = np.cos(alpha), np.sin(alpha)
    Xb = -D * ca + L * sa
    Zb = -D * sa - L * ca

    qhat = q * cbar / max(2.0 * V, 1e-6)
    Cm = Cm0 + Cma * alpha + Cmq * qhat + Cmde * delta_e
    My = qbar * S * cbar * Cm

    Fa_b = np.array([Xb, 0.0, Zb])
    Ma_b = np.array([0.0, My, 0.0])
    return Fa_b, Ma_b, V, alpha


def body_drag(vb: np.ndarray, Dx: float, Dy: float, Dz: float) -> np.ndarray:
    """
    Compute linear body-axis aerodynamic drag

    :param vb: Body-axis velocity vector [u, v, w] (m/s).
    :type vb: numpy.ndarray
    :param Dx: Drag coefficient on body X (N·s/m).
    :type Dx: float
    :param Dy: Drag coefficient on body Y (N·s/m).
    :type Dy: float
    :param Dz: Drag coefficient on body Z (N·s/m).
    :type Dz: float
    :returns: Body-axis drag force (N)
    :rtype: numpy.ndarray (3,)
    """
    return -np.array([Dx * vb[0], Dy * vb[1], Dz * vb[2]])


def forces_moments_total(x: np.ndarray, u_c: np.ndarray, P: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute total body forces and moments for the current state and inputs

    :param x: State vector: [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    :type x: numpy.ndarray
    :param u_c: Control/input vector: [d_col, d_lon, d_lat, d_ped, dn_deg]
    :type u_c: numpy.ndarray
    :param P: Parameter dictionary
    :type P: dict
    :returns:
        * Fb: Total body force (N)
        * Mb: Total body moment (N·m)
        * I: Diagonal inertia matrix (kg·m²)
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    pn, pe, pd, u, v, w, phi, th, psi, pr, qr, rr = x
    vb = np.array([u, v, w])

    dcol, dlon, dlat, dped, dn = u_c

    # Rotor thrust
    Tmag = kT(P, dn) * dcol
    cdn, sdn = np.cos(np.deg2rad(dn)), np.sin(np.deg2rad(dn))
    Fr_b = np.array([Tmag * cdn, 0.0, -Tmag * sdn])

    # Effective cyclic/pedal -> moments
    k_lat = k_eff(dn, P["moments_eff"]["k_lat_hover"], P["moments_eff"]["k_lat_airplane"])
    k_lon = k_eff(dn, P["moments_eff"]["k_lon_hover"], P["moments_eff"]["k_lon_airplane"])
    k_ped = k_eff(dn, P["moments_eff"]["k_ped_hover"], P["moments_eff"]["k_ped_airplane"])
    Mr_b = np.array([k_lat * dlat, k_lon * dlon, k_ped * dped])

    # Body drag
    Fd_b = body_drag(vb, **P["drag_body"])

    # Wing/tail
    delta_e = P["aero"]["ke_lon_to_elev"] * dlon
    Fa_b, Ma_b, _, _ = aero_longitudinal_forces_moments(u, w, qr, P, delta_e)

    #  roll/yaw damping
    Mdamp_b = np.array([P["damping"]["Lp"] * pr, 0.0, P["damping"]["Nr"] * rr])

    # Sum forces & moments
    Fb = Fr_b + Fd_b + Fa_b
    Mb = Mr_b + Ma_b + Mdamp_b

    # Diagonal inertia
    I = np.diag([P["massprops"]["Ixx"], P["massprops"]["Iyy"], P["massprops"]["Izz"]])
    return Fb, Mb, I


def f_rhs(t: float, x: np.ndarray, ufun, P: dict) -> np.ndarray:
    """
    Compute the continuous-time state derivative for the 6-DOF rigid-body model

    :param t: Time (s)
    :type t: float
    :param x: State vector: [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    :type x: numpy.ndarray
    :param ufun: Input schedule function: u(t, x) -> [d_col, d_lon, d_lat, d_ped, dn_deg]
    :type ufun: Callable
    :param P: Parameter dictionary.
    :type P: dict
    :returns: Time derivative of the state
    :rtype: numpy.ndarray
    """
    pn, pe, pd, u, v, w, phi, th, psi, pr, qr, rr = x
    vb = np.array([u, v, w])
    omg = np.array([pr, qr, rr])

    u_c = ufun(t, x)
    Fb, Mb, I = forces_moments_total(x, u_c, P)

    # Gravity in body axes
    gb = Rb2n(phi, th, psi).T @ np.array([0.0, 0.0, P["env"]["gravity"]])

    # Translational dynamics
    dvb = (1.0 / P["massprops"]["m"]) * Fb + gb - np.cross(omg, vb)

    # Rotational dynamics
    Iomg = I @ omg
    domg = np.linalg.solve(I, -np.cross(omg, Iomg) + Mb)

    # Position kinematics (NED)
    pndot, pedot, pddot = (Rb2n(phi, th, psi) @ vb)

    # Euler-angle kinematics
    etadot = T_euler(phi, th) @ omg

    return np.hstack([pndot, pedot, pddot, dvb, etadot, domg])


def enforce_ground_plane(x: np.ndarray) -> np.ndarray:
    """
    Enforce a flat ground plane (altitude >= 0) to ensure the plane will not go underground while descending or landing

    If pd > 0 (below ground), pd is assigned to 0 and the downward NED velocity is removed.

    :param x: State vector: [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    :type x: numpy.ndarray
    :returns: Adjusted state that satisfies the ground constraint.
    :rtype: numpy.ndarray
    """
    pn, pe, pd, u, v, w, phi, th, psi, pr, qr, rr = x
    if pd <= 0.0:
        return x

    pd = 0.0

    R = Rb2n(phi, th, psi)
    vn, ve, vd = R @ np.array([u, v, w])
    if vd > 0.0:
        vd = 0.0
        u, v, w = R.T @ np.array([vn, ve, vd])

    return np.array([pn, pe, pd, u, v, w, phi, th, psi, pr, qr, rr], dtype=float)


def rk4_step(fun, t: float, x: np.ndarray, dt: float, ufun, P: dict) -> np.ndarray:
    """
    Advance one step using the fourth-order Runge–Kutta method.

    :param fun: Right-hand side function f(t, x, ufun, P) -> xdot
    :type fun: Callable
    :param t: Current time (s)
    :type t: float
    :param x: Current state
    :type x: numpy.ndarray
    :param dt: Time step (s)
    :type dt: float
    :param ufun: Input schedule function
    :type ufun: Callable
    :param P: Parameter dictionary
    :type P: dict
    :returns: State advanced by one RK4 step of size dt
    :rtype: numpy.ndarray
    """
    k1 = fun(t, x, ufun, P)
    k2 = fun(t + dt / 2.0, x + dt * k1 / 2.0, ufun, P)
    k3 = fun(t + dt / 2.0, x + dt * k2 / 2.0, ufun, P)
    k4 = fun(t + dt, x + dt * k3, ufun, P)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
