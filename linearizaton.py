from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
from dynamics import f_rhs, kT
from params import Params as P


def _const_ufun(u_const: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Wrap a constant control vector into a time/state-dependent input function.

    This helper converts a constant input vector ``u_const`` into a callable
    with signature ``u(t, x)`` so that it can be passed to functions expecting
    a control schedule of the form ``ufun(t, x)``.

    :param u_const: Constant control vector (e.g. ``[d_col, d_lon, d_lat, d_ped, dn]``).
    :type u_const: numpy.ndarray

    :returns: Control schedule function ``u(t, x)`` that always returns ``u_const``.
    :rtype: Callable[[float, numpy.ndarray], numpy.ndarray]
    """
    u_const = np.array(u_const, dtype=float)
    return lambda t, x: u_const


def numerical_jacobian(x0: np.ndarray, u0: np.ndarray, P: dict, eps_x: float = 1e-6, eps_u: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute central-difference Jacobians ``A = ∂f/∂x`` and ``B = ∂f/∂u`` at ``(x0, u0)``.

    The nonlinear dynamics are given by

        x_dot = f_rhs(t, x, ufun, P),

    where ``ufun`` is taken to be constant at ``u0``. The Jacobians are computed
    using second-order central finite differences around the operating point
    ``(x0, u0)``.

    :param x0: State vector at which the Jacobians are evaluated.
    :type x0: numpy.ndarray
    :param u0: Control vector at which the Jacobians are evaluated.
    :type u0: numpy.ndarray
    :param P: Parameter dictionary passed through to :func:`dynamics.f_rhs`.
    :type P: dict
    :param eps_x: Perturbation size for state derivatives (central difference step).
                  Default is ``1e-6``.
    :type eps_x: float, optional
    :param eps_u: Perturbation size for input derivatives (central difference step).
                  Default is ``1e-6``.
    :type eps_u: float, optional

    :returns: ``(A, B)`` where
              * ``A`` is the state Jacobian ``∂f/∂x`` evaluated at ``(x0, u0)``,
                shape ``(n, n)``,
              * ``B`` is the input Jacobian ``∂f/∂u`` evaluated at ``(x0, u0)``,
                shape ``(n, m)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    x0 = np.array(x0, dtype=float).reshape(-1)
    u0 = np.array(u0, dtype=float).reshape(-1)

    n = x0.size
    m = u0.size
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    rhs = lambda xx, uu: f_rhs(0.0, xx, _const_ufun(uu), P)
    _ = rhs(x0, u0)

    # State Jacobian: d/dx
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps_x
        fp = rhs(x0 + dx, u0)
        fm = rhs(x0 - dx, u0)
        A[:, i] = (fp - fm) / (2.0 * eps_x)

    # Input Jacobian: d/du
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps_u
        fp = rhs(x0, u0 + du)
        fm = rhs(x0, u0 - du)
        B[:, j] = (fp - fm) / (2.0 * eps_u)

    return A, B


def c2d(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize a continuous-time linear system using forward Euler

    Given the continuous-time state-space model

        x_dot = A x + B u,

    this function computes a discrete-time approximation with sample period
    ``dt``:

        x[k+1] = A_d x[k] + B_d u[k],

    using the forward-Euler method

        A_d = I + A dt,
        B_d = B dt.

    :param A: Continuous-time state matrix ``A``, shape ``(n, n)``.
    :type A: numpy.ndarray
    :param B: Continuous-time input matrix ``B``, shape ``(n, m)``.
    :type B: numpy.ndarray
    :param dt: Sampling period (s).
    :type dt: float

    :returns: ``(A_d, B_d)`` where
              * ``A_d`` is the discrete-time state-transition matrix, shape ``(n, n)``,
              * ``B_d`` is the discrete-time input matrix, shape ``(n, m)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    n = A.shape[0]
    Ad = np.eye(n) + A * dt
    Bd = B * dt
    return Ad, Bd


def hover_equilibrium(P: dict, h_ref_m: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a hover equilibrium operating point at a given altitude.

    The equilibrium corresponds to:

      * Near-zero attitude and angular rates,
      * NED position with altitude ``h_ref_m`` (up-positive),
      * Nacelles fixed at 90 degrees (pure hover),
      * Collective set so that rotor thrust approximately balances weight.

    The state is defined as

        x = [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]^T,

    and the control input as

        u = [d_col, d_lon, d_lat, d_ped, dn]^T.

    :param P: Parameter dictionary containing mass properties, environment,
              and rotor efficiency data.
    :type P: dict
    :param h_ref_m: Desired hover altitude (m, up-positive).
                    Default is ``10.0``.
    :type h_ref_m: float, optional

    :returns: ``(x_eq, u_eq)`` where
              * ``x_eq`` is the 12-state equilibrium vector at hover (ndarray, shape (12,)),
              * ``u_eq`` is the 5-element equilibrium input
                ``[d_col, d_lon, d_lat, d_ped, dn]`` (ndarray, shape (5,)).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    x_eq = np.zeros(12, dtype=float)

    # NED position: altitude up = -p_d
    x_eq[2] = -h_ref_m  # p_d (down positive)

    dn_deg = 90.0
    m = P["massprops"]["m"]
    g = P["env"]["gravity"]
    d_col_eq = (m * g) / max(1e-9, kT(P, dn_deg))

    u_eq = np.array([d_col_eq, 0.0, 0.0, 0.0, dn_deg], dtype=float)
    return x_eq, u_eq


def linearization() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearize the nonlinear vehicle dynamics about a hover equilibrium at 10 m altitude.

    This routine:
      * Computes the hover equilibrium state ``x_eq`` and input ``u_eq`` at
        ``h_ref_m = 10.0`` using :func:`hover_equilibrium`.
      * Computes the continuous-time Jacobians ``A = ∂f/∂x`` and ``B = ∂f/∂u``
        at ``(x_eq, u_eq)`` using central finite differences via
        :func:`numerical_jacobian`.
      * Prints the hover input vector and the eigenvalues of ``A`` for
        inspection of local stability characteristics.

    :returns: ``(x_eq, u_eq, A, B)`` where
              * ``x_eq`` is the 12-state equilibrium vector at hover (ndarray, shape (12,)),
              * ``u_eq`` is the 5-element equilibrium control input
                ``[d_col, d_lon, d_lat, d_ped, dn]`` (ndarray, shape (5,)),
              * ``A`` is the state Jacobian ``∂f/∂x`` evaluated at ``(x_eq, u_eq)``
                (ndarray, shape (12, 12)),
              * ``B`` is the input Jacobian ``∂f/∂u`` evaluated at ``(x_eq, u_eq)``
                (ndarray, shape (12, 5)).
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    print("Linearizing about a hover operating point at h = 10 m ...")
    x_eq, u_eq = hover_equilibrium(P, h_ref_m=10.0)
    A, B = numerical_jacobian(x_eq, u_eq, P)
    eigA = np.linalg.eigvals(A)

    print("Hover equilibrium input u_eq = [d_col, d_lon, d_lat, d_ped, dn]:")
    print("  ", u_eq)
    print("Eigenvalues of A:")
    print("  ", np.round(eigA, 4))

    return x_eq, u_eq, A, B
