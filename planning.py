from typing import Tuple
import numpy as np
from inputs import _smoothstep


def generate_reference(t: np.ndarray, n0: float = 0.0, e0: float = 0.0, h0: float = 0.0, hover_alt: float = 10.0, cruise_alt: float = 20.0, east_target: float = 300.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate smooth reference trajectories for North, East, and Altitude (up-positive).

    Timeline (s):
      0–3   : takeoff, climb to hover_alt
      3–6   : hold hover_alt
      6–12  : climb to cruise_alt and accelerate East
      12–16 : hold cruise_alt, continue East
      16–20 : decelerate to stop East motion and descend to 0 m (landing)

    :param t: Time stamps (s).
    :type t: numpy.ndarray
    :param n0: Initial North reference (m).
    :type n0: float
    :param e0: Initial East reference (m).
    :type e0: float
    :param h0: Initial altitude reference (m).
    :type h0: float
    :param hover_alt: Hover altitude (m).
    :type hover_alt: float
    :param cruise_alt: Cruise altitude (m).
    :type cruise_alt: float
    :param east_target: Final East position at 16 s before reconversion (m).
    :type east_target: float

    :returns: (n_ref, e_ref, h_ref) where
              * n_ref is the North reference trajectory (m),
              * e_ref is the East reference trajectory (m),
              * h_ref is the altitude reference trajectory (m, up-positive).
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    n_ref = np.full_like(t, n0, dtype=float)

    # Altitude profile
    h_ref = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti < 3.0:
            h_ref[i] = _smoothstep(ti, 0.0, 3.0, h0, hover_alt)
        elif ti < 6.0:
            h_ref[i] = hover_alt
        elif ti < 12.0:
            h_ref[i] = _smoothstep(ti, 6.0, 12.0, hover_alt, cruise_alt)
        elif ti < 16.0:
            h_ref[i] = cruise_alt
        else:
            h_ref[i] = _smoothstep(ti, 16.0, 20.0, cruise_alt, 0.0)

    # East profile
    e_ref = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti < 6.0:
            e_ref[i] = e0
        elif ti < 12.0:
            e_ref[i] = _smoothstep(ti, 6.0, 12.0, e0, east_target * 0.6)
        elif ti < 16.0:
            e_ref[i] = _smoothstep(ti, 12.0, 16.0, east_target * 0.6, east_target)
        else:
            e_ref[i] = _smoothstep(ti, 16.0, 20.0, east_target, east_target)  # hold

    return n_ref, e_ref, h_ref
