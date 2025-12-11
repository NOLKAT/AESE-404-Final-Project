from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def baro_measurement(h_true: float, rng: np.random.Generator, sigma: float, bias: float) -> float:
    """Barometric altitude measurement: z = h + bias + N(0, sigma^2)."""
    return float(h_true + bias + rng.normal(0.0, sigma))


def simulate_baro(t: np.ndarray, X: np.ndarray, sigma: float, bias: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate barometric measurements for a given trajectory.

    Parameters
    ----------
    t : (N,)
        Time vector.
    X : (N,12)
        State trajectory (down-positive altitude in X[:,2]).
    sigma : float
        Measurement noise std (m).
    bias : float
        Constant bias (m).
    seed : int or None
        RNG seed for repeatability.

    Returns
    -------
    z : (N,)
        Baro measurements.
    h_true : (N,)
        True altitude (up-positive) for reference.
    """
    rng = np.random.default_rng(seed)
    N = len(t)
    z = np.zeros(N)
    h_true = -X[:, 2].copy()

    for i in range(N):
        z[i] = baro_measurement(h_true[i], rng, sigma, bias)

    return z, h_true
