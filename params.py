from typing import Dict, Any, List

PLOT_DIR = "plots"

# Parameters
Params: Dict[str, Any] = {
    "env": {
        "gravity": 9.80665,      # m/s^2
        "air_density": 1.225,    # kg/m^3
    },
    "massprops": {
        "m": 22500.0,       # kg
        "Ixx": 8000.0,      # kg·m^2
        "Iyy": 30000.0,     # kg·m^2
        "Izz": 35000.0,     # kg·m^2
    },
    "geometry": {
        "S": 28.0,          # m^2
        "b": 25.5,          # m
        "cbar": 28.0 / 25.5 # m
    },

    # linear body-axis drag (N per (m/s))
    "drag_body": {"Dx": 50.0, "Dy": 60.0, "Dz": 80.0},

    # Nacelle angle constraints (deg) and rate limit (deg/s)
    "nacelle": {"dn_min": 0.0, "dn_max": 90.0, "dn_rate_max_degps": 10.0},

    # Rotor effectiveness: collective -> thrust (N per unit collective)
    "rotor_eff": {"kT_hover": None, "kT_airplane_scale": 0.25},

    # Effective cyclic/pedal -> hub moments (N·m), scheduled by nacelle angle
    "moments_eff": {
        "k_lat_hover": 2.0e4, "k_lon_hover": 2.0e4, "k_ped_hover": 1.5e4,
        "k_lat_airplane": 1.2e4, "k_lon_airplane": 1.2e4, "k_ped_airplane": 1.0e4
    },

    # longitudinal aero (lift/drag + pitch moment)
    "aero": {
        "CLalpha": 5.5,        # per rad
        "alpha0_deg": -2.0,    # deg (zero-lift AoA)
        "CLde": 0.2,           # per rad
        "CD0": 0.05,           # parasitic drag
        "e": 0.80,             # Oswald efficiency
        "Cm0": 0.0,
        "Cma": -0.8,           # per rad
        "Cmq": -12.0,          # per rad (q-hat)
        "Cmde": -1.0,          # elevator-to-moment
        "ke_lon_to_elev": 0.6  # map d_lon -> delta_e (rad)
    },

    # Small roll/yaw damping
    "damping": {"Lp": -5000.0, "Nr": -6000.0},

    # Simulation horizon and step size
    "sim": {"dt": 0.01, "Tfinal": 20.0}
}

# Initial state
# [pn, pe, pd,  u, v, w,  phi, theta, psi,  p, q, r]
#  altitude = -pd => altitude = +Xm
X0: List[float] = [0.0, 0.0, -0.5,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0]
