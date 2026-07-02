"""
Material thermal properties for ice cream scoop simulation.
All values in SI units.
"""

MATERIALS = {
    "aluminum": {
        "k": 205.0,       # Thermal conductivity [W/m·K]
        "rho": 2700.0,    # Density [kg/m³]
        "cp": 900.0,      # Specific heat [J/kg·K]
        "alpha": 205.0 / (2700.0 * 900.0),  # Thermal diffusivity [m²/s]
    },
    "copper": {
        "k": 401.0,
        "rho": 8960.0,
        "cp": 385.0,
        "alpha": 401.0 / (8960.0 * 385.0),
    },
    "mineral_oil": {
        "k": 0.15,        # Pure conduction
        "rho": 850.0,
        "cp": 2100.0,
        "alpha": 0.15 / (850.0 * 2100.0),
        # Natural convection parameters
        "beta": 7.0e-4,   # Thermal expansion coefficient [1/K]
        "nu": 1.0e-4,     # Kinematic viscosity [m²/s] (typical for light oil)
    },
    "ice_cream": {
        "k": 0.55,        # Frozen ice cream ~0.5-0.6
        "rho": 550.0,     # Half ice, half air/cream
        "cp": 2100.0,
        "alpha": 0.55 / (550.0 * 2100.0),
    },
}


def oil_effective_k(T_hot, T_cold, L_char, mat="mineral_oil"):
    """
    Compute effective thermal conductivity of mineral oil in a vertical tube
    accounting for natural convection.

    Uses Rayleigh number for enclosed vertical cavity:
        Ra = g * beta * dT * L³ / (nu * alpha)

    Nusselt correlation for vertical enclosed layer (Incropera):
        Nu = 0.42 * Ra^(1/4) * Pr^(0.012)  for Ra < 10^9
        Nu = 0.046 * Ra^(1/3)               for Ra > 10^9

    Returns effective k = Nu * k_pure
    """
    oil = MATERIALS[mat]
    g = 9.81
    dT = abs(T_hot - T_cold)
    if dT < 0.01:
        return oil["k"]

    alpha = oil["alpha"]
    nu = oil["nu"]
    beta = oil["beta"]

    Ra = g * beta * dT * L_char**3 / (nu * alpha)
    Pr = nu / alpha

    if Ra < 1e9:
        Nu = 0.42 * Ra**0.25 * Pr**0.012
    else:
        Nu = 0.046 * Ra ** (1.0 / 3.0)

    k_eff = Nu * oil["k"]
    return k_eff


if __name__ == "__main__":
    # Quick test
    T_hand = 33.0 + 273.15
    T_cold = -18.0 + 273.15
    L = 0.15  # handle length
    k_eff = oil_effective_k(T_hand, T_cold, L)
    print(f"Mineral oil effective k: {k_eff:.3f} W/m·K")
    print(f"  (pure: {MATERIALS['mineral_oil']['k']} W/m·K)")
    print(f"  Ratio: {k_eff / MATERIALS['mineral_oil']['k']:.1f}x")
