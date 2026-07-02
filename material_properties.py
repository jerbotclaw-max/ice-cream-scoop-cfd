"""Material thermal properties for ice cream scoop CFD simulation."""

# Thermal conductivity [W/(m·K)]
K_ALUMINUM = 205.0
K_COPPER = 401.0
K_OIL_EFFECTIVE = 2.0  # Mineral oil with natural convection (Ra-based correlation)

# Density [kg/m³]
RHO_ALUMINUM = 2700.0
RHO_COPPER = 8960.0
RHO_OIL = 880.0

# Specific heat capacity [J/(kg·K)]
CP_ALUMINUM = 900.0
CP_COPPER = 385.0
CP_OIL = 1900.0

# Derived thermal diffusivity α = k / (ρ * cp) [m²/s]
ALPHA_ALUMINUM = K_ALUMINUM / (RHO_ALUMINUM * CP_ALUMINUM)
ALPHA_COPPER = K_COPPER / (RHO_COPPER * CP_COPPER)
ALPHA_OIL = K_OIL_EFFECTIVE / (RHO_OIL * CP_OIL)

# Boundary conditions
T_HAND = 33.0       # °C (hand surface temp)
T_ICE_CREAM = -18.0  # °C (freezer temp)
T_AMBIENT = 22.0     # °C (room temp)
H_AIR = 10.0         # W/(m²·K) (natural convection to air)

# Geometry [m]
HANDLE_OD = 0.020       # 20 mm outer diameter
HANDLE_ID = 0.018       # 18 mm inner diameter (for oil-filled tube)
HANDLE_LENGTH = 0.150   # 150 mm
GRIP_LENGTH = 0.100     # 100 mm grip zone
SCOOP_DIAMETER = 0.050  # 50 mm
SCOOP_WALL = 0.002      # 2 mm
