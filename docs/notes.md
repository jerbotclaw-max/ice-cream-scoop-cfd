# Engineering Notes

## Why Oil-Filled Aluminum Beats Copper

### The heat equation constraint

The key metric is the characteristic thermal time constant:

```
τ = ρ · cp · V / (k · A)
```

For a long cylinder (handle):
- τ ∝ (ρ·cp) · L²/k

This means:
- **Low ρ·cp** (density × heat capacity) → fast warmup
- **High k** (thermal conductivity) → fast heat propagation

### Copper vs Aluminum

| Property | Copper | Aluminum | Ratio |
|----------|--------|----------|-------|
| k (W/m·K) | 401 | 205 | 1.95× |
| ρ (kg/m³) | 8960 | 2700 | 3.32× |
| cp (J/kg·K) | 385 | 900 | 0.43× |
| ρ·cp (kJ/m³·K) | 3450 | 2430 | 1.42× |

Copper has 2× the conductivity but 1.4× the volumetric heat capacity. For this geometry (long thin handle), the ρ·cp penalty dominates over the k bonus.

### Oil Column Effect

The mineral oil in the Zeroll design does something clever:
1. The aluminum tube wall has very low thermal mass
2. Heat flows into the oil from the hand side
3. Oil circulates (natural convection) carrying heat toward the cold end
4. Heat transfers through the thin aluminum wall into the scoop head

Without the oil (solid aluminum tube), the same wall thickness gives:
- R_tube = ln(r_o/r_i) / (2π·k_al·L)
- Higher conductivity, but the wall cross-section is small

The oil effectively increases the effective cross-section for heat transport while keeping thermal mass low.

## Natural Convection in Oil

Rayleigh number for the oil column (r_i = 8mm, ΔT = 50K):

```
Ra = g · β · ΔT · D³ / (ν · α)
  = 9.81 × 7e-4 × 50 × (0.016)³ / (1e-4 × 8.5e-8)
  ≈ 1.2 × 10⁴
```

For Ra < 10⁹ (laminar natural convection in enclosed layer):
```
Nu = 0.42 · Ra^(1/4) ≈ 0.42 × 11.4 ≈ 4.8
```

Effective oil k = Nu × k_static = 4.8 × 0.15 ≈ 0.72 W/m·K

So the oil column conducts like an insulating material in series with a good conductor in parallel with the aluminum wall — overall, still limiting.

## Future Improvements

1. **Full 3D CFD** with actual oil convection cells (GPU needed)
2. **Phase change** in ice cream (latent heat of fusion ~200 kJ/kg)
3. **Ice cream melting layer** — thin liquid film at interface
4. **Transient contact resistance** between hand and handle
5. **Thermal sensation modeling** — at what point does the handle "feel warm"?
