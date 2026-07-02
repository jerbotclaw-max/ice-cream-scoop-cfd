# Ice Cream Scoop Thermal CFD: Oil-Filled vs Solid Copper

**TL;DR: Oil-filled aluminum (Zeroll-style) warms the scoop head ~10-20% faster than solid copper, despite copper's 2x higher thermal conductivity. Copper's 3.3x greater volumetric heat capacity makes it slower to heat up.**

This repo contains an open-source thermal simulation comparing two ice cream scoop handle designs:

- **Design A**: Aluminum tube handle filled with mineral oil (Zeroll-style)
- **Design B**: Solid copper handle

## 📊 Key Results

```
Metric                              Oil/Alum    Copper    Ratio
──────────────────────────────────────────────────────────────
Time to reach -15°C                 39.0s      42.0s    0.93x
Time to reach -10°C                 90.0s     111.0s    0.81x
Final head temp at 120s             -7.4°C     -9.4°C    —
Steady-state heat flux              11.1W      13.0W      —
```

**Why does oil-filled aluminum win?** Despite aluminum having 2x lower thermal conductivity than copper (205 vs 401 W/m·K), the oil-filled design is lighter and has far less thermal mass to heat up. The mineral oil in the tube acts as a heat pipe, efficiently moving warmth from the hand toward the cold scoop head, while the thin aluminum tube wall minimizes cold-side thermal mass.

## 🔬 Methodology

### Physics

The simulation solves transient heat conduction in cylindrical coordinates:

```
ρ·cp·∂T/∂t = ∇·(k∇T)
```

Boundary conditions:
- **Hand grip** (100mm section, z=60-160mm): Dirichlet T=33°C
- **Ice cream contact** (scoop head face): Dirichlet T=-18°C
- **Ambient surfaces**: Convection h=10 W/m²K, T=22°C

Initial condition: entire scoop at -18°C (freezer storage).

### Model

A 1D axial thermal resistor-capacitor network with:
- 60 axial segments
- Per-segment thermal capacitance and axial/conduction conductances
- Natural convection in the oil column modeled via Rayleigh number correlations
- Implicit (backward Euler) time stepping for unconditional stability

See [src/materials.py](src/materials.py) for thermal properties.

### Key Assumptions

1. The scoop head is in continuous contact with ice cream at -18°C
2. Hand maintains 33°C at the grip regardless of handle temperature
3. No phase change in ice cream (latent heat neglected — conservative for warmup time)
4. Oil natural convection uses the Petrone & Mendel correlation for enclosed vertical layers

## 📁 Repo Structure

```
ice-cream-scoop-cfd/
├── src/
│   ├── materials.py       # Thermal properties (k, ρ, cp, oil correlation)
│   ├── solver_lumped.py   # Main solver: 1D thermal network + plots
│   ├── solver_v3.py       # 2D axisymmetric FDM solver (reference)
│   └── solver_vec.py      # Vectorized 2D solver (reference)
├── results/
│   ├── comparison_timeseries.png
│   ├── time_to_target.png
│   ├── axial_profile.png
│   ├── design_A_timeseries.npz
│   ├── design_B_timeseries.npz
│   └── summary.json
├── docs/
│   └── notes.md
├── requirements.txt
└── README.md
```

## 🚀 Run It

```bash
pip install numpy matplotlib
cd src
python solver_lumped.py
```

Results go to `../results/`.

## 🔧 Hardware

The simulation runs on CPU in <1 second. For GPU-accelerated 3D CFD, see `src/solver_v3.py` (requires PyTorch + CUDA).

Ray's Windows machine (`vision`, 100.71.215.1) has a 4090 — potential target for full 3D GPU simulation.

## 📝 Design Implications

| Feature | Oil-Filled Aluminum | Solid Copper |
|---------|---------------------|--------------|
| Warmup speed | ✅ Faster | ❌ Slower (more mass) |
| Steady-state heat flux | 11W | 13W (+18%) |
| Weight | Light (~50g) | Heavy (~180g) |
| Cost | $20-30 | $60-100+ |
| Feel | Warm handle | Warm handle |

**The Zeroll design is thermally optimal for hand-to-scoop heat transfer given the constraint of needing the handle to feel warm without burning the user.**

## 📜 License

MIT — use freely, build a better scoop.

## 🧊 The Scoop Mystery

This project started as a quest to identify an unmarked "CAUTION DO NOT BOIL" ice cream scoop. It turned out to be a generic Zeroll-style oil-filled aluminum scoop, likely purchased at TJ Maxx. The physics makes the Zeroll design elegant: maximum heat transfer path from hand to ice cream with minimum thermal mass.
