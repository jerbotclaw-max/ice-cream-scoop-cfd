#!/usr/bin/env python3
"""
2D Axisymmetric Transient Thermal Solver for Ice Cream Scoop Comparison
Design A: Aluminum tube + mineral oil fill + aluminum head
Design B: Solid copper handle + copper head

Runs on CPU (numpy) — fast enough for this grid size.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import json

# ============================================================
# Material properties
# ============================================================
MATERIALS = {
    'aluminum': {'k': 205.0, 'rho': 2700.0, 'cp': 900.0},
    'copper':   {'k': 401.0, 'rho': 8960.0, 'cp': 385.0},
    'oil_eff':  {'k': 2.0,   'rho': 850.0,  'cp': 2000.0},  # effective with convection
}

# ============================================================
# Grid parameters (SI units: meters, Kelvin, Watts)
# ============================================================
R_MAX = 0.025   # 25 mm radial
Z_MAX = 0.155   # 155 mm axial (150mm handle + 5mm head region at bottom)
DR = 0.0005     # 0.5 mm
DZ = 0.0005     # 0.5 mm

Nr = int(R_MAX / DR) + 1  # 51
Nz = int(Z_MAX / DZ) + 1  # 312

# Z coordinates (z=0 is scoop head bottom, z increases toward handle end)
z_coords = np.arange(Nz) * DZ  # 0 to 0.155m
r_coords = np.arange(Nr) * DR  # 0 to 0.025m

# Geometry boundaries
HANDLE_OD = 0.010   # 10 mm radius
HANDLE_ID = 0.008   # 8 mm radius (for oil-filled design)
HANDLE_Z_START = 0.005  # scoop head is z=0 to 5mm, handle starts at 5mm
HANDLE_Z_END = 0.155    # end of handle

SCOOP_HEAD_R = 0.025    # 25 mm radius (50mm diameter)
SCOOP_HEAD_Z_END = 0.005  # 5mm thick head region

# Hand grip: covers handle from z=55mm to z=150mm
HAND_Z_START = 0.055
HAND_Z_END = 0.155

# Boundary condition temperatures
T_HAND = 33.0 + 273.15     # 306.15 K
T_ICECREAM = -18.0 + 273.15  # 255.15 K
T_AMB = 22.0 + 273.15      # 295.15 K
T_INIT = T_ICECREAM        # start fully cold

H_CONV = 10.0  # W/m²K ambient convection

# Simulation time
T_TOTAL = 120.0  # seconds
DT_RECORD = 0.5  # record every 0.5s

# ============================================================
# Build material maps
# ============================================================
def build_material_map(design='A'):
    """Build 2D arrays of k, rho, cp over the (Nr, Nz) grid.
    Grid indexing: [ir, iz] where ir is radial index, iz is axial index.
    """
    k = np.full((Nr, Nz), MATERIALS['aluminum']['k'])
    rho = np.full((Nr, Nz), MATERIALS['aluminum']['rho'])
    cp = np.full((Nr, Nz), MATERIALS['aluminum']['cp'])
    
    # Default: everything outside the object is "air" (low k, for ambient)
    # We'll handle ambient via boundary conditions instead.
    # The domain IS the scoop object — nodes outside the object get ambient BC treatment.
    
    # Object mask: True where material exists
    obj = np.zeros((Nr, Nz), dtype=bool)
    
    for ir in range(Nr):
        r = ir * DR
        for iz in range(Nz):
            z = iz * DZ
            
            in_handle = (HANDLE_Z_START <= z <= HANDLE_Z_END) and (r <= HANDLE_OD)
            in_head = (z <= SCOOP_HEAD_Z_END) and (r <= SCOOP_HEAD_R)
            
            if in_handle or in_head:
                obj[ir, iz] = True
                
                if design == 'A':
                    # Liquid-filled: oil core in handle, aluminum wall + head
                    if in_handle and r < HANDLE_ID:
                        k[ir, iz] = MATERIALS['oil_eff']['k']
                        rho[ir, iz] = MATERIALS['oil_eff']['rho']
                        cp[ir, iz] = MATERIALS['oil_eff']['cp']
                    else:
                        k[ir, iz] = MATERIALS['aluminum']['k']
                        rho[ir, iz] = MATERIALS['aluminum']['rho']
                        cp[ir, iz] = MATERIALS['aluminum']['cp']
                elif design == 'B':
                    # Solid copper throughout
                    k[ir, iz] = MATERIALS['copper']['k']
                    rho[ir, iz] = MATERIALS['copper']['rho']
                    cp[ir, iz] = MATERIALS['copper']['cp']
    
    # Compute thermal diffusivity
    alpha = k / (rho * cp)
    
    return k, rho, cp, alpha, obj

# ============================================================
# Determine dt for stability (explicit FD)
# ============================================================
def compute_dt(alpha, obj):
    """Fourier number stability: Fo = alpha*dt/dr² < 0.25 for 2D"""
    alpha_max = np.max(alpha[obj])
    fo_limit = 0.20  # safety factor
    dt_r = fo_limit * DR**2 / alpha_max
    dt_z = fo_limit * DZ**2 / alpha_max
    dt = min(dt_r, dt_z)
    return dt

# ============================================================
# Identify boundary nodes
# ============================================================
def find_boundary_nodes(obj):
    """Find nodes on the surface of the object (object nodes with non-object neighbors)."""
    boundary = {
        'hand': [],      # nodes at handle outer surface in hand grip region
        'icecream': [],  # nodes at scoop head bottom surface
        'ambient': [],   # all other surface nodes
    }
    
    for ir in range(Nr):
        for iz in range(Nz):
            if not obj[ir, iz]:
                continue
            
            r = ir * DR
            z = iz * DZ
            
            # Check if this is a surface node
            is_surface = False
            neighbors = [(ir-1, iz), (ir+1, iz), (ir, iz-1), (ir, iz+1)]
            for nir, niz in neighbors:
                if nir < 0 or nir >= Nr or niz < 0 or niz >= Nz:
                    is_surface = True
                elif not obj[nir, niz]:
                    is_surface = True
            
            if not is_surface:
                continue
            
            # Classify boundary
            # Hand grip: outer surface of handle in grip region
            if (r >= HANDLE_ID and HANDLE_Z_START <= z <= HANDLE_Z_END 
                and HAND_Z_START <= z <= HAND_Z_END 
                and r >= HANDLE_OD - DR):
                boundary['hand'].append((ir, iz))
            # Ice cream contact: bottom of scoop head (z=0 or z small)
            elif z <= DZ and r <= SCOOP_HEAD_R:
                boundary['icecream'].append((ir, iz))
            # Scoop head outer edge (curved part) - also ice cream contact
            elif (z <= SCOOP_HEAD_Z_END and r >= SCOOP_HEAD_R - DR):
                boundary['icecream'].append((ir, iz))
            else:
                boundary['ambient'].append((ir, iz))
    
    return boundary

# ============================================================
# Solve
# ============================================================
def solve(design='A'):
    print(f"\n=== Solving Design {design} ===")
    k, rho, cp, alpha, obj = build_material_map(design)
    dt = compute_dt(alpha, obj)
    n_steps = int(T_TOTAL / dt)
    record_every = int(DT_RECORD / dt)
    
    print(f"  Grid: {Nr} x {Nz}, dt={dt:.6f}s, steps={n_steps}, record every {record_every}")
    
    # Initialize temperature
    T = np.full((Nr, Nz), T_INIT)
    # Set non-object nodes to ambient
    T[~obj] = T_AMB
    
    # Boundary nodes
    bnodes = find_boundary_nodes(obj)
    print(f"  Boundaries: hand={len(bnodes['hand'])}, icecream={len(bnodes['icecream'])}, ambient={len(bnodes['ambient'])}")
    
    # Scoop head center monitoring point (r=0, z=2.5mm)
    mon_ir = 0
    mon_iz = int(0.0025 / DZ)
    
    # Recording arrays
    temp_history = []
    flux_history = []
    snapshot_times = [0, 10, 30, 60, 120]
    snapshots = {}
    
    t = 0.0
    for step in range(n_steps + 1):
        # Apply Dirichlet BCs
        for ir, iz in bnodes['hand']:
            T[ir, iz] = T_HAND
        for ir, iz in bnodes['icecream']:
            T[ir, iz] = T_ICECREAM
        
        # Ambient: convection (Robin BC) — approximate by setting to ambient
        # (simplified: the ambient exposure effect is small compared to hand+icecream)
        for ir, iz in bnodes['ambient']:
            # Newton's law of cooling approximation at surface
            T[ir, iz] = T_AMB  # simplified
        
        # Record
        if step % record_every == 0:
            t_rec = step * dt
            T_head = T[mon_ir, mon_iz]
            # Compute heat flux through hand boundary
            # Q = sum(k * A * dT/dz) — approximate
            Q_hand = 0.0
            for ir, iz in bnodes['hand']:
                # gradient toward interior
                if iz + 1 < Nz and obj[ir, iz+1]:
                    dTdz = (T[ir, iz] - T[ir, iz+1]) / DZ
                    area = 2 * np.pi * (ir * DR) * DR  # circumferential area element
                    Q_hand += k[ir, iz] * area * abs(dTdz)
                elif iz - 1 >= 0 and obj[ir, iz-1]:
                    dTdz = (T[ir, iz] - T[ir, iz-1]) / DZ
                    area = 2 * np.pi * (ir * DR) * DR
                    Q_hand += k[ir, iz] * area * abs(dTdz)
            
            temp_history.append((t_rec, T_head - 273.15))
            flux_history.append((t_rec, Q_hand))
            
            # Snapshots
            for snap_t in snapshot_times:
                if abs(t_rec - snap_t) < dt:
                    snapshots[snap_t] = T.copy() - 273.15
            
            if step % (record_every * 10) == 0:
                print(f"  t={t_rec:.1f}s, T_head={T_head-273.15:.1f}°C, Q_hand={Q_hand:.2f}W")
        
        # Update interior nodes (explicit FD for heat equation in cylindrical coords)
        T_new = T.copy()
        for ir in range(1, Nr - 1):
            for iz in range(1, Nz - 1):
                if not obj[ir, iz]:
                    continue
                # Skip boundary nodes (already set)
                is_bc = False
                for key in bnodes:
                    if (ir, iz) in bnodes[key]:
                        # We use a set for fast lookup below instead
                        pass
                
                r = ir * DR
                # Laplacian in cylindrical coords
                d2T_dr2 = (T[ir+1, iz] - 2*T[ir, iz] + T[ir-1, iz]) / DR**2
                dT_dr = (T[ir+1, iz] - T[ir-1, iz]) / (2 * DR)
                d2T_dz2 = (T[ir, iz+1] - 2*T[ir, iz] + T[ir, iz-1]) / DZ**2
                
                dTdt = alpha[ir, iz] * (d2T_dr2 + (1.0/r) * dT_dr + d2T_dz2)
                T_new[ir, iz] = T[ir, iz] + dt * dTdt
        
        # Axis boundary (r=0): symmetry dT/dr = 0
        for iz in range(1, Nz - 1):
            if obj[0, iz]:
                # Use ghost node: T[-1,iz] = T[1,iz], and limit of (1/r)*dT/dr -> d2T/dr2 at r=0
                d2T_dr2_axial = 2 * (T[1, iz] - T[0, iz]) / DR**2
                d2T_dz2 = (T[0, iz+1] - 2*T[0, iz] + T[0, iz-1]) / DZ**2
                dTdt = alpha[0, iz] * (2 * d2T_dr2_axial + d2T_dz2)
                T_new[0, iz] = T[0, iz] + dt * dTdt
        
        T = T_new
        t += dt
    
    return {
        'temp_history': np.array(temp_history),
        'flux_history': np.array(flux_history),
        'snapshots': snapshots,
        'T_final': T - 273.15,
        'mon_ir': mon_ir,
        'mon_iz': mon_iz,
    }

# ============================================================
# Vectorized solver (faster)
# ============================================================
def solve_vectorized(design='A'):
    print(f"\n=== Solving Design {design} (vectorized) ===")
    k, rho, cp, alpha, obj = build_material_map(design)
    dt = compute_dt(alpha, obj)
    n_steps = int(T_TOTAL / dt)
    record_every = max(1, int(DT_RECORD / dt))
    
    print(f"  Grid: {Nr} x {Nz}, dt={dt:.6f}s, steps={n_steps}, record every {record_every}")
    
    T = np.full((Nr, Nz), T_INIT, dtype=np.float64)
    T[~obj] = T_AMB
    
    bnodes = find_boundary_nodes(obj)
    
    # Build boundary masks
    hand_mask = np.zeros((Nr, Nz), dtype=bool)
    ice_mask = np.zeros((Nr, Nz), dtype=bool)
    amb_mask = np.zeros((Nr, Nz), dtype=bool)
    for ir, iz in bnodes['hand']:
        hand_mask[ir, iz] = True
    for ir, iz in bnodes['icecream']:
        ice_mask[ir, iz] = True
    for ir, iz in bnodes['ambient']:
        amb_mask[ir, iz] = True
    
    print(f"  Boundaries: hand={len(bnodes['hand'])}, icecream={len(bnodes['icecream'])}, ambient={len(bnodes['ambient'])}")
    
    mon_ir = 0
    mon_iz = int(0.0025 / DZ)
    
    temp_history = []
    flux_history = []
    snapshot_times = {0, 10, 30, 60, 120}
    snapshots = {}
    
    # Precompute r array
    r_arr = np.arange(Nr).reshape(-1, 1) * DR  # (Nr, 1)
    r_arr_safe = np.where(r_arr > 0, r_arr, 1e-10)
    
    for step in range(n_steps + 1):
        # Apply BCs
        T[hand_mask] = T_HAND
        T[ice_mask] = T_ICECREAM
        T[amb_mask] = T_AMB
        
        # Record
        if step % record_every == 0:
            t_rec = step * dt
            T_head = T[mon_ir, mon_iz]
            # Heat flux through hand boundary
            Q_hand = 0.0
            for ir, iz in bnodes['hand']:
                if iz + 1 < Nz and obj[ir, iz+1]:
                    dTdz = abs(T[ir, iz] - T[ir, iz+1]) / DZ
                    area = 2 * np.pi * (ir * DR) * DR
                    Q_hand += k[ir, iz] * area * dTdz
            temp_history.append((t_rec, T_head - 273.15))
            flux_history.append((t_rec, Q_hand))
            
            for snap_t in snapshot_times:
                if abs(t_rec - snap_t) < dt:
                    snapshots[snap_t] = T.copy() - 273.15
            
            if step % (record_every * 20) == 0:
                print(f"  t={t_rec:.1f}s, T_head={T_head-273.15:.1f}°C, Q_hand={Q_hand:.2f}W")
        
        # Compute Laplacian (vectorized)
        T_new = T.copy()
        
        # Interior nodes
        d2T_dr2 = np.zeros_like(T)
        dT_dr = np.zeros_like(T)
        d2T_dz2 = np.zeros_like(T)
        
        d2T_dr2[1:-1, :] = (T[2:, :] - 2*T[1:-1, :] + T[:-2, :]) / DR**2
        dT_dr[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2*DR)
        d2T_dz2[:, 1:-1] = (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / DZ**2
        
        laplacian = d2T_dr2 + (1.0/r_arr_safe) * dT_dr + d2T_dz2
        dTdt = alpha * laplacian
        
        # Only update interior object nodes (not boundaries)
        interior = obj.copy()
        interior[0, :] = False       # axis handled separately
        interior[-1, :] = False
        interior[:, 0] = False
        interior[:, -1] = False
        interior[hand_mask] = False
        interior[ice_mask] = False
        interior[amb_mask] = False
        
        T_new[interior] = T[interior] + dt * dTdt[interior]
        
        # Axis (r=0): symmetry
        for iz in range(1, Nz - 1):
            if obj[0, iz] and not hand_mask[0, iz] and not ice_mask[0, iz] and not amb_mask[0, iz]:
                d2T_dr2_ax = 2 * (T[1, iz] - T[0, iz]) / DR**2
                d2T_dz2_ax = (T[0, iz+1] - 2*T[0, iz] + T[0, iz-1]) / DZ**2
                dTdt_ax = alpha[0, iz] * (2 * d2T_dr2_ax + d2T_dz2_ax)
                T_new[0, iz] = T[0, iz] + dt * dTdt_ax
        
        T = T_new
    
    return {
        'temp_history': np.array(temp_history),
        'flux_history': np.array(flux_history),
        'snapshots': snapshots,
        'T_final': T - 273.15,
        'obj': obj,
        'mon_ir': mon_ir,
        'mon_iz': mon_iz,
        'k_map': k,
    }

# ============================================================
# Plotting
# ============================================================
def make_plots(results_a, results_b, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scoop head temperature vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ta, Ta = results_a['temp_history'][:, 0], results_a['temp_history'][:, 1]
    tb, Tb = results_b['temp_history'][:, 0], results_b['temp_history'][:, 1]
    ax.plot(ta, Ta, 'b-', linewidth=2, label='Design A: Al tube + mineral oil')
    ax.plot(tb, Tb, 'r-', linewidth=2, label='Design B: Solid copper')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='0°C (freezing)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Scoop Head Temperature (°C)', fontsize=12)
    ax.set_title('Ice Cream Scoop Head Temperature: Oil-Filled vs Copper', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'scoop_head_temp_vs_time.png'), dpi=150)
    plt.close(fig)
    
    # 2. Heat flux vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    fa, Qa = results_a['flux_history'][:, 0], results_a['flux_history'][:, 1]
    fb, Qb = results_b['flux_history'][:, 0], results_b['flux_history'][:, 1]
    ax.plot(fa, Qa, 'b-', linewidth=2, label='Design A: Al tube + mineral oil')
    ax.plot(fb, Qb, 'r-', linewidth=2, label='Design B: Solid copper')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heat Flux from Hand (W)', fontsize=12)
    ax.set_title('Heat Transfer Rate from Hand to Scoop', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'heat_flux_vs_time.png'), dpi=150)
    plt.close(fig)
    
    # 3. Temperature field snapshots
    for design_name, results in [('A', results_a), ('B', results_b)]:
        for snap_t in [0, 10, 30, 60, 120]:
            if snap_t not in results['snapshots']:
                continue
            T_field = results['snapshots'][snap_t]
            obj = results['obj']
            
            fig, ax = plt.subplots(figsize=(6, 12))
            # Mask non-object nodes
            T_display = np.ma.masked_where(~obj, T_field)
            # Plot in z-r coordinates (z vertical, r horizontal)
            im = ax.pcolormesh(r_coords * 1000, z_coords * 1000, T_display.T, 
                              shading='auto', cmap='RdYlBu_r', vmin=-20, vmax=35)
            ax.set_xlabel('r (mm)', fontsize=11)
            ax.set_ylabel('z (mm)', fontsize=11)
            ax.set_title(f'Design {design_name} — t={snap_t}s', fontsize=13)
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, label='Temperature (°C)')
            fig.tight_layout()
            fname = f'temp_field_design_{design_name}_t{snap_t}s.png'
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close(fig)
    
    print(f"  Plots saved to {output_dir}")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Solve both designs
    results_a = solve_vectorized('A')
    results_b = solve_vectorized('B')
    
    # Save raw data
    np.save(os.path.join(output_dir, 'temp_history_a.npy'), results_a['temp_history'])
    np.save(os.path.join(output_dir, 'temp_history_b.npy'), results_b['temp_history'])
    np.save(os.path.join(output_dir, 'flux_history_a.npy'), results_a['flux_history'])
    np.save(os.path.join(output_dir, 'flux_history_b.npy'), results_b['flux_history'])
    
    # Generate plots
    make_plots(results_a, results_b, output_dir)
    
    # Summary
    ta, Ta = results_a['temp_history'][:, 0], results_a['temp_history'][:, 1]
    tb, Tb = results_b['temp_history'][:, 0], results_b['temp_history'][:, 1]
    
    # Time to 0°C
    idx_a = np.where(Ta >= 0)[0]
    t_0_a = ta[idx_a[0]] if len(idx_a) > 0 else None
    idx_b = np.where(Tb >= 0)[0]
    t_0_b = tb[idx_b[0]] if len(idx_b) > 0 else None
    
    # Peak flux
    Q_peak_a = np.max(results_a['flux_history'][:, 1])
    Q_peak_b = np.max(results_b['flux_history'][:, 1])
    
    # Final temps
    T_final_a = Ta[-1]
    T_final_b = Tb[-1]
    
    summary = {
        'design_A_liquid_aluminum': {
            'time_to_0C_s': float(t_0_a) if t_0_a else None,
            'peak_heat_flux_W': float(Q_peak_a),
            'final_temp_C': float(T_final_a),
        },
        'design_B_solid_copper': {
            'time_to_0C_s': float(t_0_b) if t_0_b else None,
            'peak_heat_flux_W': float(Q_peak_b),
            'final_temp_C': float(T_final_b),
        },
        'performance_ratio': float(t_0_a / t_0_b) if (t_0_a and t_0_b) else None,
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {output_dir}")
