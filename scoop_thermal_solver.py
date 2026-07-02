#!/usr/bin/env python3
"""
Quasi-1D transient thermal model for ice cream scoop handle + head.
Uses a proper thermal resistance network + lumped capacitance for the head.
This is simpler and more correct than fighting a buggy 2D FD grid.

Two heat paths from hand to scoop head:
1. Through the handle body (conduction along handle length)
2. (Design A only) through oil + aluminum wall

The scoop head is treated as a lumped thermal mass.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

# ============================================================
# Properties
# ============================================================
MAT = {
    'aluminum': {'k': 205.0, 'rho': 2700.0, 'cp': 900.0},
    'copper':   {'k': 401.0, 'rho': 8960.0, 'cp': 385.0},
    'oil_eff':  {'k': 2.0,   'rho': 850.0,  'cp': 2000.0},  # w/ natural convection
}

# Geometry
HANDLE_LEN = 0.150     # 150 mm
HANDLE_OD  = 0.010     # 10 mm radius
HANDLE_ID  = 0.008     # 8 mm radius (oil core)
HEAD_R     = 0.025     # 25 mm radius
HEAD_THICK = 0.004     # 4 mm effective thickness
GRIP_LEN   = 0.095     # 95 mm of handle in hand (from z=55mm to z=150mm)
GRIP_START = 0.055     # hand starts 55mm from head end

# Temperatures
T_HAND = 33.0   # °C
T_ICE  = -18.0  # °C
T_AMB  = 22.0   # °C
T_INIT = T_ICE

H_CONV = 10.0    # W/m²K ambient convection

# Time
T_TOTAL = 120.0
N_SEGMENTS = 50  # along handle length

# ============================================================
# Thermal resistance network along handle
# ============================================================
def handle_thermal_resistance(design):
    """
    Returns thermal resistance per unit length (K·m/W) for axial conduction
    along the handle. For the oil-filled design, aluminum wall and oil core
    act in parallel.
    """
    dz = HANDLE_LEN / N_SEGMENTS
    
    if design == 'A':
        # Aluminum wall cross-section area
        A_wall = np.pi * (HANDLE_OD**2 - HANDLE_ID**2) / 4
        # Oil core area
        A_oil  = np.pi * HANDLE_ID**2 / 4
        
        k_al = MAT['aluminum']['k']
        k_oil = MAT['oil_eff']['k']
        
        # Parallel: R_total = 1 / (k_al*A_wall/dz + k_oil*A_oil/dz)
        # R per segment = dz / (k_al*A_wall + k_oil*A_oil)
        R_seg = dz / (k_al * A_wall + k_oil * A_oil)
        
        cross_A = A_wall + A_oil
        k_eff_axial = (k_al * A_wall + k_oil * A_oil) / cross_A
        
        # Also compute lateral info for convection losses
        perimeter = 2 * np.pi * HANDLE_OD
        return R_seg, dz, cross_A, k_eff_axial, perimeter, MAT['aluminum']
    
    else:
        # Solid copper
        A_full = np.pi * HANDLE_OD**2 / 4
        k_cu = MAT['copper']['k']
        R_seg = dz / (k_cu * A_full)
        
        cross_A = A_full
        k_eff_axial = k_cu
        perimeter = 2 * np.pi * HANDLE_OD
        return R_seg, dz, cross_A, k_eff_axial, perimeter, MAT['copper']

# ============================================================
# Solve: 1D finite difference along handle + lumped head
# ============================================================
def solve(design='A'):
    print(f"\n=== Design {design} ===")
    R_seg, dz, cross_A, k_eff, perimeter, handle_mat = handle_thermal_resistance(design)
    
    # Handle node temperatures (N_SEGMENTS nodes along handle)
    # Node 0 = at scoop head junction, Node N-1 = handle tip
    N = N_SEGMENTS
    T_handle = np.full(N, T_INIT)
    
    # Scoop head (lumped)
    head_volume = np.pi * HEAD_R**2 * HEAD_THICK
    if design == 'A':
        head_mass = MAT['aluminum']['rho'] * head_volume
        head_cp = MAT['aluminum']['cp']
        head_k = MAT['aluminum']['k']
    else:
        head_mass = MAT['copper']['rho'] * head_volume
        head_cp = MAT['copper']['cp']
        head_k = MAT['copper']['k']
    
    T_head = T_INIT
    head_thermal_mass = head_mass * head_cp  # J/K
    
    print(f"  Handle: {N} segments, dz={dz*1000:.1f}mm, k_eff={k_eff:.1f} W/m·K")
    print(f"  Head: mass={head_mass*1000:.1f}g, thermal_mass={head_thermal_mass:.2f} J/K")
    
    # Stability: explicit FD for handle conduction
    # Fo = alpha * dt / dz² < 0.5
    alpha_handle = k_eff / (handle_mat['rho'] * handle_mat['cp'])
    dt_cond = 0.4 * dz**2 / alpha_handle
    
    # Head time constant: dt < head_thermal_mass / (k*A/dz)
    A_contact = cross_A  # area between handle node 0 and head
    R_head_to_handle = dz / (2 * k_eff * A_contact)  # half-segment resistance
    dt_head = 0.4 * head_thermal_mass * R_head_to_handle
    
    dt = min(dt_cond, dt_head, 0.01)  # cap at 10ms
    n_steps = int(T_TOTAL / dt) + 1
    record_every = max(1, int(0.5 / dt))
    
    print(f"  dt={dt*1000:.2f}ms, steps={n_steps}, record every {record_every}")
    
    # Recording
    temp_hist = []
    flux_hist = []
    snapshots = {}
    snap_times_set = {0, 10, 30, 60, 120}
    
    for step in range(n_steps + 1):
        t = step * dt
        
        # --- Apply BCs ---
        # Handle tip (node N-1): hand grip
        T_handle[-1] = T_HAND
        
        # Nodes in hand grip region: set to hand temp (simplified: hand covers fully)
        for i in range(N):
            z_from_head = (i + 0.5) * dz  # center of node i
            if z_from_head >= GRIP_START:
                T_handle[i] = T_HAND
        
        # Scoop head: in contact with ice cream at the bottom surface
        # Heat flows: ice_cream -> head_bottom, and head -> handle_node[0]
        
        # Record
        if step % record_every == 0:
            temp_hist.append((t, T_head))
            
            # Heat flux from hand into handle
            # Find the boundary between grip and non-grip nodes
            # Heat flows from the first grip node into the last non-grip node
            Q_hand = 0.0
            for i in range(N - 1):
                z_i = (i + 0.5) * dz
                z_next = (i + 1.5) * dz
                in_grip_i = z_i >= GRIP_START
                in_grip_next = z_next >= GRIP_START
                if not in_grip_i and in_grip_next:
                    # Heat flows from grip node i+1 into non-grip node i
                    dT = T_handle[i+1] - T_handle[i]
                    Q_hand += k_eff * cross_A * dT / dz
                elif in_grip_i and not in_grip_next:
                    dT = T_handle[i] - T_handle[i+1]
                    Q_hand += k_eff * cross_A * dT / dz
            
            flux_hist.append((t, Q_hand))
            
            for st in snap_times_set:
                if abs(t - st) < dt:
                    snapshots[st] = T_handle.copy()
            
            if step % (record_every * 20) == 0:
                print(f"  t={t:.1f}s  T_head={T_head:.1f}°C  Q_hand={Q_hand:.2f}W")
        
        # --- Update ---
        T_new = T_handle.copy()
        
        # Handle interior nodes: 1D heat equation
        for i in range(1, N - 1):
            z_center = (i + 0.5) * dz
            if z_center >= GRIP_START:
                continue  # fixed at hand temp
            
            # Conduction from neighbors
            d2T = (T_handle[i+1] - 2*T_handle[i] + T_handle[i-1]) / dz**2
            dTdt = alpha_handle * d2T
            
            # Ambient convection loss (lateral)
            node_area_surface = perimeter * dz  # surface area of this segment
            node_volume = cross_A * dz
            Q_conv = H_CONV * node_area_surface * (T_handle[i] - T_AMB)
            dTdt_conv = -Q_conv / (handle_mat['rho'] * handle_mat['cp'] * node_volume)
            
            T_new[i] = T_handle[i] + dt * (dTdt + dTdt_conv)
        
        # Node 0 (at head junction): conduction to head + conduction to node 1
        # Not in grip region
        z0 = 0.5 * dz
        if z0 < GRIP_START:
            d2T_0 = (T_handle[1] - T_handle[0]) / dz**2  # one-sided (ghost = T_head direction)
            # Actually: heat balance for node 0
            # Q_from_node1 = k*A*(T[1]-T[0])/dz
            # Q_from_head = k*A*(T_head - T[0])/(dz/2)  -- head is "half segment" away
            Q_from_1 = k_eff * cross_A * (T_handle[1] - T_handle[0]) / dz
            Q_from_head = head_k * cross_A * (T_head - T_handle[0]) / (dz / 2)
            Q_conv_0 = H_CONV * perimeter * dz * (T_handle[0] - T_AMB)
            
            node_vol_0 = cross_A * dz
            dTdt_0 = (Q_from_1 + Q_from_head - Q_conv_0) / (handle_mat['rho'] * handle_mat['cp'] * node_vol_0)
            T_new[0] = T_handle[0] + dt * dTdt_0
        
        # Update handle
        T_handle = T_new
        
        # Scoop head (lumped capacitance)
        # Energy balance: heat from handle node 0 - heat lost to ice cream
        Q_from_handle = head_k * cross_A * (T_handle[0] - T_head) / (dz / 2)
        
        # Ice cream contact: bottom surface of head
        A_ice_contact = np.pi * HEAD_R**2  # full circular area
        Q_to_ice = H_CONV * 2.0 * A_ice_contact * (T_head - T_ICE)  # high h for contact
        
        # Also ambient on sides of head
        A_head_side = 2 * np.pi * HEAD_R * HEAD_THICK
        Q_head_amb = H_CONV * A_head_side * (T_head - T_AMB)
        
        dT_head = (Q_from_handle - Q_to_ice - Q_head_amb) * dt / head_thermal_mass
        T_head += dT_head
    
    print(f"  Final T_head: {T_head:.2f}°C")
    
    return {
        'temp_history': np.array(temp_hist),
        'flux_history': np.array(flux_hist),
        'snapshots': snapshots,
        'N': N,
        'dz': dz,
    }

# ============================================================
# Plots
# ============================================================
def make_plots(res_a, res_b, outdir):
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Head temp vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    ta, Ta = res_a['temp_history'][:, 0], res_a['temp_history'][:, 1]
    tb, Tb = res_b['temp_history'][:, 0], res_b['temp_history'][:, 1]
    ax.plot(ta, Ta, 'b-', linewidth=2.5, label='Design A: Al tube + mineral oil')
    ax.plot(tb, Tb, 'r-', linewidth=2.5, label='Design B: Solid copper')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='0°C (freezing point)')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Scoop Head Temperature (°C)', fontsize=13)
    ax.set_title('Ice Cream Scoop Head Warm-up: Oil-Filled vs Solid Copper', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'head_temp_vs_time.png'), dpi=150)
    plt.close(fig)
    
    # 2. Heat flux vs time
    fig, ax = plt.subplots(figsize=(10, 6))
    fa, Qa = res_a['flux_history'][:, 0], res_a['flux_history'][:, 1]
    fb, Qb = res_b['flux_history'][:, 0], res_b['flux_history'][:, 1]
    ax.plot(fa, Qa, 'b-', linewidth=2.5, label='Design A: Al tube + mineral oil')
    ax.plot(fb, Qb, 'r-', linewidth=2.5, label='Design B: Solid copper')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Heat Flux from Hand (W)', fontsize=13)
    ax.set_title('Heat Transfer Rate from Hand to Scoop', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'heat_flux_vs_time.png'), dpi=150)
    plt.close(fig)
    
    # 3. Handle temperature profiles at different times
    for label, res in [('A', res_a), ('B', res_b)]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for snap_t in [0, 5, 10, 30, 60, 120]:
            if snap_t not in res['snapshots']:
                continue
            T_prof = res['snapshots'][snap_t]
            z_vals = np.arange(res['N']) * res['dz'] * 1000  # mm
            ax.plot(z_vals, T_prof, linewidth=2, label=f't={snap_t}s')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvspan(55, 150, alpha=0.1, color='orange', label='Hand grip region')
        ax.set_xlabel('Distance from scoop head (mm)', fontsize=13)
        ax.set_ylabel('Temperature (°C)', fontsize=13)
        ax.set_title(f'Design {label}: Handle Temperature Profile Over Time', fontsize=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'handle_profile_design_{label}.png'), dpi=150)
        plt.close(fig)
    
    print(f"Plots saved to {outdir}")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(outdir, exist_ok=True)
    
    res_a = solve('A')
    res_b = solve('B')
    
    np.save(os.path.join(outdir, 'temp_history_a.npy'), res_a['temp_history'])
    np.save(os.path.join(outdir, 'temp_history_b.npy'), res_b['temp_history'])
    np.save(os.path.join(outdir, 'flux_history_a.npy'), res_a['flux_history'])
    np.save(os.path.join(outdir, 'flux_history_b.npy'), res_b['flux_history'])
    
    make_plots(res_a, res_b, outdir)
    
    # Summary
    Ta = res_a['temp_history'][:, 1]
    Tb = res_b['temp_history'][:, 1]
    Qa = res_a['flux_history'][:, 1]
    Qb = res_b['flux_history'][:, 1]
    
    idx_a = np.where(Ta >= 0)[0]
    idx_b = np.where(Tb >= 0)[0]
    
    t_0a = res_a['temp_history'][idx_a[0], 0] if len(idx_a) else None
    t_0b = res_b['temp_history'][idx_b[0], 0] if len(idx_b) else None
    
    summary = {
        'design_A_oil_filled_aluminum': {
            'time_to_0C_s': float(t_0a) if t_0a is not None else None,
            'peak_flux_W': float(np.max(Qa)),
            'final_temp_C': float(Ta[-1]),
            'steady_state_flux_W': float(Qa[-1]),
        },
        'design_B_solid_copper': {
            'time_to_0C_s': float(t_0b) if t_0b is not None else None,
            'peak_flux_W': float(np.max(Qb)),
            'final_temp_C': float(Tb[-1]),
            'steady_state_flux_W': float(Qb[-1]),
        },
    }
    if t_0a and t_0b:
        summary['copper_speedup_ratio'] = float(t_0a / t_0b)
    
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
