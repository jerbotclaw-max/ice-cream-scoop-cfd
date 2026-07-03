#!/usr/bin/env python3
"""
3D Conjugate Heat Transfer CFD Solver for Ice Cream Scoop Comparison.

Solves full Navier-Stokes + Energy equations on a 3D Cartesian grid.
- Oil region: Incompressible NS with Boussinesq buoyancy (natural convection)
- Solid regions (aluminum/copper): Pure conduction
- Conjugate heat transfer at fluid-solid interfaces

Uses projection method (Chorin) for pressure-velocity coupling.
Staggered grid: velocities at faces, pressure/temperature at centers.

Runs on GPU via PyTorch.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import os

# ============================================================
# Material Properties
# ============================================================
MATERIALS = {
    'aluminum': {'k': 205.0, 'rho': 2700.0, 'cp': 900.0},
    'copper':   {'k': 401.0, 'rho': 8960.0, 'cp': 385.0},
    'oil':      {'k': 0.15, 'rho': 850.0, 'cp': 1670.0,
                 'mu': 0.08, 'beta': 6.4e-4},  # thermal expansion coeff
}

# Boundary conditions
T_HAND = 33.0    # °C - hand temperature on grip
T_ICE  = -18.0   # °C - ice cream contact
T_AMB  = 22.0    # °C - ambient air
H_AIR  = 10.0    # W/m²K - natural convection to air

# Gravity
G = 9.81  # m/s²

# ============================================================
# Grid & Geometry
# ============================================================
# Domain: 60mm x 60mm x 200mm (x, y, z)
# z-axis: scoop head at z=0, handle extends to z=150mm
# Handle: cylinder OD=20mm centered on z-axis
#   Oil-filled: tube wall OD=20, ID=18mm
#   Copper solid: OD=20mm
# Scoop head: hemisphere D=50mm, wall=2mm at z=0

DOMAIN_X = 0.060  # m
DOMAIN_Y = 0.060
DOMAIN_Z = 0.200

NX, NY, NZ = 48, 48, 160  # grid resolution (~1.25mm cells)

# Handle geometry
HANDLE_OD = 0.020   # 20mm
HANDLE_ID = 0.018   # 18mm (inner radius for oil tube)
HANDLE_LEN = 0.150  # 150mm
HANDLE_GRIP_START = 0.050  # grip from z=50mm to z=150mm

# Scoop head
SCOOP_D = 0.050     # 50mm diameter
SCOOP_R = SCOOP_D / 2
SCOOP_WALL = 0.002  # 2mm

def build_geometry(nx, ny, nz, dx, dy, dz, design='oil'):
    """
    Build material masks for the computation domain.
    
    Returns:
        fluid_mask: boolean (nx, ny, nz) - True where oil is
        solid_mask: boolean (nx, ny, nz) - True where metal is
        k_field: float (nx, ny, nz) - thermal conductivity
        rho_cp_field: float (nx, ny, nz) - rho*cp
        bc_type: int array (nx, ny, nz) - boundary condition type
        bc_temp: float (nx, ny, nz) - boundary temperature
    """
    # Cell center coordinates
    x = (torch.arange(nx, dtype=torch.float32) + 0.5) * dx - DOMAIN_X / 2
    y = (torch.arange(ny, dtype=torch.float32) + 0.5) * dy - DOMAIN_Y / 2
    z = (torch.arange(nz, dtype=torch.float32) + 0.5) * dz
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    # Now indexing: [iz, iy, ix]
    R = torch.sqrt(X**2 + Y**2)  # radial distance from z-axis
    
    fluid_mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid_mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
    k_field = torch.zeros((nz, ny, nx), dtype=torch.float32)
    rho_cp = torch.zeros((nz, ny, nx), dtype=torch.float32)
    bc_type = torch.zeros((nz, ny, nx), dtype=torch.int32)  # 0=interior, 1=hand, 2=ice, 3=air
    bc_temp = torch.full((nz, ny, nx), T_AMB, dtype=torch.float32)
    
    # --- Scoop head region (z < scoop wall thickness) ---
    # Hemisphere of radius SCOOP_R, shell thickness SCOOP_WALL
    in_hemisphere = (Z < SCOOP_R) & (R < SCOOP_R)
    hem_r = torch.sqrt(R**2 + Z**2)
    in_shell = in_hemisphere & (hem_r > SCOOP_R - SCOOP_WALL)
    
    # Handle region: cylindrical tube
    in_handle_rad = R < HANDLE_OD / 2
    in_handle_z = (Z >= 0) & (Z < HANDLE_LEN)
    in_handle = in_handle_rad & in_handle_z
    
    if design == 'oil':
        # Oil-filled: tube wall is aluminum, interior is oil
        in_tube_wall = in_handle & (R >= HANDLE_ID / 2) & (R < HANDLE_OD / 2)
        in_oil = in_handle & (R < HANDLE_ID / 2)
        
        # Assign materials
        # Aluminum shell + scoop head
        al_mask = in_tube_wall | (in_shell & (Z < SCOOP_R))
        solid_mask[al_mask] = True
        k_field[al_mask] = MATERIALS['aluminum']['k']
        rho_cp[al_mask] = MATERIALS['aluminum']['rho'] * MATERIALS['aluminum']['cp']
        
        # Oil interior
        fluid_mask[in_oil] = True
        k_field[in_oil] = MATERIALS['oil']['k']
        rho_cp[in_oil] = MATERIALS['oil']['rho'] * MATERIALS['oil']['cp']
        
    elif design == 'copper':
        # Solid copper handle + copper scoop head
        cu_mask = in_handle | (in_shell & (Z < SCOOP_R))
        solid_mask[cu_mask] = True
        k_field[cu_mask] = MATERIALS['copper']['k']
        rho_cp[cu_mask] = MATERIALS['copper']['rho'] * MATERIALS['copper']['cp']
    
    # --- Boundary conditions ---
    # Hand grip: outer surface of handle where z > 50mm
    grip_z = (Z >= HANDLE_GRIP_START) & (Z < HANDLE_LEN)
    # Mark cells just outside handle radius as hand BC
    near_handle_surface = (R >= HANDLE_OD / 2) & (R < HANDLE_OD / 2 + 2 * dx) & grip_z
    # Actually, apply hand temp directly to handle outer surface cells
    handle_surface = in_handle & (R >= HANDLE_OD / 2 - 2*dx) & grip_z
    bc_type[handle_surface] = 1
    bc_temp[handle_surface] = T_HAND
    
    # Ice cream contact: bottom of scoop head
    scoop_bottom = in_shell & (Z < 2 * dz) & (R < SCOOP_R * 0.8)
    bc_type[scoop_bottom] = 2
    bc_temp[scoop_bottom] = T_ICE
    
    # Air: outer surfaces of scoop head not in contact with ice cream
    scoop_air = in_shell & (bc_type != 2) & (bc_type != 1)
    bc_type[scoop_air] = 3
    bc_temp[scoop_air] = T_AMB
    
    # Fill any remaining zero k (air cells - treat as insulation, low k)
    air_cells = (k_field == 0)
    k_field[air_cells] = 0.026  # air thermal conductivity
    rho_cp[air_cells] = 1.225 * 1005  # air rho*cp
    
    return fluid_mask, solid_mask, k_field, rho_cp, bc_type, bc_temp, (X, Y, Z, R)


def solve_cfd(design='oil', device='cuda', sim_time=120.0, output_dir='results'):
    """Run the full CFD simulation."""
    
    dx = DOMAIN_X / NX
    dy = DOMAIN_Y / NY
    dz = DOMAIN_Z / NZ
    
    print(f"=== 3D CFD Solver: {design.upper()} design ===")
    print(f"Grid: {NX}x{NY}x{NZ} = {NX*NY*NZ:,} cells")
    print(f"Cell size: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")
    
    fluid_mask, solid_mask, k_field, rho_cp, bc_type, bc_temp, coords = build_geometry(
        NX, NY, NZ, dx, dy, dz, design=design
    )
    
    fluid_mask = fluid_mask.to(device)
    solid_mask = solid_mask.to(device)
    k_field = k_field.to(device)
    rho_cp = rho_cp.to(device)
    bc_type = bc_type.to(device)
    bc_temp = bc_temp.to(device)
    
    n_fluid = fluid_mask.sum().item()
    n_solid = solid_mask.sum().item()
    print(f"Fluid cells: {n_fluid:,}, Solid cells: {n_solid:,}")
    
    # Temperature field - start at freezer temp
    T = torch.full((NZ, NY, NX), T_ICE, dtype=torch.float32, device=device)
    
    # Velocity field (staggered: u, v, w at faces)
    # For simplicity, store velocities at cell centers (collocated grid)
    # Use projection method for pressure-velocity coupling
    u = torch.zeros((NZ, NY, NX), dtype=torch.float32, device=device)
    v = torch.zeros((NZ, NY, NX), dtype=torch.float32, device=device)
    w = torch.zeros((NZ, NY, NX), dtype=torch.float32, device=device)
    
    # Material properties for fluid
    mu = MATERIALS['oil']['mu']
    rho_oil = MATERIALS['oil']['rho']
    beta = MATERIALS['oil']['beta']
    cp_oil = MATERIALS['oil']['cp']
    
    # Thermal diffusivity
    alpha = k_field / rho_cp  # m²/s
    
    # Time step (diffusion CFL)
    alpha_max = alpha.max().item()
    dt_diff = 0.25 * min(dx, dy, dz)**2 / max(alpha_max, 1e-8)
    
    # For NS, also check viscous CFL and buoyancy time scale
    nu = mu / rho_oil  # kinematic viscosity
    dt_visc = 0.25 * min(dx, dy, dz)**2 / nu
    
    # Buoyancy time scale: how fast does buoyancy accelerate fluid?
    # dt_buoy ~ sqrt(dx / (g * beta * dT))
    dT_buoy = 50.0  # rough estimate
    dt_buoy = 0.5 * np.sqrt(min(dx, dy, dz) / (G * beta * dT_buoy))
    
    dt = min(dt_diff, dt_visc, dt_buoy, 0.01)  # cap at 10ms
    n_steps = int(sim_time / dt) + 1
    dt = sim_time / n_steps  # adjust to hit sim_time exactly
    
    print(f"alpha_max = {alpha_max:.4e} m²/s")
    print(f"nu (oil) = {nu:.4e} m²/s")
    print(f"dt = {dt:.4e} s ({dt*1000:.3f} ms)")
    print(f"Steps: {n_steps:,}")
    
    # Check Rayleigh number for the oil tube
    # Ra = g * beta * dT * L^3 / (nu * alpha)
    L_char = HANDLE_ID  # characteristic length (tube diameter)
    Ra = G * beta * abs(T_HAND - T_ICE) * L_char**3 / (nu * (k_field[fluid_mask].mean().item() / (rho_oil * cp_oil)))
    print(f"Rayleigh number (oil tube): ~{Ra:.2f}")
    print(f"Reynolds estimate: ~{np.sqrt(Ra) * nu / (G * beta * abs(T_HAND - T_ICE) * L_char**2):.4f} (very low)")
    
    # Storage for results
    snapshots = {}
    scoop_head_temps = []
    hand_fluxes = []
    times = []
    
    # Snapshot times
    snap_times = [0, 5, 10, 30, 60, 120]
    snap_set = set(snap_times)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/cfd_{design}", exist_ok=True)
    
    # BC masks
    is_hand = (bc_type == 1)
    is_ice = (bc_type == 2)
    is_air = (bc_type == 3)
    is_interior = (bc_type == 0) | fluid_mask | (solid_mask & ~is_hand & ~is_ice & ~is_air)
    
    # Laplacian helper (3D, periodic in x,y; no-flux at z boundaries)
    def laplacian3d(f, dx, dy, dz):
        lap = torch.zeros_like(f)
        # Interior
        lap[1:-1, 1:-1, 1:-1] = (
            (f[1:-1, 1:-1, 2:] - 2*f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / dx**2 +
            (f[1:-1, 2:, 1:-1] - 2*f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / dy**2 +
            (f[2:, 1:-1, 1:-1] - 2*f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / dz**2
        )
        # Boundaries: zero-gradient (Neumann)
        # z=0 and z=-1
        for face in [0, -1]:
            idx = slice(None) if face == 0 else slice(None)
            adj = 1 if face == 0 else -2
            # Just copy neighbor (Neumann BC)
            pass  # handled implicitly by padding
        
        # Pad with Neumann BC (reflect)
        fp = torch.nn.functional.pad(f.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='replicate').squeeze(0).squeeze(0)
        lap = (
            (fp[1:-1, 1:-1, 2:] - 2*fp[1:-1, 1:-1, 1:-1] + fp[1:-1, 1:-1, :-2]) / dx**2 +
            (fp[1:-1, 2:, 1:-1] - 2*fp[1:-1, 1:-1, 1:-1] + fp[1:-1, :-2, 1:-1]) / dy**2 +
            (fp[2:, 1:-1, 1:-1] - 2*fp[1:-1, 1:-1, 1:-1] + fp[:-2, 1:-1, 1:-1]) / dz**2
        )
        return lap
    
    # Gradient helper
    def grad_z(f, dz):
        fp = torch.nn.functional.pad(f.unsqueeze(0).unsqueeze(0), (0,0,0,0,1,1), mode='replicate').squeeze(0).squeeze(0)
        return (fp[2:] - fp[:-2]) / (2 * dz)
    
    def grad_x(f, dx):
        fp = torch.nn.functional.pad(f.unsqueeze(0).unsqueeze(0), (1,1,0,0,0,0), mode='replicate').squeeze(0).squeeze(0)
        return (fp[1:-1] - fp[:-2]) / (2 * dx)  # wrong shape, fix
    
    # Simpler: central differences with replication padding
    def gradient_3d(f, dx, dy, dz):
        """Returns (dfdx, dfdy, dfdz) using central differences."""
        fp = torch.nn.functional.pad(f.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='replicate').squeeze(0).squeeze(0)
        dfdx = (fp[1:-1, 1:-1, 2:] - fp[1:-1, 1:-1, :-2]) / (2 * dx)
        dfdy = (fp[1:-1, 2:, 1:-1] - fp[1:-1, :-2, 1:-1]) / (2 * dy)
        dfdz = (fp[2:, 1:-1, 1:-1] - fp[:-2, 1:-1, 1:-1]) / (2 * dz)
        return dfdx, dfdy, dfdz
    
    # ---- Pressure solve setup (for projection method) ----
    # We need to solve: nabla²p = div(u*)/dt
    # Use Jacobi iterations (simple, GPU-friendly)
    def pressure_solve(div_u, dx, dy, dz, n_iter=30):
        """Solve Poisson equation for pressure using Jacobi iteration."""
        p = torch.zeros_like(div_u)
        dx2, dy2, dz2 = dx**2, dy**2, dz**2
        denom = 2.0 * (1/dx2 + 1/dy2 + 1/dz2)
        
        for _ in range(n_iter):
            pp = torch.nn.functional.pad(p.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='replicate').squeeze(0).squeeze(0)
            p_new = (
                (pp[1:-1, 1:-1, 2:] + pp[1:-1, 1:-1, :-2]) / dx2 +
                (pp[1:-1, 2:, 1:-1] + pp[1:-1, :-2, 1:-1]) / dy2 +
                (pp[2:, 1:-1, 1:-1] + pp[:-2, 1:-1, 1:-1]) / dz2 -
                div_u
            ) / denom
            p = p_new
        return p
    
    # ---- Main time loop ----
    t0 = time.time()
    for step in range(n_steps + 1):
        t = step * dt
        
        # Record snapshots
        t_int = int(round(t))
        if t_int in snap_set and t_int not in snapshots:
            snapshots[t_int] = T.detach().cpu().clone()
            if design == 'oil':
                snapshots[f"{t_int}_u"] = u.detach().cpu().clone()
                snapshots[f"{t_int}_w"] = w.detach().cpu().clone()
        
        # --- Record metrics ---
        # Scoop head center temperature
        # Head is at z~0, center of domain
        cx, cy, cz = NX//2, NY//2, 2  # near z=0 (scoop end)
        # Find the scoop head cell
        scoop_temp = T[cz, cy, cx].item()
        scoop_head_temps.append(scoop_temp)
        
        # Heat flux from hand (approximate: k * dT/dz at hand cells)
        hand_flux = 0.0
        if is_hand.any():
            # Average temperature gradient at hand surface
            dTdz = grad_z(T, dz)
            hand_k = k_field[is_hand].mean()
            hand_flux = (hand_k * dTdz[is_hand].abs().mean()).item()
        hand_fluxes.append(hand_flux)
        times.append(t)
        
        # --- Energy equation ---
        # dT/dt = alpha * laplacian(T) - (u.dT/dx + v.dT/dy + w.dT/dz)
        lap_T = laplacian3d(T, dx, dy, dz)
        
        if design == 'oil' and n_fluid > 0:
            dTdx, dTdy, dTdz = gradient_3d(T, dx, dy, dz)
            # Advective term only in fluid
            adv = u * dTdx + v * dTdy + w * dTdz
            adv = adv * fluid_mask  # zero in solid
            
            dTdt = alpha * lap_T - adv
        else:
            dTdt = alpha * lap_T
        
        T_new = T + dt * dTdt
        
        # Apply BCs
        T_new[is_hand] = T_HAND
        T_new[is_ice] = T_ICE
        # Air: convective BC approximated as relaxation toward ambient
        # q = h*(T_amb - T_surf) => dT/dt += h/(rho*cp*dl) * (T_amb - T)
        # Simplify: relax air BC cells toward ambient
        air_relax = H_AIR / (rho_cp[is_air] * min(dx, dy, dz))
        # Limit relaxation rate
        air_relax = torch.clamp(air_relax, max=1.0/dt*0.1)
        T_new[is_air] = T_new[is_air] + dt * air_relax * (T_AMB - T_new[is_air])
        
        T = T_new
        
        # --- Momentum equations (only in fluid region) ---
        if design == 'oil' and n_fluid > 0:
            # Buoyancy: Boussinesq approximation
            # Body force: rho * g * beta * (T - T_ref) in the -z direction (gravity)
            T_ref = T_ICE  # reference temperature
            buoy_z = -G * beta * (T - T_ref)  # negative z = downward
            
            # Velocity damping in solid (no-slip)
            # We enforce u=v=w=0 in solid cells
            solid_damp = solid_mask | (~fluid_mask)
            
            # du/dt = -dp/dx + nu * lap(u) - u.du/dx + buoy
            lap_u = laplacian3d(u, dx, dy, dz)
            lap_v = laplacian3d(v, dx, dy, dz)
            lap_w = laplacian3d(w, dx, dy, dz)
            
            duTdx, duTdy, duTdz = gradient_3d(u, dx, dy, dz)
            dvTdx, dvTdy, dvTdz = gradient_3d(v, dx, dy, dz)
            dwTdx, dwTdy, dwTdz = gradient_3d(w, dx, dy, dz)
            
            conv_u = u * duTdx + v * duTdy + w * duTdz
            conv_v = u * dvTdx + v * dvTdy + w * dvTdz
            conv_w = u * dwTdx + v * dwTdy + w * dwTdz
            
            # Intermediate velocity (without pressure gradient)
            u_star = u + dt * (nu * lap_u - conv_u)
            v_star = v + dt * (nu * lap_v - conv_v)
            w_star = w + dt * (nu * lap_w - conv_w + buoy_z)
            
            # Zero velocity in solid
            u_star[solid_damp] = 0
            v_star[solid_damp] = 0
            w_star[solid_damp] = 0
            
            # No-slip at tube walls (approximate: already zeroed in solid)
            
            # --- Projection step ---
            # Solve: lap(p) = div(u_star) / dt
            div_u = (
                (u_star[1:-1, 1:-1, 2:] - u_star[1:-1, 1:-1, :-2]) / (2*dx) +
                (v_star[1:-1, 2:, 1:-1] - v_star[1:-1, :-2, 1:-1]) / (2*dy) +
                (w_star[2:, 1:-1, 1:-1] - w_star[:-2, 1:-1, 1:-1]) / (2*dz)
            )
            # Pad div_u to full size
            div_full = torch.zeros_like(T)
            div_full[1:-1, 1:-1, 1:-1] = div_u
            
            p = pressure_solve(div_full / dt, dx, dy, dz, n_iter=20)
            
            # Correct velocity: u = u_star - dt * grad(p)
            dpdx, dpdy, dpdz = gradient_3d(p, dx, dy, dz)
            u = u_star - dt * dpdx
            v = v_star - dt * dpdy
            w = w_star - dt * dpdz
            
            # Zero velocity in solid
            u[solid_damp] = 0
            v[solid_damp] = 0
            w[solid_damp] = 0
        
        # Progress output
        if step % max(1, n_steps // 20) == 0:
            elapsed = time.time() - t0
            v_max = 0
            if design == 'oil' and n_fluid > 0:
                v_mag = torch.sqrt(u**2 + v**2 + w**2)
                v_max = v_mag[fluid_mask].max().item() if n_fluid > 0 else 0
            print(f"  t={t:6.1f}s [{step:5d}/{n_steps}] T_scoop={scoop_temp:7.2f}°C "
                  f"flux={hand_flux:.3f} W/m² v_max={v_max:.2e} m/s "
                  f"({elapsed:.1f}s elapsed)")
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # --- Save results ---
    results = {
        'design': design,
        'times': times,
        'scoop_head_temp': scoop_head_temps,
        'hand_flux': hand_fluxes,
        'grid': [NX, NY, NZ],
        'dt': dt,
        'n_steps': n_steps,
        'elapsed_wallclock': elapsed,
    }
    
    with open(f"{output_dir}/cfd_{design}/metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return T, u, v, w, snapshots, times, scoop_head_temps, hand_fluxes


def plot_results(output_dir='results'):
    """Generate all comparison plots."""
    
    # Load metrics for both designs
    metrics = {}
    for design in ['oil', 'copper']:
        path = f"{output_dir}/cfd_{design}/metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                metrics[design] = json.load(f)
    
    if not metrics:
        print("No results found!")
        return
    
    # --- Plot 1: Scoop head temperature over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'oil': '#2196F3', 'copper': '#FF9800'}
    labels = {'oil': 'Oil-filled Aluminum', 'copper': 'Solid Copper'}
    
    for design, m in metrics.items():
        ax.plot(m['times'], m['scoop_head_temp'], 
                label=labels.get(design, design), 
                color=colors.get(design, 'gray'), linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Scoop Head Temperature (°C)', fontsize=12)
    ax.set_title('Scoop Head Temperature: CFD with Natural Convection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='0°C')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cfd_head_temp_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/cfd_head_temp_comparison.png")
    
    # --- Plot 2: Heat flux over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for design, m in metrics.items():
        ax.plot(m['times'], m['hand_flux'],
                label=labels.get(design, design),
                color=colors.get(design, 'gray'), linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heat Flux (W/m²)', fontsize=12)
    ax.set_title('Hand-to-Scoop Heat Flux: CFD with Natural Convection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cfd_heat_flux_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/cfd_heat_flux_comparison.png")
    
    # --- Plot 3: Temperature cross-sections (z-r plane at y=center) ---
    for design in metrics:
        snap_dir = f"{output_dir}/cfd_{design}"
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        snap_times = [0, 5, 10, 30, 60, 120]
        for i, t_snap in enumerate(snap_times):
            # Load snapshot
            snap_file = f"{snap_dir}/T_t{t_snap}.pt"
            if os.path.exists(snap_file):
                T_snap = torch.load(snap_file, weights_only=False)
                # Take cross-section at y = NY/2
                cs = T_snap[:, NY//2, :].numpy()
                im = axes[i].imshow(cs, aspect='auto', cmap='coolwarm',
                                     vmin=-18, vmax=33, origin='lower',
                                     extent=[-DOMAIN_X/2*1000, DOMAIN_X/2*1000, 0, DOMAIN_Z*1000])
                axes[i].set_title(f't = {t_snap}s', fontsize=11)
                axes[i].set_xlabel('x (mm)')
                axes[i].set_ylabel('z (mm)')
            else:
                axes[i].text(0.5, 0.5, f'No data at t={t_snap}', ha='center', va='center')
                axes[i].set_title(f't = {t_snap}s')
        
        plt.suptitle(f'Temperature Field (r-z plane): {labels.get(design, design)}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cfd_{design}_temperature_fields.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/cfd_{design}_temperature_fields.png")
    
    # --- Plot 4: Oil velocity field (if oil design) ---
    oil_snap_u = f"{output_dir}/cfd_oil/u_t30.pt"
    oil_snap_w = f"{output_dir}/cfd_oil/w_t30.pt"
    if os.path.exists(oil_snap_u) and os.path.exists(oil_snap_w):
        u_snap = torch.load(oil_snap_u, weights_only=False)
        w_snap = torch.load(oil_snap_w, weights_only=False)
        T_snap = torch.load(f"{output_dir}/cfd_oil/T_t30.pt", weights_only=False)
        
        cs_T = T_snap[:, NY//2, :].numpy()
        cs_u = u_snap[:, NY//2, :].numpy()
        cs_w = w_snap[:, NY//2, :].numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Temperature
        im1 = ax1.imshow(cs_T, aspect='auto', cmap='coolwarm', vmin=-18, vmax=33,
                          origin='lower', extent=[-DOMAIN_X/2*1000, DOMAIN_X/2*1000, 0, DOMAIN_Z*1000])
        ax1.set_title('Temperature (°C) at t=30s', fontsize=12)
        plt.colorbar(im1, ax=ax1)
        
        # Velocity vectors (subsampled)
        step = 3
        z_coords = np.arange(0, NZ, step) * (DOMAIN_Z/NZ) * 1000
        x_coords = np.arange(0, NX, step) * (DOMAIN_X/NX) * 1000 - DOMAIN_X/2*1000
        Zq, Xq = np.meshgrid(z_coords, x_coords, indexing='ij')
        
        speed = np.sqrt(cs_u[::step, ::step]**2 + cs_w[::step, ::step]**2)
        # Scale arrows for visibility
        scale = max(speed.max(), 1e-10)
        ax2.quiver(Xq, Zq, cs_u[::step, ::step]/scale, cs_w[::step, ::step]/scale,
                    scale=20, width=0.003, alpha=0.7)
        ax2.imshow(cs_T, aspect='auto', cmap='coolwarm', vmin=-18, vmax=33,
                    origin='lower', extent=[-DOMAIN_X/2*1000, DOMAIN_X/2*1000, 0, DOMAIN_Z*1000],
                    alpha=0.3)
        ax2.set_title('Oil Velocity Vectors at t=30s', fontsize=12)
        ax2.set_xlabel('x (mm)')
        ax2.set_ylabel('z (mm)')
        
        plt.suptitle('Oil-Filled Handle: Natural Convection at t=30s', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cfd_oil_velocity_field.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/cfd_oil_velocity_field.png")


def save_snapshots(T, u, v, w, t, design, output_dir):
    """Save snapshot tensors."""
    d = f"{output_dir}/cfd_{design}"
    os.makedirs(d, exist_ok=True)
    torch.save(T.cpu(), f"{d}/T_t{t}.pt")
    if design == 'oil':
        torch.save(u.cpu(), f"{d}/u_t{t}.pt")
        torch.save(w.cpu(), f"{d}/w_t{t}.pt")


if __name__ == '__main__':
    import sys
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        prop = torch.cuda.get_device_properties(0)
        print(f"Memory: {prop.total_mem / 1e9:.1f} GB")
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    design = sys.argv[2] if len(sys.argv) > 2 else 'both'
    
    if design in ('oil', 'both'):
        print("\n" + "="*60)
        print("Running OIL-FILLED design simulation")
        print("="*60)
        T_oil, u_oil, v_oil, w_oil, snaps_oil, times_oil, st_oil, hf_oil = solve_cfd(
            design='oil', device=device, sim_time=120.0, output_dir=output_dir
        )
        # Save key snapshots
        for t_snap in [0, 5, 10, 30, 60, 120]:
            if t_snap in snaps_oil:
                save_snapshots(snaps_oil[t_snap], snaps_oil.get(f"{t_snap}_u", torch.zeros_like(T_oil)),
                              torch.zeros_like(T_oil), snaps_oil.get(f"{t_snap}_w", torch.zeros_like(T_oil)),
                              t_snap, 'oil', output_dir)
    
    if design in ('copper', 'both'):
        print("\n" + "="*60)
        print("Running SOLID COPPER design simulation")
        print("="*60)
        T_cu, u_cu, v_cu, w_cu, snaps_cu, times_cu, st_cu, hf_cu = solve_cfd(
            design='copper', device=device, sim_time=120.0, output_dir=output_dir
        )
        for t_snap in [0, 5, 10, 30, 60, 120]:
            if t_snap in snaps_cu:
                save_snapshots(snaps_cu[t_snap], torch.zeros_like(T_cu),
                              torch.zeros_like(T_cu), torch.zeros_like(T_cu),
                              t_snap, 'copper', output_dir)
    
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    plot_results(output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for design_name in ['oil', 'copper']:
        path = f"{output_dir}/cfd_{design_name}/metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            print(f"\n{design_name.upper()}:")
            print(f"  Time steps: {m['n_steps']:,}")
            print(f"  dt: {m['dt']*1000:.3f} ms")
            print(f"  Final scoop temp: {m['scoop_head_temp'][-1]:.1f}°C")
            print(f"  Time to 0°C: ", end="")
            found = False
            for t, T_val in zip(m['times'], m['scoop_head_temp']):
                if T_val >= 0:
                    print(f"{t:.1f}s")
                    found = True
                    break
            if not found:
                print("did not reach 0°C")
            print(f"  Final heat flux: {m['hand_flux'][-1]:.3f} W/m²")
