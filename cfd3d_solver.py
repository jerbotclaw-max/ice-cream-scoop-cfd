#!/usr/bin/env python3
"""
3D Conjugate Heat Transfer CFD Solver for Ice Cream Scoop
==========================================================
Solves incompressible Navier-Stokes + energy equation with:
- Projection method for pressure-velocity coupling
- Boussinesq approximation for natural convection in oil
- Immersed boundary method (marker cells) for geometry
- GPU acceleration via PyTorch CUDA

Runs on NVIDIA 4090.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import argparse
import os

# ============================================================================
# MATERIAL PROPERTIES
# ============================================================================
MATERIALS = {
    'aluminum': {'k': 205.0, 'rho': 2700.0, 'cp': 900.0, 'mu': 1e30, 'beta': 0.0},
    'copper':   {'k': 401.0, 'rho': 8960.0, 'cp': 385.0, 'mu': 1e30, 'beta': 0.0},
    'oil':      {'k': 0.15,  'rho': 850.0,  'cp': 1670.0, 'mu': 0.08, 'beta': 0.00064},
    'air':      {'k': 0.026, 'rho': 1.2,    'cp': 1005.0, 'mu': 1.8e-5, 'beta': 0.00343},
}

# Thermal diffusivity
def alpha(mat):
    p = MATERIALS[mat]
    return p['k'] / (p['rho'] * p['cp'])

# ============================================================================
# GEOMETRY SETUP
# ============================================================================
class ScoopGeometry:
    """Builds marker fields for the ice cream scoop on a 3D Cartesian grid."""
    
    def __init__(self, nx, ny, nz, dx, dy, dz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        
        # Cell type: 0=air, 1=oil, 2=aluminum, 3=copper, 4=ice_cream
        self.cell_type = np.zeros((nx, ny, nz), dtype=np.int32)
        
        # Boundary condition flags
        self.bc_hand = np.zeros((nx, ny, nz), dtype=bool)      # T=33C
        self.bc_icecream = np.zeros((nx, ny, nz), dtype=bool)   # T=-18C
        self.bc_ambient = np.zeros((nx, ny, nz), dtype=bool)    # convective
        
    def build_oil_design(self):
        """Design A: Aluminum tube with mineral oil fill + aluminum head."""
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        
        # Coordinates (cell centers)
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        z = (np.arange(nz) + 0.5) * dz
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Center of scoop in xy
        cx = nx * dx / 2
        cy = ny * dy / 2
        
        # Radial distance from axis
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # --- HANDLE REGION (z from 0 to 0.150m) ---
        handle_mask = Z <= 0.150
        
        # Handle outer radius = 10mm, inner radius = 9mm
        r_outer = 0.010  # 10mm
        r_inner = 0.009  # 9mm
        
        # Aluminum wall (between r_inner and r_outer)
        wall_mask = handle_mask & (R >= r_inner) & (R <= r_outer)
        self.cell_type[wall_mask] = 2  # aluminum
        
        # Oil fill (inside r_inner)
        oil_mask = handle_mask & (R < r_inner)
        self.cell_type[oil_mask] = 1  # oil
        
        # --- SCOOP HEAD (hemisphere at z=0, D=50mm, wall=2mm) ---
        # Hemisphere of radius 25mm centered at z=0
        head_r_outer = 0.025  # 25mm
        head_r_inner = 0.023  # 23mm (2mm wall)
        head_center_z = 0.0
        
        # Distance from head center
        dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - head_center_z)**2)
        
        # Hemisphere: z <= 0 and within radius
        head_outer_mask = (Z <= 0) & (dist_head <= head_r_outer) & (dist_head >= head_r_inner)
        self.cell_type[head_outer_mask] = 2  # aluminum head wall
        
        # Interior of head (air/void inside hemisphere - treat as oil for coupling)
        head_interior_mask = (Z <= 0) & (dist_head < head_r_inner) & (Z > -head_r_inner)
        # Connect to handle oil - fill with oil
        head_interior_mask = head_interior_mask & (R < r_inner * 5)  # rough
        self.cell_type[head_interior_mask] = 1  # oil (connected to handle oil)
        
        # Actually, let's be simpler: fill the head interior with oil
        head_interior = (Z <= 0.001) & (dist_head < head_r_inner)
        self.cell_type[head_interior] = 1  # oil
        
        # --- BOUNDARY CONDITIONS ---
        # Hand grip: outer surface of handle, z from 50mm to 150mm
        hand_mask = handle_mask & (Z >= 0.050) & (Z <= 0.150) & \
                    (R >= r_outer - dx) & (R <= r_outer + dx)
        self.bc_hand[hand_mask] = True
        
        # Ice cream: bottom of scoop head (outer surface, lower hemisphere)
        icecream_mask = (Z <= -0.020) & (dist_head >= head_r_outer - dx) & \
                        (dist_head <= head_r_outer + dx)
        self.bc_icecream[icecream_mask] = True
        
        # Ambient: exposed outer surfaces not hand or ice cream
        # Outer surface of handle below hand zone
        ambient_lower_handle = handle_mask & (Z < 0.050) & \
                               (R >= r_outer - dx) & (R <= r_outer + dx)
        # Outer surface of head not in ice cream contact
        ambient_head = (Z > -0.005) & (Z <= 0.005) & \
                       (dist_head >= head_r_outer - dx) & (dist_head <= head_r_outer + dx)
        self.bc_ambient[ambient_lower_handle] = True
        self.bc_ambient[ambient_head] = True
        
    def build_copper_design(self):
        """Design B: Solid copper handle + copper head."""
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        z = (np.arange(nz) + 0.5) * dz
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        cx = nx * dx / 2
        cy = ny * dy / 2
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # --- SOLID COPPER HANDLE ---
        handle_mask = Z <= 0.150
        r_outer = 0.010
        
        copper_handle = handle_mask & (R <= r_outer)
        self.cell_type[copper_handle] = 3  # copper
        
        # --- COPPER SCOOP HEAD ---
        head_r_outer = 0.025
        head_r_inner = 0.023
        head_center_z = 0.0
        dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - head_center_z)**2)
        
        head_wall = (Z <= 0) & (dist_head <= head_r_outer) & (dist_head >= head_r_inner)
        self.cell_type[head_wall] = 3  # copper
        
        # Fill head interior with copper too (solid scoop)
        head_solid = (Z <= 0) & (dist_head < head_r_inner)
        self.cell_type[head_solid] = 3  # copper
        
        # --- BOUNDARY CONDITIONS (same as oil design) ---
        hand_mask = handle_mask & (Z >= 0.050) & (Z <= 0.150) & \
                    (R >= r_outer - dx) & (R <= r_outer + dx)
        self.bc_hand[hand_mask] = True
        
        icecream_mask = (Z <= -0.020) & \
                        (dist_head >= head_r_outer - dx) & (dist_head <= head_r_outer + dx)
        self.bc_icecream[icecream_mask] = True
        
        ambient_lower_handle = handle_mask & (Z < 0.050) & \
                               (R >= r_outer - dx) & (R <= r_outer + dx)
        ambient_head = (Z > -0.005) & (Z <= 0.005) & \
                       (dist_head >= head_r_outer - dx) & (dist_head <= head_r_outer + dx)
        self.bc_ambient[ambient_lower_handle] = True
        self.bc_ambient[ambient_head] = True


# ============================================================================
# CFD SOLVER
# ============================================================================
class CFD3DSolver:
    """3D incompressible Navier-Stokes + energy on GPU."""
    
    def __init__(self, geom, design_name='oil'):
        self.geom = geom
        self.design_name = design_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        nx, ny, nz = geom.nx, geom.ny, geom.nz
        self.nx, self.ny, self.nz = nx, ny, nz
        
        # Move cell types to GPU
        ct = torch.tensor(geom.cell_type, device=self.device)
        self.is_oil = (ct == 1).float()
        self.is_solid = ((ct == 2) | (ct == 3)).float()
        self.is_aluminum = (ct == 2).float()
        self.is_copper = (ct == 3).float()
        self.is_air = (ct == 0).float()
        self.is_fluid = self.is_oil  # only oil is fluid
        
        # BC masks
        self.bc_hand = torch.tensor(geom.bc_hand, device=self.device)
        self.bc_icecream = torch.tensor(geom.bc_icecream, device=self.device)
        self.bc_ambient = torch.tensor(geom.bc_ambient, device=self.device)
        
        # Grid spacing
        self.dx = geom.dx
        self.dy = geom.dy
        self.dz = geom.dz
        
        # Effective material properties (blended per cell)
        self._setup_materials()
        
        # Fields
        self.T = torch.full((nx, ny, nz), -18.0, device=self.device)  # Initial: -18C
        self.u = torch.zeros((nx, ny, nz), device=self.device)
        self.v = torch.zeros((nx, ny, nz), device=self.device)
        self.w = torch.zeros((nx, ny, nz), device=self.device)
        self.p = torch.zeros((nx, ny, nz), device=self.device)  # pressure
        
        # Time
        self.t = 0.0
        
        # Reference state for Boussinesq
        self.T_ref = 0.0  # reference temperature
        
        # Gravity (z is up, so negative)
        self.g = -9.81  # m/s²
        
    def _setup_materials(self):
        """Set per-cell thermal diffusivity and viscosity."""
        # Thermal conductivity field
        k_field = np.zeros_like(self.geom.cell_type, dtype=np.float32)
        rho_field = np.zeros_like(self.geom.cell_type, dtype=np.float32)
        cp_field = np.zeros_like(self.geom.cell_type, dtype=np.float32)
        mu_field = np.zeros_like(self.geom.cell_type, dtype=np.float32)
        beta_field = np.zeros_like(self.geom.cell_type, dtype=np.float32)
        
        ct = self.geom.cell_type
        for mat_name, mat_id in [('air', 0), ('oil', 1), ('aluminum', 2), ('copper', 3)]:
            m = MATERIALS[mat_name]
            mask = ct == mat_id
            k_field[mask] = m['k']
            rho_field[mask] = m['rho']
            cp_field[mask] = m['cp']
            mu_field[mask] = m['mu']
            beta_field[mask] = m['beta']
        
        self.k = torch.tensor(k_field, device=self.device)
        self.rho = torch.tensor(rho_field, device=self.device)
        self.cp = torch.tensor(cp_field, device=self.device)
        self.mu = torch.tensor(mu_field, device=self.device)
        self.beta = torch.tensor(beta_field, device=self.device)
        
        # Thermal diffusivity
        self.alpha = self.k / (self.rho * self.cp)
        
        # For numerical stability, ensure minimum values
        self.alpha = torch.clamp(self.alpha, min=1e-8)
        
    def _laplacian(self, f):
        """3D Laplacian using central differences with Neumann BC."""
        lap = torch.zeros_like(f)
        lap[1:-1, :, :] += (f[2:, :, :] - 2*f[1:-1, :, :] + f[:-2, :, :]) / self.dx**2
        lap[:, 1:-1, :] += (f[:, 2:, :] - 2*f[:, 1:-1, :] + f[:, :-2, :]) / self.dy**2
        lap[:, :, 1:-1] += (f[:, :, 2:] - 2*f[:, :, 1:-1] + f[:, :, :-2]) / self.dz**2
        # Neumann BC (zero gradient at boundaries)
        lap[0, :, :] = lap[1, :, :]
        lap[-1, :, :] = lap[-2, :, :]
        lap[:, 0, :] = lap[:, 1, :]
        lap[:, -1, :] = lap[:, -2, :]
        lap[:, :, 0] = lap[:, :, 1]
        lap[:, :, -1] = lap[:, :, -2]
        return lap
    
    def _grad_z(self, f):
        """Z-gradient with central differences."""
        g = torch.zeros_like(f)
        g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * self.dz)
        g[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / self.dz
        g[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / self.dz
        return g
    
    def _advect(self, f, u, v, w, dt):
        """Semi-Lagrangian advection."""
        # Backtrace
        x_idx = torch.arange(self.nx, device=self.device, dtype=torch.float32)
        y_idx = torch.arange(self.ny, device=self.device, dtype=torch.float32)
        z_idx = torch.arange(self.nz, device=self.device, dtype=torch.float32)
        
        # Simple upwind advection (more stable)
        dfdx = torch.zeros_like(f)
        dfdy = torch.zeros_like(f)
        dfdz = torch.zeros_like(f)
        
        dfdx[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * self.dx)
        dfdy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * self.dy)
        dfdz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * self.dz)
        
        return f - dt * (u * dfdx + v * dfdy + w * dfdz)
    
    def step(self, dt):
        """One time step of the coupled solver."""
        
        # ---- 1. Energy equation ----
        # dT/dt + u·∇T = α∇²T
        # Advection only in fluid
        T_adv = self._advect(self.T, self.u * self.is_fluid, 
                              self.v * self.is_fluid, self.w * self.is_fluid, dt)
        
        # Diffusion
        lap_T = self._laplacian(T_adv)
        T_new = T_adv + dt * self.alpha * lap_T
        
        # ---- 2. Apply temperature BCs ----
        # Hand grip: T = 33C
        T_new[self.bc_hand] = 33.0
        # Ice cream: T = -18C
        T_new[self.bc_icecream] = -18.0
        # Ambient: convective BC approximated as relaxation toward 22C
        amb_mask = self.bc_ambient
        if amb_mask.any():
            T_new[amb_mask] = T_new[amb_mask] + dt * 10.0 * (22.0 - T_new[amb_mask]) / \
                              (self.rho[amb_mask] * self.cp[amb_mask] * self.dx)
        
        # ---- 3. Momentum equation (only in fluid) ----
        # Boussinesq buoyancy: ρ' = -ρ₀β(T - T_ref)
        # Force only in z-direction (gravity)
        buoyancy = self.is_fluid * self.rho * self.beta * (T_new - self.T_ref) * self.g
        
        # u, v, w updates: semi-implicit
        # du/dt = -∇p/ρ + ν∇²u + f
        nu = self.mu / self.rho  # kinematic viscosity
        
        # Velocity prediction (without pressure)
        lap_u = self._laplacian(self.u)
        lap_v = self._laplacian(self.v)
        lap_w = self._laplacian(self.w)
        
        u_star = self.u + dt * nu * lap_u
        v_star = self.v + dt * nu * lap_v
        w_star = self.w + dt * (nu * lap_w + buoyancy / self.rho)
        
        # Zero velocity in solids
        u_star = u_star * self.is_fluid
        v_star = v_star * self.is_fluid
        w_star = w_star * self.is_fluid
        
        # ---- 4. Pressure projection ----
        # ∇²p = ρ/dt * ∇·u*
        div_u = torch.zeros_like(u_star)
        div_u[1:-1, :, :] += (u_star[2:, :, :] - u_star[:-2, :, :]) / (2 * self.dx)
        div_u[:, 1:-1, :] += (v_star[:, 2:, :] - v_star[:, :-2, :]) / (2 * self.dy)
        div_u[:, :, 1:-1] += (w_star[:, :, 2:] - w_star[:, :, :-2]) / (2 * self.dz)
        
        # Solve pressure Poisson equation (Jacobi iterations)
        p = self.p.clone()
        rho_avg = 850.0  # oil density for pressure solve
        rhs = rho_avg / dt * div_u
        
        for _ in range(50):  # Jacobi iterations
            lap_p = self._laplacian(p)
            p = p + 0.8 * (rhs - lap_p) / (2/self.dx**2 + 2/self.dy**2 + 2/self.dz**2)
            # Neumann BC for pressure
            p[0, :, :] = p[1, :, :]
            p[-1, :, :] = p[-2, :, :]
            p[:, 0, :] = p[:, 1, :]
            p[:, -1, :] = p[:, -2, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, :, -1] = p[:, :, -2]
        
        self.p = p
        
        # Velocity correction
        dpdx = torch.zeros_like(p)
        dpdy = torch.zeros_like(p)
        dpdz = torch.zeros_like(p)
        dpdx[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2 * self.dx)
        dpdy[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2 * self.dy)
        dpdz[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2 * self.dz)
        
        self.u = (u_star - dt / rho_avg * dpdx) * self.is_fluid
        self.v = (v_star - dt / rho_avg * dpdy) * self.is_fluid
        self.w = (w_star - dt / rho_avg * dpdz) * self.is_fluid
        
        # ---- 5. Update temperature ----
        self.T = T_new
        self.t += dt
        
    def get_scoop_head_temp(self):
        """Average temperature at the scoop head center."""
        # Head center region: near z=0, near axis
        ct = torch.tensor(self.geom.cell_type, device=self.device)
        head_mask = (ct == 2) | (ct == 3)  # aluminum or copper (head wall)
        # Only the scoop head part (z < 0.005m)
        z = (torch.arange(self.nz, device=self.device) + 0.5) * self.dz
        z_mask = z[None, None, :] < 0.005
        combined = head_mask & z_mask
        if combined.any():
            return self.T[combined].mean().item()
        return -18.0
    
    def get_heat_flux_hand(self):
        """Heat flux from hand into scoop (W)."""
        # Approximate: k * dT/dx at hand surface
        hand_cells = self.bc_hand
        if not hand_cells.any():
            return 0.0
        # Temperature gradient normal to surface (approximate with neighbor)
        T_hand = 33.0
        # Find cells just inside from hand BC
        T_inner = self.T.clone()
        # Average T of cells adjacent to hand boundary
        flux = 0.0
        count = 0
        idx = torch.where(hand_cells)
        for i in range(len(idx[0]) // 10):  # sample every 10th
            ix, iy, iz = idx[0][i*10], idx[1][i*10], idx[2][i*10]
            # Check inward neighbor
            R_inner = torch.sqrt(((ix.float() + 0.5) * self.dx - self.nx*self.dx/2)**2 + 
                                 ((iy.float() + 0.5) * self.dy - self.ny*self.dy/2)**2)
            if R_inner > self.dx:  # go inward
                direction = -1 if ix > self.nx//2 else 1
                nx_ = int(ix + direction)
                if 0 <= nx_ < self.nx:
                    dT = T_hand - self.T[nx_, iy, iz].item()
                    flux += self.k[ix, iy, iz].item() * dT / self.dx
                    count += 1
        if count > 0:
            return flux / count * len(idx[0])  # scale back up
        return 0.0


# ============================================================================
# POST-PROCESSING
# ============================================================================
def plot_results(results_a, results_b, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Temperature over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_a['time'], results_a['head_temp'], 'b-', label='Oil + Aluminum', linewidth=2)
    ax.plot(results_b['time'], results_b['head_temp'], 'r-', label='Solid Copper', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='0°C (freezing)')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Scoop Head Temperature (°C)', fontsize=14)
    ax.set_title('Ice Cream Scoop Head Temperature: Oil-Filled vs Solid Copper', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/head_temp_comparison.png', dpi=150)
    plt.close()
    
    # Print key metrics
    print("\n=== RESULTS SUMMARY ===")
    for name, res in [('Oil + Aluminum', results_a), ('Solid Copper', results_b)]:
        final_temp = res['head_temp'][-1]
        time_to_0 = None
        for i, t in enumerate(res['time']):
            if res['head_temp'][i] >= 0:
                time_to_0 = t
                break
        print(f"{name}:")
        print(f"  Final head temp: {final_temp:.1f}°C")
        print(f"  Time to 0°C: {time_to_0:.1f}s" if time_to_0 else "  Did not reach 0°C")
    
    # Save data as JSON
    data = {
        'design_a_oil_aluminum': {
            'time': results_a['time'],
            'head_temp': results_a['head_temp'],
        },
        'design_b_copper': {
            'time': results_b['time'],
            'head_temp': results_b['head_temp'],
        }
    }
    with open(f'{output_dir}/cfd3d_results.json', 'w') as f:
        json.dump(data, f, indent=2)


def plot_cross_section(solver, output_dir, design_name, timestamp):
    """Plot temperature cross-section through the axis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    T = solver.T.cpu().numpy()
    
    # X-Z plane through center (y = ny/2)
    ny2 = solver.ny // 2
    slice_xz = T[:, ny2, :]
    
    ax = axes[0]
    im = ax.imshow(slice_xz.T, origin='lower', aspect='auto', cmap='RdYlBu_r',
                   extent=[0, solver.nx * solver.dx * 1000, 0, solver.nz * solver.dz * 1000],
                   vmin=-20, vmax=35)
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Z (mm)', fontsize=12)
    ax.set_title(f'{design_name} - Temperature (t={timestamp:.0f}s)', fontsize=14)
    plt.colorbar(im, ax=ax, label='°C')
    
    # Velocity field (z-component) for oil design
    if solver.design_name == 'oil':
        w = solver.w.cpu().numpy()
        slice_w = w[:, ny2, :]
        ax2 = axes[1]
        im2 = ax2.imshow(slice_w.T, origin='lower', aspect='auto', cmap='RdBu',
                         extent=[0, solver.nx * solver.dx * 1000, 0, solver.nz * solver.dz * 1000],
                         vmin=-0.01, vmax=0.01)
        ax2.set_xlabel('X (mm)', fontsize=12)
        ax2.set_ylabel('Z (mm)', fontsize=12)
        ax2.set_title(f'{design_name} - Oil Velocity W (t={timestamp:.0f}s)', fontsize=14)
        plt.colorbar(im2, ax=ax2, label='m/s')
    else:
        ax2 = axes[1]
        ax2.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cross_section_{design_name}_t{timestamp:.0f}.png', dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def run_simulation(design='oil', sim_time=120.0, output_dir='results_3d'):
    """Run a single simulation case."""
    print(f"\n{'='*60}")
    print(f"Running simulation: {design}")
    print(f"{'='*60}")
    
    # Grid: 64x64x256 (manageable on GPU)
    nx, ny, nz = 64, 64, 256
    # Domain: ~30mm x 30mm x 200mm
    dx = 0.030 / nx   # ~0.47mm
    dy = 0.030 / ny
    dz = 0.200 / nz   # ~0.78mm
    
    print(f"Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} cells")
    print(f"Cell size: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")
    
    # Build geometry
    geom = ScoopGeometry(nx, ny, nz, dx, dy, dz)
    if design == 'oil':
        geom.build_oil_design()
    else:
        geom.build_copper_design()
    
    # Count cells
    oil_cells = np.sum(geom.cell_type == 1)
    al_cells = np.sum(geom.cell_type == 2)
    cu_cells = np.sum(geom.cell_type == 3)
    air_cells = np.sum(geom.cell_type == 0)
    print(f"Cells: oil={oil_cells}, aluminum={al_cells}, copper={cu_cells}, air={air_cells}")
    
    # Create solver
    solver = CFD3DSolver(geom, design_name=design)
    
    # Time stepping
    # CFL: dt < 0.5 * dx² / alpha_max
    alpha_max = max(alpha('aluminum'), alpha('copper'), alpha('oil'))
    dt_diff = 0.25 * min(dx, dy, dz)**2 / alpha_max
    # Also CFL for advection
    dt_adv = 0.5 * min(dx, dy, dz) / 0.01  # assume max velocity 1cm/s
    dt = min(dt_diff, dt_adv, 0.01)  # cap at 10ms
    print(f"Max alpha: {alpha_max:.2e}")
    print(f"Diffusion dt limit: {dt_diff:.4f}s")
    print(f"Advection dt limit: {dt_adv:.4f}s")
    print(f"Using dt = {dt:.4f}s")
    
    n_steps = int(sim_time / dt)
    print(f"Total steps: {n_steps}")
    
    # Results tracking
    times = []
    head_temps = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Snapshot times
    snap_times = [0, 5, 10, 30, 60, 120]
    
    t0 = time.time()
    for step in range(n_steps + 1):
        if step % 100 == 0:
            head_temp = solver.get_scoop_head_temp()
            times.append(solver.t)
            head_temps.append(head_temp)
            
            elapsed = time.time() - t0
            remaining = elapsed / max(step, 1) * (n_steps - step)
            print(f"Step {step}/{n_steps} | t={solver.t:.1f}s | "
                  f"Head T={head_temp:.1f}°C | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
            
            # Snapshot
            for st in snap_times:
                if abs(solver.t - st) < dt / 2:
                    plot_cross_section(solver, output_dir, design, solver.t)
        
        if step < n_steps:
            solver.step(dt)
    
    total_time = time.time() - t0
    print(f"\nSimulation complete in {total_time:.1f}s wall time")
    
    return {'time': times, 'head_temp': head_temps, 'solver': solver}


def main():
    parser = argparse.ArgumentParser(description='3D CFD Ice Cream Scoop Solver')
    parser.add_argument('--design', choices=['oil', 'copper', 'both'], default='both')
    parser.add_argument('--time', type=float, default=120.0)
    parser.add_argument('--output', default='results_3d')
    args = parser.parse_args()
    
    output_dir = args.output
    
    if args.design in ('oil', 'both'):
        res_a = run_simulation('oil', args.time, output_dir)
    else:
        res_a = None
    
    if args.design in ('copper', 'both'):
        res_b = run_simulation('copper', args.time, output_dir)
    else:
        res_b = None
    
    if res_a and res_b:
        plot_results(res_a, res_b, output_dir)
    
    print("\nDone! Results saved to", output_dir)


if __name__ == '__main__':
    main()
