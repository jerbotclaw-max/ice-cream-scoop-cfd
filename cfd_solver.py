#!/usr/bin/env python3
"""
3D Conjugate Heat Transfer CFD Solver for Ice Cream Scoop Comparison.

Solves full Navier-Stokes + Energy equations on a 3D Cartesian grid.
- Oil region: Incompressible NS with Boussinesq buoyancy (natural convection)
- Solid regions: Pure conduction
- Conjugate heat transfer at fluid-solid interfaces

Uses projection method (Chorin) for pressure-velocity coupling.
GPU-accelerated via PyTorch.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, time, os, sys

# ============================================================
# Material Properties
# ============================================================
K_AL = 205.0; RHO_AL = 2700.0; CP_AL = 900.0
K_CU = 401.0; RHO_CU = 8960.0; CP_CU = 385.0
K_OIL = 0.15; RHO_OIL = 850.0; CP_OIL = 1670.0
MU_OIL = 0.08; BETA_OIL = 6.4e-4  # thermal expansion
G_GRAV = 9.81

T_HAND = 33.0; T_ICE = -18.0; T_AMB = 22.0; H_AIR = 10.0

# Domain: x,y span the handle tube diameter, z goes from scoop to hand
DOMAIN_X = 0.06   # 60mm 
DOMAIN_Y = 0.06
DOMAIN_Z = 0.20   # 200mm total

HANDLE_OD = 0.020; HANDLE_ID = 0.018; HANDLE_LEN = 0.150
SCOOP_R = 0.025; SCOOP_WALL = 0.002
GRIP_START = 0.050  # hand grips from z=50mm to z=150mm

NX, NY, NZ = 48, 48, 160

def pad3d(f, scheme='replicate'):
    """Pad 3D tensor with 1 cell on each face."""
    return torch.nn.functional.pad(f.unsqueeze(0).unsqueeze(0),
                                   (1,1,1,1,1,1), mode=scheme).squeeze(0).squeeze(0)

def laplacian(f, dx, dy, dz):
    """3D Laplacian with Neumann BC (zero-gradient at boundaries)."""
    fp = pad3d(f)
    return (
        (fp[1:-1,1:-1,2:] - 2*fp[1:-1,1:-1,1:-1] + fp[1:-1,1:-1,:-2]) / dx**2 +
        (fp[1:-1,2:,1:-1] - 2*fp[1:-1,1:-1,1:-1] + fp[1:-1,:-2,1:-1]) / dy**2 +
        (fp[2:,1:-1,1:-1] - 2*fp[1:-1,1:-1,1:-1] + fp[:-2,1:-1,1:-1]) / dz**2
    )

def grad_x(f, dx):
    fp = pad3d(f)
    return (fp[1:-1,1:-1,2:] - fp[1:-1,1:-1,:-2]) / (2*dx)

def grad_y(f, dy):
    fp = pad3d(f)
    return (fp[1:-1,2:,1:-1] - fp[1:-1,:-2,1:-1]) / (2*dy)

def grad_z(f, dz):
    fp = pad3d(f)
    return (fp[2:,1:-1,1:-1] - fp[:-2,1:-1,1:-1]) / (2*dz)

def div3d(fx, fy, fz, dx, dy, dz):
    """Divergence of vector field."""
    return grad_x(fx, dx) + grad_y(fy, dy) + grad_z(fz, dz)

def pressure_poisson(p, div_u, dx, dy, dz, n_iter=25):
    """Jacobi iteration for pressure Poisson equation."""
    dx2, dy2, dz2 = dx**2, dy**2, dz**2
    denom = 2.0*(1/dx2 + 1/dy2 + 1/dz2)
    p_data = p.clone()
    for _ in range(n_iter):
        pp = pad3d(p_data)
        p_new = (
            (pp[1:-1,1:-1,2:] + pp[1:-1,1:-1,:-2]) / dx2 +
            (pp[1:-1,2:,1:-1] + pp[1:-1,:-2,1:-1]) / dy2 +
            (pp[2:,1:-1,1:-1] + pp[:-2,1:-1,1:-1]) / dz2 -
            div_u
        ) / denom
        p_data = p_new
    return p_data

def build_masks(nx, ny, nz, dx, dy, dz, design):
    """Build material and BC masks."""
    x = (torch.arange(nx, dtype=torch.float32) + 0.5) * dx - DOMAIN_X/2
    y = (torch.arange(ny, dtype=torch.float32) + 0.5) * dy - DOMAIN_Y/2
    z = (torch.arange(nz, dtype=torch.float32) + 0.5) * dz
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)

    fluid = torch.zeros((nz,ny,nx), dtype=torch.bool)
    solid = torch.zeros((nz,ny,nx), dtype=torch.bool)
    k = torch.zeros((nz,ny,nx), dtype=torch.float32)
    rho_cp = torch.zeros((nz,ny,nx), dtype=torch.float32)

    # Handle tube: z in [0, HANDLE_LEN], r < HANDLE_OD/2
    in_handle = (Z >= 0) & (Z < HANDLE_LEN) & (R < HANDLE_OD/2)
    # Scoop head: hemisphere r < SCOOP_R, z < SCOOP_R, thin shell
    in_hemi = (Z < SCOOP_R) & (R**2 + Z**2 < SCOOP_R**2)
    in_shell = in_hemi & (torch.sqrt(R**2 + Z**2) > SCOOP_R - SCOOP_WALL)

    if design == 'oil':
        in_tube_wall = in_handle & (R >= HANDLE_ID/2)
        in_oil = in_handle & (R < HANDLE_ID/2)
        al = in_tube_wall | (in_shell & (Z < SCOOP_R))
        solid[al] = True; k[al] = K_AL; rho_cp[al] = RHO_AL * CP_AL
        fluid[in_oil] = True; k[in_oil] = K_OIL; rho_cp[in_oil] = RHO_OIL * CP_OIL
    else:  # copper
        cu = in_handle | (in_shell & (Z < SCOOP_R))
        solid[cu] = True; k[cu] = K_CU; rho_cp[cu] = RHO_CU * CP_CU

    # Air cells (outside scoop)
    air = ~fluid & ~solid
    k[air] = 0.026; rho_cp[air] = 1.225 * 1005

    # BC masks
    grip_z = (Z >= GRIP_START) & (Z < HANDLE_LEN)
    handle_outer = in_handle & (R >= HANDLE_OD/2 - 2*dx) & grip_z
    scoop_bottom = in_shell & (Z < 3*dz) & (R < SCOOP_R*0.8)
    scoop_air_surf = in_shell & ~scoop_bottom

    bc_hand = handle_outer
    bc_ice = scoop_bottom
    bc_air = scoop_air_surf | (air & (
        ((Z < HANDLE_LEN) & (R < HANDLE_OD/2 + 2*dx)) |
        ((Z < SCOOP_R*1.5) & (R < SCOOP_R + 2*dx))
    ))

    return fluid, solid, k, rho_cp, bc_hand, bc_ice, bc_air, (X,Y,Z,R)

def solve(design='oil', device='cuda', sim_time=120.0, out_dir='results'):
    dx, dy, dz = DOMAIN_X/NX, DOMAIN_Y/NY, DOMAIN_Z/NZ
    
    print(f"\n{'='*60}")
    print(f"CFD SOLVER: {design.upper()}")
    print(f"{'='*60}")
    print(f"Grid: {NX}x{NY}x{NZ} = {NX*NY*NZ:,} cells  ({dx*1000:.1f} x {dy*1000:.1f} x {dz*1000:.1f} mm cells)")

    fluid, solid, k, rho_cp, bc_hand, bc_ice, bc_air, (X,Y,Z,R) = \
        build_masks(NX, NY, NZ, dx, dy, dz, design)
    fluid = fluid.to(device); solid = solid.to(device)
    k = k.to(device); rho_cp = rho_cp.to(device)
    bc_hand = bc_hand.to(device); bc_ice = bc_ice.to(device); bc_air = bc_air.to(device)
    X_dev = X.to(device); Z_dev = Z.to(device)

    alpha = k / rho_cp
    nu = MU_OIL / RHO_OIL

    print(f"Fluid cells: {fluid.sum():,}, Solid cells: {solid.sum():,}")
    
    # Time step
    alpha_max = alpha.max().item()
    dt_diff = 0.2 * min(dx,dy,dz)**2 / max(alpha_max, 1e-8)
    dt_visc = 0.2 * min(dx,dy,dz)**2 / nu
    dt_buoy = np.sqrt(min(dx,dy,dz) / (G_GRAV * BETA_OIL * 50))
    dt = min(dt_diff, dt_visc, dt_buoy, 5e-4)
    n_steps = max(int(sim_time / dt) + 1, 1000)
    dt = sim_time / n_steps
    
    print(f"alpha_max={alpha_max:.3e}, nu={nu:.3e}")
    print(f"dt={dt:.3e}s ({dt*1000:.3f}ms), steps={n_steps:,}")
    
    # Rayleigh number
    alpha_oil = K_OIL / (RHO_OIL * CP_OIL)
    Ra = G_GRAV * BETA_OIL * abs(T_HAND - T_ICE) * HANDLE_ID**3 / (nu * alpha_oil)
    print(f"Rayleigh number: {Ra:.0f}")

    # Fields
    T = torch.full((NZ,NY,NX), T_ICE, dtype=torch.float32, device=device)
    u = torch.zeros((NZ,NY,NX), dtype=torch.float32, device=device)
    v = torch.zeros((NZ,NY,NX), dtype=torch.float32, device=device)
    w = torch.zeros((NZ,NY,NX), dtype=torch.float32, device=device)

    snap_times = {0, 5, 10, 30, 60, 120}
    snapshots = {}
    scoop_temps = []; hand_fluxes = []; times_list = []

    os.makedirs(f"{out_dir}/cfd_{design}", exist_ok=True)

    t0 = time.time()
    scoop_cx, scoop_cy, scoop_cz = NX//2, NY//2, 2

    for step in range(n_steps + 1):
        t = step * dt

        # Snapshot
        t_int = int(t)
        if t_int in snap_times and t_int not in snapshots:
            snapshots[t_int] = T.detach().cpu().clone()
            if design == 'oil':
                snapshots[f"u_{t_int}"] = u.detach().cpu().clone()
                snapshots[f"w_{t_int}"] = w.detach().cpu().clone()

        # Metrics
        scoop_temps.append(T[scoop_cz, scoop_cy, scoop_cx].item())
        times_list.append(t)
        if bc_hand.any():
            dTdz = grad_z(T, dz)
            hk = k[bc_hand].mean()
            hand_fluxes.append((hk * dTdz[bc_hand].abs().mean()).item())
        else:
            hand_fluxes.append(0.0)

        if step % (n_steps//10) == 0:
            v_max = 0.0
            if design == 'oil' and fluid.any():
                vm = torch.sqrt(u**2 + v**2 + w**2)
                v_max = vm[fluid].max().item() if fluid.any() else 0.0
            print(f"  t={t:6.1f}s [{step:6d}/{n_steps}] "
                  f"T_head={scoop_temps[-1]:7.2f}°C flux={hand_fluxes[-1]:.3f} W/m² "
                  f"v_max={v_max:.2e} m/s")

        # ---- Energy equation ----
        lap_T = laplacian(T, dx, dy, dz)
        alpha_field = alpha  # already on device

        if design == 'oil' and fluid.any():
            dTdx = grad_x(T, dx); dTdy = grad_y(T, dy); dTdz_g = grad_z(T, dz)
            conv = (u * dTdx + v * dTdy + w * dTdz_g) * fluid.float()
            dTdt = alpha_field * lap_T - conv
        else:
            dTdt = alpha_field * lap_T

        T = T + dt * dTdt

        # BC: Dirichlet
        T = torch.where(bc_hand, torch.tensor(T_HAND, device=device), T)
        T = torch.where(bc_ice, torch.tensor(T_ICE, device=device), T)

        # BC: Air convection
        if bc_air.any():
            # dT/dt += h*k/(rho*cp) / dx * (T_amb - T)
            coeff = H_AIR * k / (rho_cp * dz)
            coeff = torch.where(bc_air, coeff, torch.zeros_like(coeff))
            coeff = torch.clamp(coeff, max=1.0/dt * 0.2)
            T = T + dt * coeff * (T_AMB - T)

        # ---- Momentum (NS only in fluid) ----
        if design == 'oil' and fluid.any():
            # Buoyancy: body force in z direction (Boussinesq)
            buoy = -G_GRAV * BETA_OIL * (T - T_ICE)

            # Viscous diffusion
            lap_u_ = laplacian(u, dx, dy, dz)
            lap_v_ = laplacian(v, dx, dy, dz)
            lap_w_ = laplacian(w, dx, dy, dz)

            # Advection
            u_grad_u = u * grad_x(u,dx) + v * grad_y(u,dy) + w * grad_z(u,dz)
            u_grad_v = u * grad_x(v,dx) + v * grad_y(v,dy) + w * grad_z(v,dz)
            u_grad_w = u * grad_x(w,dx) + v * grad_y(w,dy) + w * grad_z(w,dz)

            # Tentative velocity
            u_star = u + dt * (nu * lap_u_ - u_grad_u)
            v_star = v + dt * (nu * lap_v_ - u_grad_v)
            w_star = w + dt * (nu * lap_w_ - u_grad_w + buoy)

            # Zero in solid
            mask = (~fluid).float()
            u_star = u_star * fluid.float()
            v_star = v_star * fluid.float()
            w_star = w_star * fluid.float()

            # Projection
            div_u = div3d(u_star, v_star, w_star, dx, dy, dz)
            p = pressure_poisson(torch.zeros_like(T), div_u, dx, dy, dz, n_iter=20)
            u = u_star - dt * grad_x(p, dx)
            v = v_star - dt * grad_y(p, dy)
            w = w_star - dt * grad_z(p, dz)

            u = u * fluid.float(); v = v * fluid.float(); w = w * fluid.float()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save snapshots
    for t_snap in snap_times:
        if t_snap in snapshots:
            torch.save(snapshots[t_snap].cpu(), f"{out_dir}/cfd_{design}/T_{t_snap}.pt")
            if design == 'oil':
                uk = snapshots.get(f"u_{t_snap}")
                wk = snapshots.get(f"w_{t_snap}")
                if uk is not None:
                    torch.save(uk.cpu(), f"{out_dir}/cfd_{design}/u_{t_snap}.pt")
                if wk is not None:
                    torch.save(wk.cpu(), f"{out_dir}/cfd_{design}/w_{t_snap}.pt")

    # Save metrics
    results = {
        'design': design,
        'times': times_list,
        'scoop_temp': scoop_temps,
        'hand_flux': hand_fluxes,
        'n_steps': n_steps,
        'dt': dt,
        'Ra': Ra,
        'elapsed': elapsed,
    }
    with open(f"{out_dir}/cfd_{design}/metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    return times_list, scoop_temps, hand_fluxes, snapshots, T

def plot_results(out_dir='results'):
    """Generate comparison plots."""
    metrics = {}
    for d in ['oil', 'copper']:
        p = f"{out_dir}/cfd_{d}/metrics.json"
        if os.path.exists(p):
            with open(p) as f:
                metrics[d] = json.load(f)

    if not metrics:
        print("No results!"); return

    colors = {'oil': '#1976D2', 'copper': '#FF6D00'}
    labels = {'oil': 'Oil-filled Aluminum', 'copper': 'Solid Copper'}

    # 1. Scoop head temperature
    fig, ax = plt.subplots(figsize=(10,6))
    for d, m in metrics.items():
        ax.plot(m['times'], m['scoop_temp'], label=labels.get(d,d),
                color=colors.get(d,'gray'), lw=2)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Scoop Head Temperature (°C)', fontsize=12)
    ax.set_title('CFD: Scoop Head Temperature vs Time (Natural Convection)', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cfd_head_temp.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/cfd_head_temp.png")

    # 2. Heat flux
    fig, ax = plt.subplots(figsize=(10,6))
    for d, m in metrics.items():
        ax.plot(m['times'], m['hand_flux'], label=labels.get(d,d),
                color=colors.get(d,'gray'), lw=2)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Heat Flux (W/m²)')
    ax.set_title('CFD: Hand-to-Scoop Heat Flux', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cfd_heat_flux.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/cfd_heat_flux.png")

    # 3. Temperature cross-sections (z-x plane at y=center)
    for d, m in metrics.items():
        snap_dir = f"{out_dir}/cfd_{d}"
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for i, t_snap in enumerate([0, 5, 10, 30, 60, 120]):
            pth = f"{snap_dir}/T_{t_snap}.pt"
            if os.path.exists(pth):
                T = torch.load(pth, weights_only=False)
                # z-x plane at y_center
                cs = T[:, NY//2, :].numpy()
                im = axes[i].imshow(cs, aspect='auto', cmap='coolwarm',
                                     vmin=T_ICE, vmax=T_HAND, origin='lower',
                                     extent=[-DOMAIN_X/2*1000, DOMAIN_X/2*1000,
                                              0, DOMAIN_Z*1000])
                axes[i].set_title(f't={t_snap}s', fontsize=11)
                axes[i].set_xlabel('x (mm)'); axes[i].set_ylabel('z (mm)')
            else:
                axes[i].text(0.5, 0.5, f'No data', ha='center', va='center')
                axes[i].set_title(f't={t_snap}s')
        plt.suptitle(f'{labels.get(d,d)}: Temperature Field', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cfd_{d}_temp_field.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir}/cfd_{d}_temp_field.png")

    # 4. Oil velocity field
    u_pth = f"{out_dir}/cfd_oil/u_30.pt"
    w_pth = f"{out_dir}/cfd_oil/w_30.pt"
    T_pth = f"{out_dir}/cfd_oil/T_30.pt"
    if all(os.path.exists(p) for p in [u_pth, w_pth, T_pth]):
        u30 = torch.load(u_pth, weights_only=False)
        w30 = torch.load(w_pth, weights_only=False)
        T30 = torch.load(T_pth, weights_only=False)
        cs_T = T30[:, NY//2, :].numpy()
        cs_u = u30[:, NY//2, :].numpy()
        cs_w = w30[:, NY//2, :].numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        im1 = ax1.imshow(cs_T, aspect='auto', cmap='coolwarm', vmin=T_ICE, vmax=T_HAND,
                          origin='lower', extent=[-DOMAIN_X/2*1000, DOMAIN_X/2*1000, 0, DOMAIN_Z*1000])
        ax1.set_title('Temperature at t=30s'); plt.colorbar(im1, ax=ax1)
        ax1.set_xlabel('x (mm)'); ax1.set_ylabel('z (mm)')

        step = 4
        z_c = np.arange(0, NZ, step) * (DOMAIN_Z/NZ) * 1000
        x_c = np.arange(0, NX, step) * (DOMAIN_X/NX) * 1000 - DOMAIN_X/2*1000
        Zq, Xq = np.meshgrid(z_c, x_c, indexing='ij')
        speed = np.sqrt(cs_u[::step,::step]**2 + cs_w[::step,::step]**2)
        sc = max(speed.max(), 1e-12)
        ax2.quiver(Xq, Zq, cs_u[::step,::step]/sc, cs_w[::step,::step]/sc,
                    scale=18, width=0.003, alpha=0.8)
        ax2.imshow(cs_T, aspect='auto', cmap='coolwarm', vmin=T_ICE, vmax=T_HAND,
                    origin='lower', extent=[-DOMAIN_X/2*1000, DOMAIN_X/2*1000, 0, DOMAIN_Z*1000],
                    alpha=0.4)
        ax2.set_title('Oil Velocity Vectors at t=30s')
        ax2.set_xlabel('x (mm)'); ax2.set_ylabel('z (mm)')
        plt.suptitle('Oil-Filled Handle: Natural Convection', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cfd_oil_velocity.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir}/cfd_oil_velocity.png")

    # 5. Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for d, m in metrics.items():
        print(f"\n{labels.get(d,d).upper()}:")
        print(f"  Time to 0°C: ", end="")
        found = False
        for tt, Tv in zip(m['times'], m['scoop_temp']):
            if Tv >= 0:
                print(f"{tt:.1f}s"); found = True; break
        if not found: print("did not reach 0°C")
        print(f"  Final temp: {m['scoop_temp'][-1]:.1f}°C")
        print(f"  Peak heat flux: {max(m['hand_flux']):.3f} W/m²")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    run_design = sys.argv[2] if len(sys.argv) > 2 else 'both'

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(out_dir, exist_ok=True)

    if run_design in ('oil', 'both'):
        solve('oil', device, 120.0, out_dir)

    if run_design in ('copper', 'both'):
        solve('copper', device, 120.0, out_dir)

    print("\nGenerating plots...")
    plot_results(out_dir)
    print("Done!")
