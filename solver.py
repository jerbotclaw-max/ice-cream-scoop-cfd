#!/usr/bin/env python3
"""
2D Axisymmetric Transient Thermal Solver for Ice Cream Scoop Comparison.
GPU-accelerated via PyTorch. Fully vectorized.

Simplified physical model:
- Handle is a cylinder (solid copper OR aluminum tube with oil core)
- Scoop head is a solid metal disk at the end (not hollow - we care about 
  heat reaching the scooping surface)
- Heat flows: hand -> handle surface -> along handle axis -> into scoop head -> tip
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#  Material Properties 
K_AL = 205.0; RHO_AL = 2700.0; CP_AL = 900.0
K_CU = 401.0; RHO_CU = 8960.0; CP_CU = 385.0
K_OIL = 2.0; RHO_OIL = 880.0; CP_OIL = 1900.0

ALPHA_AL = K_AL / (RHO_AL * CP_AL)
ALPHA_CU = K_CU / (RHO_CU * CP_CU)
ALPHA_OIL = K_OIL / (RHO_OIL * CP_OIL)

T_HAND = 33.0
T_ICE = -18.0
T_AMB = 22.0

# Geometry [m]
HANDLE_OD = 0.020
HANDLE_ID = 0.018
HANDLE_LEN = 0.150
GRIP_LEN = 0.100
SCOOP_D = 0.050
SCOOP_THICKNESS = 0.004  # 4mm solid scoop head (not thin shell)

#  Grid 
NR = 80
NZ = 300
# Domain: R from 0 to scoop radius (25mm), Z from 0 to handle end
Z_DOMAIN = HANDLE_LEN + SCOOP_D  # 200mm
R_DOMAIN = SCOOP_D / 2 + 0.002   # 27mm
DR = R_DOMAIN / NR
DZ = Z_DOMAIN / NZ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Grid: {NR}x{NZ} | dr={DR*1e3:.3f}mm dz={DZ*1e3:.3f}mm")

DT = 0.0005
T_TOTAL = 120.0
N_STEPS = int(T_TOTAL / DT)


def build_fields(design):
    """Build alpha and k fields. Layout along Z:
    z=0 is the TIP of the scoop head (ice cream end)
    z=scoop_thickness is where scoop meets handle
    z=scoop_thickness..scoop_thickness+handle_len is the handle
    """
    r_idx = torch.arange(NR, device=device, dtype=torch.float32)
    z_idx = torch.arange(NZ, device=device, dtype=torch.float32)
    R, Z = torch.meshgrid(r_idx, z_idx, indexing='ij')

    scoop_z = max(1, int(SCOOP_THICKNESS / DZ))      # solid metal disk
    handle_z0 = scoop_z
    handle_z1 = min(NZ-1, int((SCOOP_THICKNESS + HANDLE_LEN) / DZ))
    od_r = max(1, int((HANDLE_OD / 2) / DR))
    id_r = max(1, int((HANDLE_ID / 2) / DR))
    scoop_r = max(1, int((SCOOP_D / 2) / DR))

    # Default: ambient air (low conductivity)
    alpha = torch.full((NR, NZ), 3e-6, device=device)  # air-like
    k = torch.full((NR, NZ), 0.03, device=device)

    if design == 'oil':
        # Scoop head: solid aluminum disk (radius = scoop_r, height = scoop_z)
        in_scoop = (Z < scoop_z) & (R < scoop_r)
        alpha[in_scoop] = ALPHA_AL
        k[in_scoop] = K_AL

        # Handle: aluminum tube wall (id_r to od_r) + oil core (0 to id_r)
        in_handle = (Z >= handle_z0) & (Z < handle_z1)
        in_wall = in_handle & (R >= id_r) & (R < od_r)
        in_oil = in_handle & (R < id_r)
        alpha[in_wall] = ALPHA_AL
        k[in_wall] = K_AL
        alpha[in_oil] = ALPHA_OIL
        k[in_oil] = K_OIL

        mat_k = K_AL
    else:
        # Solid copper everywhere
        in_scoop = (Z < scoop_z) & (R < scoop_r)
        alpha[in_scoop] = ALPHA_CU
        k[in_scoop] = K_CU

        in_handle = (Z >= handle_z0) & (Z < handle_z1) & (R < od_r)
        alpha[in_handle] = ALPHA_CU
        k[in_handle] = K_CU

        mat_k = K_CU

    info = {
        'scoop_z': scoop_z,
        'handle_z0': handle_z0,
        'handle_z1': handle_z1,
        'od_r': od_r,
        'scoop_r': scoop_r,
        'grip_z0': handle_z1 - int(GRIP_LEN / DZ),
        'grip_z1': handle_z1,
    }
    return alpha, k, info, R


def run_sim(design, label):
    alpha, k, info, R_phys = build_fields(design)
    T = torch.full((NR, NZ), T_ICE, dtype=torch.float32, device=device)

    a_max = alpha.max().item()
    dt_limit = 0.25 / (a_max * (1/DR**2 + 1/DZ**2))
    dt = min(DT, dt_limit * 0.9)
    n_steps = int(T_TOTAL / dt)
    print(f"  [{label}] dt={dt*1e3:.3f}ms (limit {dt_limit*1e3:.3f}ms), steps={n_steps}")

    # Monitor at scoop tip surface (r=0, z=0) - the business end
    # Also monitor scoop center (r=0, z=scoop_z//2)
    scoop_tip = (0, 0)
    scoop_mid = (0, info['scoop_z'] // 2)

    grip_z0 = info['grip_z0']
    grip_z1 = info['grip_z1']
    od_r = info['od_r']

    # Heat flux monitor at handle surface mid-grip
    flux_z = (grip_z0 + grip_z1) // 2
    flux_r = od_r - 1

    temp_hist = []
    flux_hist = []
    snapshots = {}
    snap_times = [0, 5, 10, 30, 60, 120]
    save_every = max(1, n_steps // 2000)

    # Precompute 1/r for r>0
    inv_r = torch.zeros(NR, device=device, dtype=torch.float32)
    inv_r[1:] = 1.0 / (torch.arange(1, NR, device=device, dtype=torch.float32) * DR)

    for step in range(n_steps):
        t = step * dt

        # Interior update (r>0, z interior)
        d2r = (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / (DR**2)
        d2z = (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / (DZ**2)
        dr_c = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2*DR)
        lap = d2r + d2z + inv_r[1:-1].unsqueeze(1) * dr_c
        T[1:-1, 1:-1] += dt * alpha[1:-1, 1:-1] * lap

        # Axis r=0 (symmetry: Laplacian = 2*d2r + d2z)
        d2r_ax = (T[1, 1:-1] - T[0, 1:-1]) / (DR**2)
        d2z_ax = (T[0, 2:] - 2*T[0, 1:-1] + T[0, :-2]) / (DZ**2)
        T[0, 1:-1] += dt * alpha[0, 1:-1] * (2*d2r_ax + d2z_ax)

        # BCs
        # Hand grip: 33C on handle outer surface
        T[max(0, od_r-1):od_r, grip_z0:grip_z1] = T_HAND
        # Ice cream contact at scoop tip (z=0)
        T[0:info['scoop_r'], 0:1] = T_ICE
        # Far field ambient
        T[NR-1, :] = T_AMB
        T[:, NZ-1] = T_AMB

        # Record
        if step % save_every == 0:
            tt = T[scoop_mid[0], scoop_mid[1]].item()
            temp_hist.append((t, tt))
            if 0 < flux_r < NR-1:
                dTdr = abs(T[flux_r, flux_z].item() - T[flux_r-1, flux_z].item()) / DR
                q = k[flux_r, flux_z].item() * dTdr
            else:
                q = 0.0
            flux_hist.append((t, q))

        for st in snap_times:
            if abs(t - st) < dt and st not in snapshots:
                snapshots[st] = T.cpu().numpy().copy()

        if step % (n_steps // 10) == 0:
            tt = T[scoop_mid[0], scoop_mid[1]].item()
            print(f"  [{label}] t={t:.1f}s  T_scoop_mid={tt:.2f}C")

    snapshots[120] = T.cpu().numpy().copy()

    return {
        'temp_hist': temp_hist,
        'flux_hist': flux_hist,
        'snapshots': snapshots,
        'info': info,
        'label': label,
    }


def plot_all(res_oil, res_cu):
    # 1) Scoop temp over time
    fig, ax = plt.subplots(figsize=(10, 6))
    to = [r[0] for r in res_oil['temp_hist']]
    To = [r[1] for r in res_oil['temp_hist']]
    tc = [r[0] for r in res_cu['temp_hist']]
    Tc = [r[1] for r in res_cu['temp_hist']]
    ax.plot(to, To, 'b-', lw=2, label='Oil-filled Aluminum')
    ax.plot(tc, Tc, 'r-', lw=2, label='Solid Copper')
    ax.axhline(0, color='gray', ls='--', alpha=0.5, label='0C (freezing)')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Scoop Head Temperature (C)', fontsize=13)
    ax.set_title('Scoop Head Warm-up: Oil-filled vs Solid Copper Handle', fontsize=14)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3); ax.set_xlim(0, 120)
    plt.tight_layout(); plt.savefig('results/scoop_temp_over_time.png', dpi=150); plt.close()
    print("-> results/scoop_temp_over_time.png")

    # 2) Heat flux
    fig, ax = plt.subplots(figsize=(10, 6))
    qo = [r[1] for r in res_oil['flux_hist']]
    qc = [r[1] for r in res_cu['flux_hist']]
    ax.plot(to, qo, 'b-', lw=2, label='Oil-filled Aluminum')
    ax.plot(tc, qc, 'r-', lw=2, label='Solid Copper')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Heat Flux from Hand (W/m2)', fontsize=13)
    ax.set_title('Hand to Scoop Heat Flux', fontsize=14)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3); ax.set_xlim(0, 120)
    plt.tight_layout(); plt.savefig('results/heat_flux_over_time.png', dpi=150); plt.close()
    print("-> results/heat_flux_over_time.png")

    # 3) Snapshots
    for name, res in [('oil', res_oil), ('copper', res_cu)]:
        for st in [0, 10, 30, 60, 120]:
            if st in res['snapshots']:
                fig, ax = plt.subplots(figsize=(5, 10))
                data = res['snapshots'][st]
                im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=-20, vmax=35,
                              origin='lower', extent=[0, R_DOMAIN*1e3, 0, Z_DOMAIN*1e3])
                ax.set_title(f"{res['label']} @ t={st}s", fontsize=12)
                ax.set_xlabel('r (mm)'); ax.set_ylabel('z (mm)')
                plt.colorbar(im, ax=ax, label='C')
                plt.tight_layout()
                plt.savefig(f'results/snap_{name}_t{st}s.png', dpi=150)
                plt.close()
                print(f"-> results/snap_{name}_t{st}s.png")

    # 4) Side-by-side comparisons
    for st in [10, 30, 60, 120]:
        if st in res_oil['snapshots'] and st in res_cu['snapshots']:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 10))
            for ax, data, lbl in [(a1, res_oil['snapshots'][st], 'Oil-filled Al'),
                                   (a2, res_cu['snapshots'][st], 'Solid Cu')]:
                im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=-20, vmax=35,
                              origin='lower', extent=[0, R_DOMAIN*1e3, 0, Z_DOMAIN*1e3])
                ax.set_title(f'{lbl} @ {st}s', fontsize=12)
                ax.set_xlabel('r (mm)'); ax.set_ylabel('z (mm)')
                plt.colorbar(im, ax=ax, label='C')
            plt.suptitle(f'Temperature Field @ t={st}s', fontsize=14, y=0.98)
            plt.tight_layout()
            plt.savefig(f'results/compare_t{st}s.png', dpi=150)
            plt.close()
            print(f"-> results/compare_t{st}s.png")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for label, res in [('Oil-filled Al', res_oil), ('Solid Copper', res_cu)]:
        th = res['temp_hist']
        t0 = next((t for t, T in th if T >= 0), None)
        t5 = next((t for t, T in th if T >= 5), None)
        t10 = next((t for t, T in th if T >= 10), None)
        t_fin = th[-1][1] if th else None
        t0s = f"{t0:.1f}s" if t0 else "N/A"
        t5s = f"{t5:.1f}s" if t5 else "N/A"
        t10s = f"{t10:.1f}s" if t10 else "N/A"
        print(f"  {label:18s}: ->0C @ {t0s}, ->5C @ {t5s}, ->10C @ {t10s}, T_final={t_fin:.1f}C")

    if res_oil['temp_hist'] and res_cu['temp_hist']:
        t0_o = next((t for t,T in res_oil['temp_hist'] if T >= 0), 999)
        t0_c = next((t for t,T in res_cu['temp_hist'] if T >= 0), 999)
        if t0_o > 0 and t0_c > 0 and t0_c < 999:
            print(f"\n  Copper reaches 0C {t0_o/t0_c:.1f}x faster than oil-filled")
    print("="*60)


if __name__ == '__main__':
    print("="*60)
    print("Ice Cream Scoop Thermal CFD - GPU Solver (PyTorch)")
    print("="*60)

    print("\nDesign A: Oil-filled Aluminum Handle")
    res_oil = run_sim('oil', 'Oil-filled Aluminum')

    print("\nDesign B: Solid Copper Handle")
    res_cu = run_sim('copper', 'Solid Copper')

    print("\nGenerating plots...")
    plot_all(res_oil, res_cu)

    print("\nDone! Results in results/")
