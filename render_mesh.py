#!/usr/bin/env python3
"""
Render the ice cream scoop CFD mesh - FIXED geometry.
The scoop head hemisphere extends below z=0, handle goes from z=0 to z=150mm.
Domain: z from -25mm (bottom of head) to +200mm (top of handle + margin).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import os


class ScoopGeometry:
    """Fixed geometry with proper z-domain covering negative z for head."""
    
    def __init__(self, nx, ny, nz, dx, dy, dz, z_offset):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.z_offset = z_offset  # z value at first cell
        self.cell_type = np.zeros((nx, ny, nz), dtype=np.int32)
        self.bc_hand = np.zeros((nx, ny, nz), dtype=bool)
        self.bc_icecream = np.zeros((nx, ny, nz), dtype=bool)
        self.bc_ambient = np.zeros((nx, ny, nz), dtype=bool)

    def _coords(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        z = (np.arange(nz) + 0.5) * dz + self.z_offset  # shift to cover negative z
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return X, Y, Z

    def build_oil_design(self):
        X, Y, Z = self._coords()
        nx, ny, nz = self.nx, self.ny, self.nz
        dx = self.dx
        
        cx = nx * self.dx / 2
        cy = ny * self.dy / 2
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # Handle: z from 0 to 150mm
        handle_mask = (Z >= 0) & (Z <= 0.150)
        r_outer, r_inner = 0.010, 0.009

        wall = handle_mask & (R >= r_inner) & (R <= r_outer)
        self.cell_type[wall] = 2  # aluminum
        oil = handle_mask & (R < r_inner)
        self.cell_type[oil] = 1  # oil

        # Head: hemisphere centered at z=0, extends to z=-25mm
        head_ro, head_ri = 0.025, 0.023
        dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + Z**2)
        head_wall = (Z <= 0) & (dist_head <= head_ro) & (dist_head >= head_ri)
        self.cell_type[head_wall] = 2  # aluminum
        head_int = (Z <= 0) & (dist_head < head_ri)
        self.cell_type[head_int] = 1  # oil

        # BCs
        hand = handle_mask & (Z >= 0.050) & (Z <= 0.150) & \
               (R >= r_outer - dx) & (R <= r_outer + dx)
        self.bc_hand[hand] = True
        ice = (Z <= -0.010) & (dist_head >= head_ro - dx) & (dist_head <= head_ro + dx)
        self.bc_icecream[ice] = True

    def build_copper_design(self):
        X, Y, Z = self._coords()
        dx = self.dx
        nx, ny, nz = self.nx, self.ny, self.nz
        
        cx = nx * self.dx / 2
        cy = ny * self.dy / 2
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)

        handle_mask = (Z >= 0) & (Z <= 0.150)
        r_outer = 0.010
        self.cell_type[handle_mask & (R <= r_outer)] = 3

        head_ro, head_ri = 0.025, 0.023
        dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + Z**2)
        self.cell_type[(Z <= 0) & (dist_head <= head_ro) & (dist_head >= head_ri)] = 3
        self.cell_type[(Z <= 0) & (dist_head < head_ri)] = 3

        hand = handle_mask & (Z >= 0.050) & (Z <= 0.150) & \
               (R >= r_outer - dx) & (R <= r_outer + dx)
        self.bc_hand[hand] = True
        ice = (Z <= -0.010) & (dist_head >= head_ro - dx) & (dist_head <= head_ro + dx)
        self.bc_icecream[ice] = True


def render_cross_section(geom, title, filename):
    ct = geom.cell_type
    nx, ny, nz = ct.shape
    cy = ny // 2
    slice_2d = ct[:, cy, :]

    colors = {0: [1, 1, 1], 1: [1.0, 0.84, 0.0], 2: [0.55, 0.57, 0.60], 3: [0.72, 0.45, 0.20]}
    img = np.ones((nz, nx, 3))
    for i in range(nx):
        for k in range(nz):
            mat = slice_2d[i, k]
            if mat in colors:
                img[k, i] = colors[mat]
    img = img[::-1, :, :]

    z_min = geom.z_offset * 1000
    z_max = (geom.z_offset + nz * geom.dz) * 1000
    x_max = nx * geom.dx * 1000

    fig, ax = plt.subplots(figsize=(8, 16))
    ax.imshow(img, extent=[-x_max/2, x_max/2, z_min, z_max], aspect='equal', interpolation='nearest')

    # BCs
    hand_slice = geom.bc_hand[:, cy, :]
    hp = np.where(hand_slice)
    if len(hp[0]) > 0:
        ax.scatter(hp[0]*geom.dx*1000 - x_max/2, 
                   (hp[1]+0.5)*geom.dz*1000 + geom.z_offset*1000,
                   color='red', s=3, label='Hand (33°C)', alpha=0.8, marker='s')
    ice_slice = geom.bc_icecream[:, cy, :]
    ip = np.where(ice_slice)
    if len(ip[0]) > 0:
        ax.scatter(ip[0]*geom.dx*1000 - x_max/2,
                   (ip[1]+0.5)*geom.dz*1000 + geom.z_offset*1000,
                   color='royalblue', s=3, label='Ice cream (-18°C)', alpha=0.8, marker='s')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(title, fontsize=13, fontweight='bold')

    items = [Patch(facecolor='#FFD700', label='Mineral Oil'),
             Patch(facecolor='#8C8C92', label='Aluminum')]
    if np.any(ct == 3):
        items.append(Patch(facecolor='#B87333', label='Copper'))
    items.append(Patch(facecolor='red', label='Hand BC (33°C)'))
    items.append(Patch(facecolor='royalblue', label='Ice Cream BC (-18°C)'))
    ax.legend(handles=items, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def render_3d(title, filename, design='oil'):
    fig = plt.figure(figsize=(10, 16))
    ax = fig.add_subplot(111, projection='3d')

    r_handle = 5
    len_handle = 75
    r_head = 12.5
    wall = 1

    if design == 'oil':
        solid_c = '#909098'
        fluid_c = '#FFD700'
    else:
        solid_c = '#B87333'
        fluid_c = None

    theta = np.linspace(0, 2*np.pi, 60)

    # Handle cylinder (back half for cutaway)
    u = np.linspace(0, np.pi, 40)
    z_h = np.linspace(0, len_handle, 40)
    U, ZH = np.meshgrid(u, z_h)
    ax.plot_surface(r_handle * np.cos(U), r_handle * np.sin(U), ZH,
                    color=solid_c, alpha=0.25)

    # Handle caps
    ax.plot(r_handle * np.cos(theta), r_handle * np.sin(theta), len_handle,
            color=solid_c, linewidth=1.5)

    # Head hemisphere (outer, back half)
    phi = np.linspace(0, 2*np.pi, 60)
    psi = np.linspace(0, np.pi/2, 30)
    PH, PS = np.meshgrid(phi, psi)
    XH = r_head * np.cos(PH) * np.sin(PS)
    YH = r_head * np.sin(PH) * np.sin(PS)
    ZH2 = -r_head * np.cos(PS)
    mask = YH >= -0.3
    ax.plot_surface(np.where(mask, XH, np.nan), np.where(mask, YH, np.nan),
                    np.where(mask, ZH2, np.nan), color=solid_c, alpha=0.25)

    # Head interior
    ri = r_head - wall
    XI = ri * np.cos(PH) * np.sin(PS)
    YI = ri * np.sin(PH) * np.sin(PS)
    ZI = -ri * np.cos(PS)
    mask_i = YI >= -0.3
    if design == 'oil':
        ax.plot_surface(np.where(mask_i, XI, np.nan), np.where(mask_i, YI, np.nan),
                        np.where(mask_i, ZI, np.nan), color=fluid_c, alpha=0.35)
    else:
        ax.plot_surface(np.where(mask_i, XI, np.nan), np.where(mask_i, YI, np.nan),
                        np.where(mask_i, ZI, np.nan), color=solid_c, alpha=0.4)

    # Oil column in handle (cutaway)
    if design == 'oil':
        ax.plot_surface((r_handle - wall) * np.cos(U), (r_handle - wall) * np.sin(U), ZH,
                        color=fluid_c, alpha=0.2)

    ax.set_xlim(-r_head * 1.3, r_head * 1.3)
    ax.set_ylim(-r_head * 1.3, r_head * 1.3)
    ax.set_zlim(-r_head * 1.3, len_handle * 1.1)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title, fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch as P
    if design == 'oil':
        ax.legend(handles=[P(facecolor=fluid_c, alpha=0.5, label='Mineral Oil'),
                           P(facecolor=solid_c, alpha=0.4, label='Aluminum')],
                  loc='upper left', fontsize=10)
    else:
        ax.legend(handles=[P(facecolor=solid_c, alpha=0.5, label='Copper (solid)')],
                  loc='upper left', fontsize=10)

    ax.view_init(elev=5, azim=-75)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    # Domain: z from -30mm to +200mm (covers head + handle)
    nx, ny, nz = 48, 48, 184
    dx = 0.030 / nx       # 0.625mm
    dy = 0.030 / ny
    dz = 0.230 / nz       # ~1.25mm
    z_offset = -0.030      # z starts at -30mm (head extends to -25mm)

    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results, exist_ok=True)

    print("Building Design A (Oil + Aluminum)...")
    g = ScoopGeometry(nx, ny, nz, dx, dy, dz, z_offset)
    g.build_oil_design()
    
    ct = g.cell_type
    print(f"  Oil: {np.sum(ct==1)}, Aluminum: {np.sum(ct==2)}")
    
    # Verify head exists
    z_coords = (np.arange(nz) + 0.5) * dz + z_offset
    head_z = np.where(z_coords < 0)[0]
    if len(head_z) > 0:
        head_cells = ct[:, :, head_z]
        print(f"  Head region (z<0): {np.sum(head_cells != 0)} non-air cells")
    
    render_cross_section(g, 'Design A: Oil-Filled Aluminum\nMesh Cross-Section',
                         os.path.join(results, 'mesh_design_a.png'))
    render_3d('Design A: Oil-Filled Aluminum\n3D Cutaway',
              os.path.join(results, 'mesh_3d_a.png'), 'oil')

    print("\nBuilding Design B (Solid Copper)...")
    g2 = ScoopGeometry(nx, ny, nz, dx, dy, dz, z_offset)
    g2.build_copper_design()
    print(f"  Copper: {np.sum(g2.cell_type==3)}")
    render_cross_section(g2, 'Design B: Solid Copper\nMesh Cross-Section',
                         os.path.join(results, 'mesh_design_b.png'))
    render_3d('Design B: Solid Copper\n3D Cutaway',
              os.path.join(results, 'mesh_3d_b.png'), 'copper')

    print("\nDone!")


if __name__ == '__main__':
    main()
