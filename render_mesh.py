#!/usr/bin/env python3
"""
Render the 3D CFD mesh for the ice cream scoop simulation.
Clean side-view cross-section showing the actual scoop shape.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os


class ScoopGeometry:
    def __init__(self, nx, ny, nz, dx, dy, dz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.cell_type = np.zeros((nx, ny, nz), dtype=np.int32)
        self.bc_hand = np.zeros((nx, ny, nz), dtype=bool)
        self.bc_icecream = np.zeros((nx, ny, nz), dtype=bool)
        self.bc_ambient = np.zeros((nx, ny, nz), dtype=bool)

    def build_oil_design(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        z = (np.arange(nz) + 0.5) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        cx = nx * dx / 2
        cy = ny * dy / 2
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        handle_mask = Z <= 0.150
        r_outer, r_inner = 0.010, 0.009
        wall = handle_mask & (R >= r_inner) & (R <= r_outer)
        self.cell_type[wall] = 2
        oil = handle_mask & (R < r_inner)
        self.cell_type[oil] = 1
        head_ro, head_ri = 0.025, 0.023
        dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + Z**2)
        head_wall = (Z <= 0) & (dist_head <= head_ro) & (dist_head >= head_ri)
        self.cell_type[head_wall] = 2
        head_int = (Z <= 0.001) & (dist_head < head_ri)
        self.cell_type[head_int] = 1
        hand = handle_mask & (Z >= 0.050) & (Z <= 0.150) & (R >= r_outer - dx) & (R <= r_outer + dx)
        self.bc_hand[hand] = True
        ice = (Z <= -0.020) & (dist_head >= head_ro - dx) & (dist_head <= head_ro + dx)
        self.bc_icecream[ice] = True

    def build_copper_design(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        z = (np.arange(nz) + 0.5) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        cx = nx * dx / 2
        cy = ny * dy / 2
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        handle_mask = Z <= 0.150
        r_outer = 0.010
        self.cell_type[handle_mask & (R <= r_outer)] = 3
        head_ro, head_ri = 0.025, 0.023
        dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + Z**2)
        self.cell_type[(Z <= 0) & (dist_head <= head_ro) & (dist_head >= head_ri)] = 3
        self.cell_type[(Z <= 0) & (dist_head < head_ri)] = 3
        hand = handle_mask & (Z >= 0.050) & (Z <= 0.150) & (R >= r_outer - dx) & (R <= r_outer + dx)
        self.bc_hand[hand] = True
        ice = (Z <= -0.020) & (dist_head >= head_ro - dx) & (dist_head <= head_ro + dx)
        self.bc_icecream[ice] = True


def render_cross_section(geom, title, filename):
    """Clean 2D cross-section side view - shows actual scoop shape."""
    ct = geom.cell_type
    nx, ny, nz = ct.shape
    cy = ny // 2

    # Extract center plane (X-Z slice)
    slice_2d = ct[:, cy, :]

    # Material colors (RGB)
    cmap = {
        0: [1, 1, 1],        # air - white
        1: [1.0, 0.84, 0.0], # oil - gold
        2: [0.6, 0.6, 0.62], # aluminum - silver-gray
        3: [0.72, 0.45, 0.20], # copper - bronze
    }

    img = np.ones((nz, nx, 3))
    for i in range(nx):
        for k in range(nz):
            mat = slice_2d[i, k]
            if mat in cmap:
                img[k, i] = cmap[mat]
            # Flip so z=0 (head) is at bottom
    img = img[::-1, :, :]

    fig, ax = plt.subplots(figsize=(6, 14))
    ax.imshow(img, extent=[0, nx*geom.dx*1000, -nz*geom.dz*1000, 0],
              aspect='equal', interpolation='nearest')

    # Annotate boundary conditions
    hand_slice = geom.bc_hand[:, cy, :]
    hand_pts = np.where(hand_slice)
    if len(hand_pts[0]) > 0:
        ax.scatter(hand_pts[0]*geom.dx*1000, -(hand_pts[1]+0.5)*geom.dz*1000,
                   color='red', s=2, label='Hand grip (33°C)', alpha=0.7, marker='s')

    ice_slice = geom.bc_icecream[:, cy, :]
    ice_pts = np.where(ice_slice)
    if len(ice_pts[0]) > 0:
        ax.scatter(ice_pts[0]*geom.dx*1000, -(ice_pts[1]+0.5)*geom.dz*1000,
                   color='royalblue', s=2, label='Ice cream (-18°C)', alpha=0.7, marker='s')

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    legend_items = [Patch(facecolor='#FFD700', label='Mineral Oil'),
                    Patch(facecolor='#999999', label='Aluminum')]
    if 3 in slice_2d:
        legend_items.append(Patch(facecolor='#B87333', label='Copper'))
    legend_items.append(Patch(facecolor='red', label='Hand BC (33°C)'))
    legend_items.append(Patch(facecolor='royalblue', label='Ice Cream BC (-18°C)'))
    ax.legend(handles=legend_items, loc='upper right', fontsize=9)

    ax.set_xlim(-5, nx*geom.dx*1000 + 5)
    ax.set_ylim(-nz*geom.dz*1000 + 5, 5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def render_3d_clean(title, filename, design='oil'):
    """Render a clean 3D representation using matplotlib patches."""
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Draw handle as cylinder
    from matplotlib.colors import to_rgba
    import matplotlib.patches as mpatches

    # Parameters in mm
    handle_len = 150
    handle_r = 10
    head_r = 25
    head_wall = 2

    theta = np.linspace(0, 2*np.pi, 60)

    # Handle outer surface
    if design == 'oil':
        handle_color = '#A0A0A0'
        oil_color = '#FFD700'
    else:
        handle_color = '#B87333'
        oil_color = None

    # Draw handle as series of circles
    z_vals = np.linspace(0, handle_len, 30)
    for z in z_vals:
        x = handle_r * np.cos(theta)
        y = handle_r * np.sin(theta)
        ax.plot(x, y, z, color=handle_color, alpha=0.3, linewidth=0.5)

    # Handle outline (top and bottom circles)
    for z in [0, handle_len]:
        x = handle_r * np.cos(theta)
        y = handle_r * np.sin(theta)
        ax.plot(x, y, z, color=handle_color, linewidth=2)

    # Vertical lines on handle
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = handle_r * np.cos(angle)
        y = handle_r * np.sin(angle)
        ax.plot([x, x], [y, y], [0, handle_len], color=handle_color, alpha=0.4, linewidth=0.8)

    # Scoop head (hemisphere)
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi/2, 20)
    x_head = head_r * np.outer(np.cos(u), np.sin(v))
    y_head = head_r * np.outer(np.sin(u), np.sin(v))
    z_head = -head_r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_head, y_head, z_head, alpha=0.3, color=handle_color)

    # Inner oil cavity (cutaway view - show half)
    if design == 'oil':
        # Oil column in handle
        oil_r = handle_r - head_wall
        z_oil = np.linspace(0, handle_len, 30)
        for z in z_oil:
            x = oil_r * np.cos(theta) * (theta < np.pi)  # half cutaway
            y = oil_r * np.sin(theta) * (theta < np.pi)
            ax.plot(x, y, z, color=oil_color, alpha=0.5, linewidth=0.5)

        # Oil in head
        inner_r = head_r - head_wall
        u2 = np.linspace(0, np.pi, 40)
        v2 = np.linspace(0, np.pi/2, 20)
        x_in = inner_r * np.outer(np.cos(u2), np.sin(v2))
        y_in = inner_r * np.outer(np.sin(u2), np.sin(v2))
        z_in = -inner_r * np.outer(np.ones_like(u2), np.cos(v2))
        ax.plot_surface(x_in, y_in, z_in, alpha=0.4, color=oil_color)

    # Cutaway plane (remove front half)
    ax.set_xlim(-head_r*1.2, head_r*1.2)
    ax.set_ylim(-head_r*1.2, head_r*1.2)
    ax.set_zlim(-head_r*1.2, handle_len*1.05)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    handles = [mpatches.Patch(facecolor=handle_color, alpha=0.5, label='Aluminum' if design=='oil' else 'Copper')]
    if design == 'oil':
        handles.append(mpatches.Patch(facecolor=oil_color, alpha=0.5, label='Mineral Oil'))
    ax.legend(handles=handles, loc='upper left', fontsize=10)

    ax.view_init(elev=15, azim=-60)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    nx, ny, nz = 48, 48, 160
    dx = 0.030 / nx
    dy = 0.030 / ny
    dz = 0.200 / nz

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Cross-sections
    print("Building Design A...")
    geom_a = ScoopGeometry(nx, ny, nz, dx, dy, dz)
    geom_a.build_oil_design()
    render_cross_section(geom_a, 'Design A: Oil-Filled Aluminum\n(Cross-Section)',
                         os.path.join(results_dir, 'mesh_design_a.png'))
    render_3d_clean('Design A: Oil-Filled Aluminum\n(3D Cutaway)',
                    os.path.join(results_dir, 'mesh_3d_a.png'), 'oil')

    print("Building Design B...")
    geom_b = ScoopGeometry(nx, ny, nz, dx, dy, dz)
    geom_b.build_copper_design()
    render_cross_section(geom_b, 'Design B: Solid Copper\n(Cross-Section)',
                         os.path.join(results_dir, 'mesh_design_b.png'))
    render_3d_clean('Design B: Solid Copper\n(3D Cutaway)',
                    os.path.join(results_dir, 'mesh_3d_b.png'), 'copper')

    ct_a = geom_a.cell_type
    ct_b = geom_b.cell_type
    print(f"\nDesign A: {np.sum(ct_a==1)} oil cells, {np.sum(ct_a==2)} aluminum cells")
    print(f"Design B: {np.sum(ct_b==3)} copper cells")
    print("Done!")


if __name__ == '__main__':
    main()
