#!/usr/bin/env python3
"""
Render the 3D CFD mesh for the ice cream scoop simulation.
Shows cell types (oil, aluminum, copper, air) as a 3D voxel visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

# Import geometry from the solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Extract ScoopGeometry directly to avoid torch import
import json

# Material properties (copied from cfd3d_solver)
MATERIALS = {
    'aluminum': {'k': 205.0, 'rho': 2700.0, 'cp': 900.0, 'mu': 1e30, 'beta': 0.0},
    'copper': {'k': 401.0, 'rho': 8960.0, 'cp': 385.0, 'mu': 1e30, 'beta': 0.0},
    'oil': {'k': 0.15, 'rho': 850.0, 'cp': 1670.0, 'mu': 0.08, 'beta': 0.00064},
    'air': {'k': 0.026, 'rho': 1.2, 'cp': 1005.0, 'mu': 1.8e-5, 'beta': 0.00343},
}

try:
    from cfd3d_solver import ScoopGeometry
except ImportError:
    # Inline copy of ScoopGeometry (no torch dependency)
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
            r_outer = 0.010
            r_inner = 0.009
            wall_mask = handle_mask & (R >= r_inner) & (R <= r_outer)
            self.cell_type[wall_mask] = 2
            oil_mask = handle_mask & (R < r_inner)
            self.cell_type[oil_mask] = 1
            head_r_outer = 0.025
            head_r_inner = 0.023
            dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + Z**2)
            head_outer_mask = (Z <= 0) & (dist_head <= head_r_outer) & (dist_head >= head_r_inner)
            self.cell_type[head_outer_mask] = 2
            head_interior = (Z <= 0.001) & (dist_head < head_r_inner)
            self.cell_type[head_interior] = 1
            hand_mask = handle_mask & (Z >= 0.050) & (Z <= 0.150) & (R >= r_outer - dx) & (R <= r_outer + dx)
            self.bc_hand[hand_mask] = True
            icecream_mask = (Z <= -0.020) & (dist_head >= head_r_outer - dx) & (dist_head <= head_r_outer + dx)
            self.bc_icecream[icecream_mask] = True
            ambient_lower_handle = handle_mask & (Z < 0.050) & (R >= r_outer - dx) & (R <= r_outer + dx)
            self.bc_ambient[ambient_lower_handle] = True
        
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
            copper_handle = handle_mask & (R <= r_outer)
            self.cell_type[copper_handle] = 3
            head_r_outer = 0.025
            head_r_inner = 0.023
            dist_head = np.sqrt((X - cx)**2 + (Y - cy)**2 + Z**2)
            head_wall = (Z <= 0) & (dist_head <= head_r_outer) & (dist_head >= head_r_inner)
            self.cell_type[head_wall] = 3
            head_solid = (Z <= 0) & (dist_head < head_r_inner)
            self.cell_type[head_solid] = 3
            hand_mask = handle_mask & (Z >= 0.050) & (Z <= 0.150) & (R >= r_outer - dx) & (R <= r_outer + dx)
            self.bc_hand[hand_mask] = True
            icecream_mask = (Z <= -0.020) & (dist_head >= head_r_outer - dx) & (dist_head <= head_r_outer + dx)
            self.bc_icecream[icecream_mask] = True


def render_design(geom, design_name, filename):
    """Render the 3D mesh as voxel slices."""
    ct = geom.cell_type
    nx, ny, nz = ct.shape
    
    fig = plt.figure(figsize=(16, 10))
    
    # --- Plot 1: 3D voxel cutaway (showing half the mesh) ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Cutaway: show only y >= cy (half cut)
    cy = ny // 2
    
    # Material colors
    colors = {
        0: None,           # air (don't draw)
        1: '#FFD700',       # oil - gold
        2: '#A0A0A0',       # aluminum - gray
        3: '#B87333',       # copper - bronze
        4: '#87CEEB',       # ice cream - light blue
    }
    labels = {0: 'Air', 1: 'Mineral Oil', 2: 'Aluminum', 3: 'Copper', 4: 'Ice Cream'}
    
    # Collect voxels for each material
    voxel_data = np.zeros((nx, ny//2, nz), dtype=bool)
    voxel_colors = np.empty((nx, ny//2, nz), dtype=object)
    
    for i in range(nx):
        for j in range(cy, ny):
            for k in range(nz):
                mat = ct[i, j, k]
                if mat in colors and colors[mat] is not None:
                    voxel_data[i, j-cy, k] = True
                    voxel_colors[i, j-cy, k] = colors[mat]
    
    ax1.voxels(voxel_data, facecolors=voxel_colors, edgecolors='k', linewidth=0.1)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[m], label=labels[m]) 
                       for m in [1, 2, 3] if m in colors]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'{design_name}\n3D Mesh Cutaway', fontsize=12)
    
    # Scale to mm
    ax1.set_xticks(np.linspace(0, nx, 5))
    ax1.set_xticklabels([f'{v:.0f}' for v in np.linspace(0, nx*geom.dx*1000, 5)])
    ax1.set_yticks(np.linspace(0, ny//2, 5))
    ax1.set_yticklabels([f'{v:.0f}' for v in np.linspace(0, (ny//2)*geom.dy*1000, 5)])
    ax1.set_zticks(np.linspace(0, nz, 6))
    z_labels = np.linspace(0, nz*geom.dz*1000, 6)
    ax1.set_zticklabels([f'{v:.0f}' for v in z_labels])
    
    # --- Plot 2: Cross-section slice (side view) ---
    ax2 = fig.add_subplot(122)
    
    # Take a slice at y = cy (center plane)
    slice_2d = ct[:, cy, :]  # x-z plane
    
    # Create colored image
    img = np.ones((nz, nx, 3))  # white background
    color_map = {
        0: [1.0, 1.0, 1.0],    # air - white
        1: [1.0, 0.84, 0.0],   # oil - gold
        2: [0.63, 0.63, 0.63], # aluminum - gray
        3: [0.72, 0.45, 0.20], # copper - bronze
        4: [0.53, 0.81, 0.92], # ice cream - light blue
    }
    
    for i in range(nx):
        for k in range(nz):
            mat = slice_2d[i, k]
            if mat in color_map:
                img[nz-1-k, i] = color_map[mat]  # flip z for display
    
    ax2.imshow(img, extent=[0, nx*geom.dx*1000, 0, nz*geom.dz*1000], 
               aspect='auto', interpolation='nearest')
    
    # Mark boundary conditions
    bc_hand_slice = geom.bc_hand[:, cy, :]
    bc_ice_slice = geom.bc_icecream[:, cy, :]
    
    # Overlay BC markers
    hand_pts = np.where(bc_hand_slice)
    if len(hand_pts[0]) > 0:
        ax2.scatter(hand_pts[0]*geom.dx*1000, [nz-1-k for k in hand_pts[1] * geom.dz * 1000 / geom.dz], 
                   color='red', s=1, label='Hand (33°C)', alpha=0.5)
    ice_pts = np.where(bc_ice_slice)
    if len(ice_pts[0]) > 0:
        ax2.scatter([p*geom.dx*1000 for p in ice_pts[0]], 
                   [nz*geom.dz*1000 - k*geom.dz*1000 for k in ice_pts[1]],
                   color='blue', s=1, label='Ice Cream (-18°C)', alpha=0.5)
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title(f'{design_name}\nCross-Section (YZ center plane)', fontsize=12)
    
    # Legend for materials
    from matplotlib.patches import Patch
    legend_el = [
        Patch(facecolor='#FFD700', label='Mineral Oil'),
        Patch(facecolor='#A0A0A0', label='Aluminum'),
        Patch(facecolor='#B87333', label='Copper'),
        Patch(facecolor='red', label='Hand BC (33°C)'),
        Patch(facecolor='blue', label='Ice Cream BC (-18°C)'),
    ]
    ax2.legend(handles=legend_el, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    # Grid parameters (same as solver)
    # Domain: 30mm x 30mm x 200mm
    # Grid: 48 x 48 x 160
    nx, ny, nz = 48, 48, 160
    dx = 0.030 / nx   # ~0.625mm
    dy = 0.030 / ny
    dz = 0.200 / nz   # ~1.25mm
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Design A: Oil-filled aluminum
    print("Building Design A (Oil + Aluminum)...")
    geom_a = ScoopGeometry(nx, ny, nz, dx, dy, dz)
    geom_a.build_oil_design()
    render_design(geom_a, 'Design A: Oil-Filled Aluminum', 
                   os.path.join(results_dir, 'mesh_design_a.png'))
    
    # Print stats
    ct_a = geom_a.cell_type
    print(f"  Oil cells: {np.sum(ct_a == 1)}")
    print(f"  Aluminum cells: {np.sum(ct_a == 2)}")
    print(f"  Air cells: {np.sum(ct_a == 0)}")
    
    # Design B: Solid copper
    print("\nBuilding Design B (Solid Copper)...")
    geom_b = ScoopGeometry(nx, ny, nz, dx, dy, dz)
    geom_b.build_copper_design()
    render_design(geom_b, 'Design B: Solid Copper',
                   os.path.join(results_dir, 'mesh_design_b.png'))
    
    ct_b = geom_b.cell_type
    print(f"  Copper cells: {np.sum(ct_b == 3)}")
    print(f"  Air cells: {np.sum(ct_b == 0)}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
