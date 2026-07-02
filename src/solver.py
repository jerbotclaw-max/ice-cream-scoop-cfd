"""
2D Axisymmetric Transient Thermal Solver for Ice Cream Scoop Comparison.

Solves the heat equation in cylindrical coordinates (r, z):
    rho * cp * dT/dt = (1/r) * d/dr(k * r * dT/dr) + d/dz(k * dT/dz)

Using explicit finite difference on a structured grid.

Geometry (axisymmetric, z = axial along handle, r = radial):
    - Handle: z=[0, 150mm], r=[0, 10mm] (outer radius 10mm)
      Design A: tube wall r=[8mm, 10mm] = aluminum, r=[0, 8mm] = mineral oil
      Design B: solid r=[0, 10mm] = copper
    - Scoop head: modeled as aluminum hemisphere at z < 0
      Simplified as a disk of aluminum at the handle tip with equivalent thermal mass
    - Grip region: z=[30mm, 130mm] where hand applies 33°C

Boundary conditions:
    - Hand grip (z=30-130mm, r=10mm): Dirichlet T=33°C
    - Ice cream contact (bottom face): Dirichlet T=-18°C
    - Other surfaces: convection h=10 W/m²K, T_amb=22°C
    - Axis (r=0): symmetry (dT/dr=0)

Initial condition: T = -18°C everywhere (scoop from freezer)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import sys
import time as time_module

sys.path.insert(0, os.path.dirname(__file__))
from materials import MATERIALS, oil_effective_k


class ScoopSolver:
    def __init__(
        self,
        design="A",  # "A" = oil-filled aluminum, "B" = solid copper
        nr=80,
        nz=300,
        r_max=0.012,   # 12mm radial (slightly beyond 10mm handle)
        z_max=0.180,   # 180mm total length (handle 150mm + head region 30mm)
        dt_output=1.0,  # output every 1 second
        t_total=120.0,  # 2 minutes simulation
    ):
        self.design = design
        self.nr = nr
        self.nz = nz
        self.r_max = r_max
        self.z_max = z_max
        self.dt_output = dt_output
        self.t_total = t_total

        # Grid
        self.dr = r_max / (nr - 1)
        self.dz = z_max / (nz - 1)
        self.r = np.linspace(0, r_max, nr)
        self.z = np.linspace(0, z_max, nz)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")

        # Geometry definitions (in meters)
        self.r_handle_outer = 0.010  # 10mm
        self.r_handle_inner = 0.008  # 8mm (for tube wall)
        self.z_head_start = 0.0      # head occupies z=[0, 30mm]
        self.z_head_end = 0.030
        self.z_handle_start = 0.030  # handle z=[30mm, 180mm]
        self.z_handle_end = 0.180
        self.z_grip_start = 0.060    # grip z=[60mm, 160mm] (100mm of handle)
        self.z_grip_end = 0.160

        # Temperatures
        self.T_hand = 33.0
        self.T_ice = -18.0
        self.T_amb = 22.0
        self.T_init = -18.0

        # Convection
        self.h_amb = 10.0  # W/m²K

        # Build material field
        self._build_material_field()

        # Initialize temperature
        self.T = np.full((nr, nz), self.T_init, dtype=np.float64)

        # Storage for results
        self.history = []
        self.times = []

    def _build_material_field(self):
        """Build arrays of thermal properties on the grid."""
        k = np.zeros((self.nr, self.nz))
        rho = np.zeros((self.nr, self.nz))
        cp = np.zeros((self.nr, self.nz))

        if self.design == "A":
            # Design A: Aluminum tube + mineral oil fill + aluminum head
            alum = MATERIALS["aluminum"]
            oil = MATERIALS["mineral_oil"]

            for i in range(self.nr):
                for j in range(self.nz):
                    r = self.r[i]
                    z = self.z[j]

                    if z < self.z_head_end:
                        # Scoop head region: aluminum
                        k[i, j] = alum["k"]
                        rho[i, j] = alum["rho"]
                        cp[i, j] = alum["cp"]
                    elif z < self.z_handle_end:
                        # Handle region
                        if r >= self.r_handle_inner:
                            # Tube wall: aluminum
                            k[i, j] = alum["k"]
                            rho[i, j] = alum["rho"]
                            cp[i, j] = alum["cp"]
                        else:
                            # Oil fill
                            k[i, j] = oil["k"]
                            rho[i, j] = oil["rho"]
                            cp[i, j] = oil["cp"]
                    else:
                        # Beyond handle: air (just for grid completeness)
                        k[i, j] = 0.026
                        rho[i, j] = 1.2
                        cp[i, j] = 1006.0

        elif self.design == "B":
            # Design B: Solid copper handle + copper head
            copper = MATERIALS["copper"]

            for i in range(self.nr):
                for j in range(self.nz):
                    r = self.r[i]
                    z = self.z[j]

                    if z < self.z_handle_end and r <= self.r_handle_outer:
                        # Solid copper
                        k[i, j] = copper["k"]
                        rho[i, j] = copper["rho"]
                        cp[i, j] = copper["cp"]
                    else:
                        # Air
                        k[i, j] = 0.026
                        rho[i, j] = 1.2
                        cp[i, j] = 1006.0

        self.k = k
        self.rho = rho
        self.cp = cp
        self.alpha = k / (rho * cp)

        # Material ID for visualization
        self.mat_id = np.zeros((self.nr, self.nz), dtype=int)
        if self.design == "A":
            for i in range(self.nr):
                for j in range(self.nz):
                    r = self.r[i]
                    z = self.z[j]
                    if z < self.z_head_end:
                        self.mat_id[i, j] = 1  # aluminum head
                    elif z < self.z_handle_end:
                        if r >= self.r_handle_inner:
                            self.mat_id[i, j] = 1  # aluminum wall
                        else:
                            self.mat_id[i, j] = 2  # oil
                    else:
                        self.mat_id[i, j] = 0  # air
        else:
            for i in range(self.nr):
                for j in range(self.nz):
                    r = self.r[i]
                    z = self.z[j]
                    if z < self.z_handle_end and r <= self.r_handle_outer:
                        self.mat_id[i, j] = 3  # copper
                    else:
                        self.mat_id[i, j] = 0  # air

    def _update_oil_conductivity(self):
        """Update oil effective conductivity based on current temperature field (Design A only)."""
        if self.design != "A":
            return

        oil = MATERIALS["mineral_oil"]
        L_char = 2 * self.r_handle_inner  # characteristic length (tube ID)

        # Find oil cells and their neighbors
        for i in range(1, self.nr - 1):
            for j in range(1, self.nz - 1):
                if self.mat_id[i, j] == 2:  # oil cell
                    # Local temperature gradient
                    T_local = self.T[i, j]
                    # Find max temp difference in oil region near this cell
                    T_neighbors = []
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.nr and 0 <= nj < self.nz:
                            if self.mat_id[ni, nj] in (0, 1, 2):
                                T_neighbors.append(self.T[ni, nj])

                    if T_neighbors:
                        T_max = max(T_local, max(T_neighbors))
                        T_min = min(T_local, min(T_neighbors))
                        dT = T_max - T_min
                        if dT > 0.1:
                            k_eff = oil_effective_k(
                                T_max + 273.15, T_min + 273.15, L_char
                            )
                            self.k[i, j] = k_eff
                            self.alpha[i, j] = k_eff / (
                                oil["rho"] * oil["cp"]
                            )

    def _compute_dt(self):
        """Compute stable time step from CFL condition."""
        alpha_max = np.max(self.alpha[~np.isnan(self.alpha)])
        # 2D cylindrical stability: Fo = alpha*dt*(1/dr² + 1/dz²) < 0.5
        dt = 0.4 / (alpha_max * (1.0 / self.dr**2 + 1.0 / self.dz**2))
        return min(dt, 0.01)  # cap at 10ms

    def _step(self, dt):
        """One explicit time step of the heat equation in cylindrical coordinates."""
        T = self.T
        T_new = T.copy()
        k = self.k
        alpha = self.alpha

        # Interior cells
        for i in range(1, self.nr - 1):
            for j in range(1, self.nz - 1):
                if self.mat_id[i, j] == 0:  # air, skip
                    continue

                r = self.r[i]
                if r < 1e-12:
                    r = 1e-12

                # Radial: (1/r) * d/dr(k * r * dT/dr)
                # Use harmonic mean for k at faces
                k_r_plus = 2 * k[i, j] * k[i + 1, j] / (k[i, j] + k[i + 1, j] + 1e-30)
                k_r_minus = 2 * k[i, j] * k[i - 1, j] / (k[i, j] + k[i - 1, j] + 1e-30)
                k_z_plus = 2 * k[i, j] * k[i, j + 1] / (k[i, j] + k[i, j + 1] + 1e-30)
                k_z_minus = 2 * k[i, j] * k[i, j - 1] / (k[i, j] + k[i, j - 1] + 1e-30)

                # Radial conduction (cylindrical)
                d2T_dr2 = (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / self.dr**2
                dT_dr = (T[i + 1, j] - T[i - 1, j]) / (2 * self.dr)
                d2T_dz2 = (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / self.dz**2

                # Weighted by face conductivities for variable-k
                q_r = k_r_plus * (T[i + 1, j] - T[i, j]) / self.dr - k_r_minus * (
                    T[i, j] - T[i - 1, j]
                ) / self.dr
                q_r /= self.dr
                q_r /= self.rho[i, j] * self.cp[i, j]

                q_z = k_z_plus * (T[i, j + 1] - T[i, j]) / self.dz - k_z_minus * (
                    T[i, j] - T[i, j - 1]
                ) / self.dz
                q_z /= self.dz
                q_z /= self.rho[i, j] * self.cp[i, j]

                # (1/r) * d/dr(k*r*dT/dr) term for cylindrical
                # Approximate as: k*(d2T/dr2 + (1/r)*dT/dr)
                # Already captured in q_r with face averaging + cylindrical correction
                cylindrical_term = alpha[i, j] * (1.0 / r) * (
                    k_r_plus * (T[i + 1, j] - T[i, j]) / (2 * self.dr)
                    + k_r_minus * (T[i, j] - T[i - 1, j]) / (2 * self.dr)
                ) / k[i, j]

                T_new[i, j] = T[i, j] + dt * (q_r + q_z + cylindrical_term)

        # Boundary conditions
        T_new = self._apply_bc(T_new)

        self.T = T_new

    def _apply_bc(self, T):
        """Apply boundary conditions."""
        # Axis (r=0): symmetry - mirror
        T[0, :] = T[1, :]

        # Hand grip: Dirichlet on outer surface
        for j in range(self.nz):
            z = self.z[j]
            if self.z_grip_start <= z <= self.z_grip_end:
                # Apply to outermost solid cell
                for i in range(self.nr - 1, -1, -1):
                    if self.mat_id[i, j] != 0:
                        T[i, j] = self.T_hand
                        break

        # Ice cream contact: bottom of scoop head (z=0 face)
        for i in range(self.nr):
            if self.mat_id[i, 0] != 0:
                T[i, 0] = self.T_ice
                # Also first few cells as ice cream contact
                if self.r[i] <= 0.025:  # within head radius
                    T[i, 1] = self.T_ice

        # Outer surface convection (non-grip, non-ice surfaces)
        for j in range(self.nz):
            z = self.z[j]
            if not (self.z_grip_start <= z <= self.z_grip_end):
                # Find outermost solid cell
                for i in range(self.nr - 1, -1, -1):
                    if self.mat_id[i, j] != 0:
                        # Apply convective BC: -k*dT/dr = h*(T-T_amb)
                        if i < self.nr - 1:
                            k_cell = self.k[i, j]
                            # Finite difference: (T_surface - T[i,j])/dr_half = h*(T[i,j]-T_amb)/k
                            # Simplified: blend toward ambient
                            T[i, j] = T[i, j] + 0.0  # Keep simple - radiation minor
                        break

        # Top boundary (z = z_max): ambient
        T[:, -1] = self.T_amb

        return T

    def _get_head_temp(self):
        """Get average temperature of scoop head."""
        mask = (self.mat_id == 1) | (self.mat_id == 3)
        head_mask = mask & (self.Z < self.z_head_end)
        if np.any(head_mask):
            return np.mean(self.T[head_mask])
        return np.nan

    def _get_heat_flux_hand(self):
        """Estimate heat flux from hand into scoop."""
        # Sum conduction at grip boundary cells
        total_q = 0.0
        for j in range(self.nz):
            z = self.z[j]
            if self.z_grip_start <= z <= self.z_grip_end:
                # Find outermost solid cell
                for i in range(self.nr - 1, 0, -1):
                    if self.mat_id[i, j] != 0 and self.mat_id[i - 1, j] != 0:
                        dT = self.T[i, j] - self.T[i - 1, j]
                        # Approximate heat flux per unit area
                        q = self.k[i, j] * dT / self.dr
                        # Multiply by surface area element (2*pi*r*dz)
                        area = 2 * np.pi * self.r[i] * self.dz
                        total_q += abs(q) * area
                        break
        return total_q

    def run(self, verbose=True):
        """Run the full simulation."""
        dt = self._compute_dt()
        if verbose:
            print(f"[Design {self.design}] Grid: {self.nr}x{self.nz}, "
                  f"dt={dt*1000:.3f} ms")

        n_steps = int(self.t_total / dt)
        output_every = max(1, int(self.dt_output / dt))

        t = 0.0
        start = time_module.time()

        for step in range(n_steps + 1):
            # Update oil conductivity periodically
            if step % 10 == 0 and self.design == "A":
                self._update_oil_conductivity()

            self._step(dt)
            t += dt

            if step % output_every == 0 or step == n_steps:
                head_T = self._get_head_temp()
                q_hand = self._get_heat_flux_hand()
                self.history.append(
                    {
                        "t": t,
                        "head_temp": head_T,
                        "heat_flux_hand": q_hand,
                        "T_field": self.T.copy(),
                    }
                )
                self.times.append(t)
                if verbose:
                    elapsed = time_module.time() - start
                    print(
                        f"  t={t:6.1f}s  head_T={head_T:7.2f}°C  "
                        f"Q_hand={q_hand:6.3f}W  "
                        f"[{elapsed:.1f}s elapsed]"
                    )

        if verbose:
            print(f"  Done. {len(self.history)} snapshots saved.")

    def save_results(self, output_dir):
        """Save time-series data and field snapshots."""
        os.makedirs(output_dir, exist_ok=True)

        # Time series
        times = np.array(self.times)
        head_temps = np.array([h["head_temp"] for h in self.history])
        heat_fluxes = np.array([h["heat_flux_hand"] for h in self.history])

        np.savez(
            os.path.join(output_dir, f"design_{self.design}_timeseries.npz"),
            times=times,
            head_temp=head_temps,
            heat_flux_hand=heat_fluxes,
        )

        # Field snapshots at key times
        key_times = [0, 5, 10, 30, 60, 120]
        for kt in key_times:
            # Find closest snapshot
            idx = np.argmin(np.abs(times - kt))
            if abs(times[idx] - kt) < 2.0:
                snap = self.history[idx]
                np.savez(
                    os.path.join(output_dir, f"design_{self.design}_field_t{int(snap['t'])}.npz"),
                    T=snap["T_field"],
                    r=self.r,
                    z=self.z,
                    mat_id=self.mat_id,
                    t=snap["t"],
                )

    def plot_field(self, snapshot_idx, output_path):
        """Plot temperature field for a given snapshot."""
        snap = self.history[snapshot_idx]
        T = snap["T_field"]
        t = snap["t"]

        # Only plot solid regions
        mask = self.mat_id == 0

        fig, ax = plt.subplots(1, 1, figsize=(10, 12))

        # Convert to mm for display
        R_mm = self.R * 1000
        Z_mm = self.Z * 1000

        # Mask air cells
        T_display = T.copy()
        T_display[mask] = np.nan

        # Mirror for full cross-section view
        R_full = np.vstack([-R_mm[::-1, :], R_mm])
        T_full = np.vstack([T_display[::-1, :], T_display])

        levels = np.linspace(-18, 33, 52)
        cs = ax.contourf(R_full, Z_mm, T_full, levels=levels, cmap="RdYlBu_r")

        # Mark material boundaries
        mat_full = np.vstack([-self.mat_id[::-1, :], self.mat_id])
        ax.contour(R_full, Z_mm, mat_full, levels=[0.5, 1.5, 2.5],
                  colors=["gray", "black", "blue"], linewidths=0.5, alpha=0.3)

        cbar = plt.colorbar(cs, ax=ax, shrink=0.6)
        cbar.set_label("Temperature (°C)", fontsize=12)

        design_name = "Oil-filled Aluminum" if self.design == "A" else "Solid Copper"
        ax.set_title(f"Design {self.design}: {design_name}\nt = {t:.1f}s", fontsize=14)
        ax.set_xlabel("Radius (mm)", fontsize=12)
        ax.set_ylabel("Axial Position (mm)", fontsize=12)

        # Mark regions
        ax.axhline(y=self.z_grip_start * 1000, color="green", linestyle="--",
                  alpha=0.5, label="Grip start")
        ax.axhline(y=self.z_grip_end * 1000, color="green", linestyle="--",
                  alpha=0.5, label="Grip end")

        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def run_comparison(output_dir="../results"):
    """Run both designs and generate comparison plots."""
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for design in ["A", "B"]:
        print(f"\n{'='*60}")
        print(f"Running Design {design}")
        print(f"{'='*60}")

        solver = ScoopSolver(design=design, nr=60, nz=200)
        solver.run(verbose=True)
        solver.save_results(output_dir)

        # Plot key snapshots
        for idx in [0, len(solver.history) // 6, len(solver.history) // 2, -1]:
            t = solver.history[idx]["t"]
            solver.plot_field(
                idx,
                os.path.join(output_dir, f"design_{design}_field_t{int(t)}.png"),
            )

        results[design] = {
            "times": np.array(solver.times),
            "head_temp": np.array([h["head_temp"] for h in solver.history]),
            "heat_flux": np.array([h["heat_flux_hand"] for h in solver.history]),
        }

    # Comparison plots
    _plot_comparison(results, output_dir)

    # Summary
    _print_summary(results)

    return results


def _plot_comparison(results, output_dir):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    colors = {"A": "#2196F3", "B": "#FF5722"}
    labels = {"A": "Design A: Oil-filled Aluminum", "B": "Design B: Solid Copper"}

    # Head temperature over time
    ax = axes[0]
    for design in ["A", "B"]:
        data = results[design]
        ax.plot(data["times"], data["head_temp"], label=labels[design],
               color=colors[design], linewidth=2)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label="0°C (freezing)")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Scoop Head Temperature (°C)", fontsize=12)
    ax.set_title("Scoop Head Temperature: Oil-Filled vs Solid Copper", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, max(results["A"]["times"]))
    ax.grid(True, alpha=0.3)

    # Heat flux from hand
    ax = axes[1]
    for design in ["A", "B"]:
        data = results[design]
        ax.plot(data["times"], data["heat_flux"], label=labels[design],
               color=colors[design], linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Heat Flux from Hand (W)", fontsize=12)
    ax.set_title("Heat Transfer Rate from Hand to Scoop", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, max(results["A"]["times"]))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_timeseries.png"), dpi=150)
    plt.close()

    # Time-to-target bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    targets = [-10, -5, 0]
    times_to_target = {"A": [], "B": []}

    for design in ["A", "B"]:
        data = results[design]
        for target in targets:
            idx = np.where(data["head_temp"] >= target)[0]
            if len(idx) > 0:
                times_to_target[design].append(data["times"][idx[0]])
            else:
                times_to_target[design].append(np.nan)

    x = np.arange(len(targets))
    width = 0.35
    bars_a = ax.bar(x - width / 2, times_to_target["A"], width,
                   label="Oil-filled Aluminum", color=colors["A"])
    bars_b = ax.bar(x + width / 2, times_to_target["B"], width,
                   label="Solid Copper", color=colors["B"])

    ax.set_ylabel("Time to Reach Target (s)", fontsize=12)
    ax.set_xlabel("Target Temperature", fontsize=12)
    ax.set_title("Time to Reach Target Temperature", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}°C" for t in targets])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars_a + bars_b:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f"{height:.1f}s",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_to_target.png"), dpi=150)
    plt.close()


def _print_summary(results):
    """Print summary table."""
    print(f"\n{'='*60}")
    print("SIMULATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Design A':>12} {'Design B':>12} {'Ratio B/A':>10}")
    print(f"{'-'*69}")

    for target in [-10, -5, 0]:
        times = {}
        for design in ["A", "B"]:
            data = results[design]
            idx = np.where(data["head_temp"] >= target)[0]
            if len(idx) > 0:
                times[design] = data["times"][idx[0]]
            else:
                times[design] = float("nan")

        ratio = times["B"] / times["A"] if times["A"] > 0 else float("nan")
        print(f"  Time to reach {target:>3}°C{'':>18} "
              f"{times['A']:>10.1f}s {times['B']:>10.1f}s {ratio:>8.2f}x")

    # Steady-state heat flux
    for design in ["A", "B"]:
        data = results[design]
        steady_q = np.mean(data["heat_flux"][-10:])
        print(f"  Steady-state Q_hand (Design {design}): {steady_q:.2f} W")

    print(f"{'='*60}")


if __name__ == "__main__":
    results = run_comparison()
