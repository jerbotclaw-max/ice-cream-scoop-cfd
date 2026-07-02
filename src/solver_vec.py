"""
Vectorized 2D Axisymmetric Transient Thermal Solver for Ice Cream Scoop Comparison.

Uses numpy array operations (no Python loops over grid) for speed.
Solves heat equation in cylindrical coordinates (r, z).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time as time_module

sys.path.insert(0, os.path.dirname(__file__))
from materials import MATERIALS, oil_effective_k


class ScoopSolverVec:
    def __init__(
        self,
        design="A",
        nr=60,
        nz=200,
        t_total=120.0,
    ):
        self.design = design
        self.nr = nr
        self.nz = nz
        self.t_total = t_total

        # Geometry (meters)
        self.r_max = 0.012
        self.z_max = 0.180
        self.dr = self.r_max / (nr - 1)
        self.dz = self.z_max / (nz - 1)
        self.r = np.linspace(0, self.r_max, nr)
        self.z = np.linspace(0, self.z_max, nz)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")

        # Regions
        self.r_outer = 0.010
        self.r_inner = 0.008
        self.z_head = 0.030
        self.z_grip1 = 0.060
        self.z_grip2 = 0.160

        # Temperatures
        self.T_hand = 33.0
        self.T_ice = -18.0
        self.T_amb = 22.0
        self.T_init = -18.0
        self.h_amb = 10.0

        # Build material masks and property arrays
        self._build_materials()

        # Init temperature
        self.T = np.full((nr, nz), self.T_init)

        self.history = []
        self.times = []

    def _build_materials(self):
        """Build material property arrays using vectorized masks."""
        R = self.R
        Z = self.Z

        # Material masks
        if self.design == "A":
            self.head_mask = Z < self.z_head
            self.wall_mask = (Z >= self.z_head) & (Z < self.z_max) & (R >= self.r_inner)
            self.oil_mask = (Z >= self.z_head) & (Z < self.z_max) & (R < self.r_inner)
            self.solid_mask = self.head_mask | self.wall_mask
            self.air_mask = ~self.solid_mask

        else:  # Design B
            self.solid_mask = (Z < self.z_max) & (R <= self.r_outer)
            self.head_mask = Z < self.z_head
            self.wall_mask = (Z >= self.z_head) & (Z < self.z_max) & (R <= self.r_outer)
            self.oil_mask = np.zeros_like(R, dtype=bool)
            self.air_mask = ~self.solid_mask

        # Property arrays
        alum = MATERIALS["aluminum"]
        copper = MATERIALS["copper"]
        oil = MATERIALS["mineral_oil"]

        self.k = np.zeros_like(R)
        self.rho = np.zeros_like(R)
        self.cp = np.zeros_like(R)

        if self.design == "A":
            self.k[self.head_mask] = alum["k"]
            self.rho[self.head_mask] = alum["rho"]
            self.cp[self.head_mask] = alum["cp"]

            self.k[self.wall_mask] = alum["k"]
            self.rho[self.wall_mask] = alum["rho"]
            self.cp[self.wall_mask] = alum["cp"]

            self.k[self.oil_mask] = oil["k"]
            self.rho[self.oil_mask] = oil["rho"]
            self.cp[self.oil_mask] = oil["cp"]
        else:
            self.k[self.solid_mask] = copper["k"]
            self.rho[self.solid_mask] = copper["rho"]
            self.cp[self.solid_mask] = copper["cp"]

        # Air
        self.k[self.air_mask] = 0.026
        self.rho[self.air_mask] = 1.2
        self.cp[self.air_mask] = 1006.0

        self.alpha = self.k / (self.rho * self.cp)

        # Material ID for plotting
        self.mat_id = np.zeros_like(R, dtype=int)
        self.mat_id[self.head_mask] = 1
        if self.design == "A":
            self.mat_id[self.wall_mask] = 1
            self.mat_id[self.oil_mask] = 2
        else:
            self.mat_id[self.wall_mask] = 3

        # Precompute face conductivities (harmonic mean)
        # k at (i+1/2, j) face
        k_safe = np.maximum(self.k, 1e-30)
        self.k_r_face = 2 * k_safe * np.roll(k_safe, -1, axis=0) / (k_safe + np.roll(k_safe, -1, axis=0))
        self.k_z_face = 2 * k_safe * np.roll(k_safe, -1, axis=1) / (k_safe + np.roll(k_safe, -1, axis=1))

    def _compute_dt(self):
        """Stable time step."""
        a_max = np.max(self.alpha[self.solid_mask | self.air_mask])
        dt = 0.4 / (a_max * (1.0 / self.dr**2 + 1.0 / self.dz**2))
        return min(dt, 0.005)

    def _step(self, dt):
        """Vectorized explicit step."""
        T = self.T
        k = self.k
        rho_cp = self.rho * self.cp
        dr = self.dr
        dz = self.dz

        # Shifted arrays
        T_ip = np.roll(T, -1, axis=0)  # T[i+1, j]
        T_im = np.roll(T, 1, axis=0)   # T[i-1, j]
        T_jp = np.roll(T, -1, axis=1)  # T[i, j+1]
        T_jm = np.roll(T, 1, axis=1)   # T[i, j-1]

        k_ip = np.roll(self.k_r_face, 0, axis=0)  # k at i+1/2 face
        k_im = np.roll(self.k_r_face, 1, axis=0)   # k at i-1/2 face
        k_jp = np.roll(self.k_z_face, 0, axis=1)
        k_jm = np.roll(self.k_z_face, 1, axis=1)

        # Radial heat flux: d/dr(k * dT/dr)
        # At i+1/2 face: k_ip * (T_ip - T) / dr
        # At i-1/2 face: k_im * (T - T_im) / dr
        # Net: (k_ip*(T_ip - T) - k_im*(T - T_im)) / dr^2
        q_r = (k_ip * (T_ip - T) / dr - k_im * (T - T_im) / dr) / dr

        # Axial heat flux
        q_z = (k_jp * (T_jp - T) / dz - k_jm * (T - T_jm) / dz) / dz

        # Cylindrical term: (k/r) * dT/dr
        # Approximate: (1/r) * (k_ip*(T_ip-T) + k_im*(T-T_im)) / (2*dr)
        r_safe = np.maximum(self.R, self.dr * 0.5)  # avoid r=0
        q_cyl = (1.0 / r_safe) * (k_ip * (T_ip - T) + k_im * (T - T_im)) / (2.0 * dr)

        # Combined
        dTdt = (q_r + q_z + q_cyl) / rho_cp

        # Only update solid cells
        T_new = T.copy()
        update_mask = self.solid_mask.copy()
        # Don't update boundary cells (handled by BC)
        update_mask[0, :] = False
        update_mask[-1, :] = False
        update_mask[:, 0] = False
        update_mask[:, -1] = False

        T_new[update_mask] = T[update_mask] + dt * dTdt[update_mask]

        # Apply BCs
        T_new = self._apply_bc(T_new)
        self.T = T_new

    def _apply_bc(self, T):
        """Apply boundary conditions."""
        # Axis symmetry (r=0)
        T[0, :] = T[1, :]

        # Hand grip: constant temp on outer cells
        grip_z = (self.Z >= self.z_grip1) & (self.Z <= self.z_grip2)
        grip_cells = self.wall_mask & grip_z
        # Set outermost wall cell in grip region to hand temp
        for j in range(self.nz):
            if grip_z[0, j]:  # check if in grip z-range (any i)
                for i in range(self.nr - 1, -1, -1):
                    if self.solid_mask[i, j]:
                        T[i, j] = self.T_hand
                        break

        # Ice cream contact at bottom
        head_bottom = self.head_mask[:, 0]
        T[:, 0] = np.where(self.solid_mask[:, 0], self.T_ice, self.T_amb)

        # Top boundary
        T[:, -1] = self.T_amb

        return T

    def _get_head_temp(self):
        if np.any(self.head_mask):
            return float(np.mean(self.T[self.head_mask]))
        return float("nan")

    def _get_heat_flux(self):
        """Approximate heat input from hand."""
        total_q = 0.0
        for j in range(self.nz):
            if self.z_grip1 <= self.z[j] <= self.z_grip2:
                for i in range(self.nr - 1, 0, -1):
                    if self.solid_mask[i, j] and self.solid_mask[i - 1, j]:
                        dT = self.T[i, j] - self.T[i - 1, j]
                        q = self.k[i, j] * dT / self.dr
                        area = 2 * np.pi * self.r[i] * self.dz
                        total_q += abs(q) * area
                        break
        return total_q

    def run(self, verbose=True):
        dt = self._compute_dt()
        n_steps = int(self.t_total / dt)
        output_every = max(1, n_steps // 120)  # ~120 snapshots

        if verbose:
            print(f"[Design {self.design}] Grid: {self.nr}x{self.nz}, dt={dt*1000:.3f} ms, steps={n_steps}")

        start = time_module.time()
        t = 0.0
        for step in range(n_steps + 1):
            self._step(dt)
            t += dt

            if step % output_every == 0 or step == n_steps:
                ht = self._get_head_temp()
                q = self._get_heat_flux()
                self.history.append({"t": t, "head_temp": ht, "heat_flux": q, "T_field": self.T.copy()})
                self.times.append(t)
                if verbose:
                    el = time_module.time() - start
                    print(f"  t={t:6.1f}s  head={ht:7.2f}°C  Q={q:6.3f}W  [{el:.1f}s]")

        if verbose:
            print(f"  Done. {len(self.history)} snapshots.")

    def save_results(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        times = np.array(self.times)
        ht = np.array([h["head_temp"] for h in self.history])
        q = np.array([h["heat_flux"] for h in self.history])
        np.savez(os.path.join(outdir, f"design_{self.design}_timeseries.npz"),
                 times=times, head_temp=ht, heat_flux=q)

        # Key field snapshots
        for kt in [0, 5, 10, 30, 60, 120]:
            idx = np.argmin(np.abs(times - kt))
            if abs(times[idx] - kt) < 3.0:
                snap = self.history[idx]
                np.savez(os.path.join(outdir, f"design_{self.design}_field_t{int(snap['t'])}.npz"),
                         T=snap["T_field"], r=self.r, z=self.z, mat_id=self.mat_id, t=snap["t"])

    def plot_field(self, idx, path):
        snap = self.history[idx]
        T = snap["T_field"].copy()
        t = snap["t"]

        T[self.air_mask] = np.nan

        fig, ax = plt.subplots(figsize=(8, 14))
        R_mm = self.R * 1000
        Z_mm = self.Z * 1000

        R_full = np.vstack([-R_mm[::-1, :], R_mm])
        T_full = np.vstack([T[::-1, :], T])

        levels = np.linspace(-18, 33, 52)
        cs = ax.contourf(R_full, Z_mm, T_full, levels=levels, cmap="RdYlBu_r")
        plt.colorbar(cs, ax=ax, shrink=0.5, label="Temperature (°C)")

        name = "Oil-filled Aluminum" if self.design == "A" else "Solid Copper"
        ax.set_title(f"Design {self.design}: {name}\nt = {t:.1f}s", fontsize=13)
        ax.set_xlabel("r (mm)")
        ax.set_ylabel("z (mm)")
        ax.axhline(self.z_grip1 * 1000, color="green", ls="--", alpha=0.4)
        ax.axhline(self.z_grip2 * 1000, color="green", ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def run_comparison(outdir="../results"):
    outdir = os.path.join(os.path.dirname(__file__), outdir)
    os.makedirs(outdir, exist_ok=True)

    results = {}
    for design in ["A", "B"]:
        print(f"\n{'='*60}\nDesign {design}\n{'='*60}")
        solver = ScoopSolverVec(design=design, nr=50, nz=160)
        solver.run(verbose=True)
        solver.save_results(outdir)

        for si in [0, len(solver.history) // 4, len(solver.history) // 2, -1]:
            t_val = solver.history[si]["t"]
            solver.plot_field(si, os.path.join(outdir, f"design_{design}_field_t{int(t_val)}.png"))

        results[design] = {
            "times": np.array(solver.times),
            "head_temp": np.array([h["head_temp"] for h in solver.history]),
            "heat_flux": np.array([h["heat_flux"] for h in solver.history]),
        }

    _plot_comparison(results, outdir)
    _print_summary(results)
    return results


def _plot_comparison(results, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = {"A": "#2196F3", "B": "#FF5722"}
    labels = {"A": "Design A: Oil-filled Aluminum", "B": "Design B: Solid Copper"}

    ax = axes[0]
    for d in ["A", "B"]:
        ax.plot(results[d]["times"], results[d]["head_temp"], label=labels[d],
               color=colors[d], linewidth=2)
    ax.axhline(0, color="gray", ls=":", alpha=0.5, label="0°C")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Head Temperature (°C)")
    ax.set_title("Scoop Head Temperature: Oil-Filled vs Solid Copper")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, max(results["A"]["times"]))

    ax = axes[1]
    for d in ["A", "B"]:
        ax.plot(results[d]["times"], results[d]["heat_flux"], label=labels[d],
               color=colors[d], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat Flux from Hand (W)")
    ax.set_title("Heat Transfer from Hand")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, max(results["A"]["times"]))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_timeseries.png"), dpi=150)
    plt.close()

    # Time to target
    fig, ax = plt.subplots(figsize=(8, 6))
    targets = [-10, -5, 0]
    t2t = {"A": [], "B": []}
    for d in ["A", "B"]:
        for tgt in targets:
            idx = np.where(results[d]["head_temp"] >= tgt)[0]
            t2t[d].append(results[d]["times"][idx[0]] if len(idx) > 0 else np.nan)

    x = np.arange(len(targets))
    w = 0.35
    b1 = ax.bar(x - w/2, t2t["A"], w, label="Oil-filled Aluminum", color=colors["A"])
    b2 = ax.bar(x + w/2, t2t["B"], w, label="Solid Copper", color=colors["B"])
    ax.set_ylabel("Time (s)"); ax.set_xlabel("Target Temperature")
    ax.set_title("Time to Reach Target Temperature")
    ax.set_xticks(x); ax.set_xticklabels([f"{t}°C" for t in targets])
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for b in list(b1) + list(b2):
        h = b.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.1f}s", xy=(b.get_x() + b.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_to_target.png"), dpi=150)
    plt.close()


def _print_summary(results):
    print(f"\n{'='*70}")
    print("SIMULATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<40} {'Design A':>12} {'Design B':>12} {'B/A':>8}")
    print("-" * 72)

    for tgt in [-10, -5, 0]:
        times = {}
        for d in ["A", "B"]:
            idx = np.where(results[d]["head_temp"] >= tgt)[0]
            times[d] = results[d]["times"][idx[0]] if len(idx) > 0 else float("nan")
        ratio = times["B"] / times["A"] if times["A"] and times["A"] > 0 else float("nan")
        print(f"  Time to {tgt:>3}°C{'':>26} {times['A']:>10.1f}s {times['B']:>10.1f}s {ratio:>6.2f}x")

    for d in ["A", "B"]:
        sq = np.mean(results[d]["heat_flux"][-10:])
        print(f"  Steady-state Q_hand (Design {d}): {sq:.2f} W")

    print("=" * 72)


if __name__ == "__main__":
    run_comparison()
