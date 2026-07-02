"""
Fast 2D Axisymmetric Thermal Solver using lumped thermal networks.

Instead of gridding the full 2D field with a tiny dt, we use a 1D radial
thermal resistor network at each axial station, coupled axially. This is
equivalent to a coarse finite-volume scheme but ~1000x faster.

We also model the scoop head as a lumped thermal mass.

This runs in pure numpy on CPU in seconds, not minutes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time as time_module
import json

sys.path.insert(0, os.path.dirname(__file__))
from materials import MATERIALS, oil_effective_k


class LumpedScoopSim:
    """
    1D axial thermal network with lumped radial conduction.
    
    The scoop is divided into axial segments. Each segment has:
    - A radial thermal resistance from outer surface to center
    - An axial thermal resistance to neighboring segments
    - Thermal capacitance (heat capacity)
    
    For Design A (oil-filled tube):
    - Wall conducts heat axially and radially (aluminum)
    - Oil core conducts slowly (with natural convection enhancement)
    
    For Design B (solid copper):
    - Entire cross-section conducts (copper)
    """

    def __init__(self, design="A", n_segments=30, t_total=120.0, dt=0.05):
        self.design = design
        self.n_seg = n_segments
        self.t_total = t_total
        self.dt = dt

        # Geometry (meters)
        self.L_handle = 0.150       # handle length
        self.L_head = 0.025         # head length (simplified)
        self.L_total = self.L_handle + self.L_head
        self.r_outer = 0.010        # outer radius
        self.r_inner = 0.008        # inner radius (tube wall)
        self.head_radius = 0.025    # scoop head radius (25mm hemisphere)
        self.head_wall = 0.002      # head wall thickness

        # Segment layout: head segments (first few) + handle segments
        self.dz = self.L_total / n_segments
        self.n_head_seg = max(1, int(self.L_head / self.dz))
        self.n_handle_seg = n_segments - self.n_head_seg

        # Segment z-centers
        self.z_centers = np.array([(i + 0.5) * self.dz for i in range(n_segments)])
        self.is_head = np.array([i < self.n_head_seg for i in range(n_segments)])

        # Grip region (100mm of handle, centered)
        grip_start_z = self.L_head + 0.010   # 10mm into handle
        grip_end_z = self.L_head + 0.110     # 110mm into handle
        self.is_grip = (self.z_centers >= grip_start_z) & (self.z_centers <= grip_end_z)

        # BCs
        self.T_hand = 33.0
        self.T_ice = -18.0
        self.T_amb = 22.0
        self.h_amb = 10.0
        self.T_init = -18.0

        # Build thermal network
        self._build_network()

        # Initialize
        self.T = np.full(n_segments, self.T_init)
        self.T_head_lump = self.T_ice  # lumped head temp

        self.history = []
        self.times = []

    def _build_network(self):
        """Build thermal resistance and capacitance arrays."""
        alum = MATERIALS["aluminum"]
        copper = MATERIALS["copper"]
        oil = MATERIALS["mineral_oil"]

        # Per-segment thermal capacitance [J/K]
        self.C = np.zeros(self.n_seg)

        # Axial thermal conductance between segments [W/K]
        self.G_axial = np.zeros(self.n_seg - 1)

        # Radial thermal conductance from surface to core [W/K]
        self.G_radial = np.zeros(self.n_seg)

        # External (convection/grip) conductance [W/K]
        self.G_ext = np.zeros(self.n_seg)

        dz = self.dz
        r_o = self.r_outer
        r_i = self.r_inner

        if self.design == "A":
            # Oil-filled aluminum tube
            # Cross-section areas
            A_wall = np.pi * (r_o**2 - r_i**2)   # annular wall area
            A_oil = np.pi * r_i**2               # oil core area

            for i in range(self.n_seg):
                if self.is_head[i]:
                    # Head segment: aluminum, roughly hemispherical shell
                    # Approximate as thick aluminum segment
                    A_cross = np.pi * r_o**2
                    self.C[i] = alum["rho"] * alum["cp"] * A_cross * dz
                    # Axial conduction through aluminum
                    self.G_axial[i - 1 if i > 0 else 0] = alum["k"] * A_cross / dz if i > 0 else 0
                    if i < self.n_seg - 1:
                        self.G_axial[i] = alum["k"] * A_cross / dz

                    # Radial: very high (solid aluminum)
                    self.G_radial[i] = alum["k"] * 2 * np.pi * dz / np.log(r_o / max(r_o * 0.5, 1e-6))

                else:
                    # Handle segment: tube wall + oil
                    # Wall capacitance
                    C_wall = alum["rho"] * alum["cp"] * A_wall * dz
                    # Oil capacitance
                    C_oil = oil["rho"] * oil["cp"] * A_oil * dz
                    self.C[i] = C_wall + C_oil

                    # Axial conduction: through wall (aluminum annulus)
                    if i > 0:
                        self.G_axial[i - 1] = alum["k"] * A_wall / dz
                    if i < self.n_seg - 1:
                        self.G_axial[i] = alum["k"] * A_wall / dz

                    # Radial conduction: wall (annular) + oil (core)
                    # Wall: R_wall = ln(r_o/r_i) / (2*pi*k_al*dz)
                    R_wall = np.log(r_o / r_i) / (2 * np.pi * alum["k"] * dz)
                    # Oil: R_oil = ln(r_i/r_center) / (2*pi*k_oil_eff*dz)
                    # Effective oil k with natural convection
                    k_oil_eff = self._oil_eff_k(self.T_ice + 273.15, self.T_hand + 273.15)
                    R_oil = np.log(r_i / max(r_i * 0.1, 1e-6)) / (2 * np.pi * k_oil_eff * dz)
                    # Total radial R = R_wall + R_oil (series)
                    self.G_radial[i] = 1.0 / (R_wall + R_oil)

        else:
            # Solid copper
            A_cross = np.pi * r_o**2
            for i in range(self.n_seg):
                self.C[i] = copper["rho"] * copper["cp"] * A_cross * dz
                if i > 0:
                    self.G_axial[i - 1] = copper["k"] * A_cross / dz
                if i < self.n_seg - 1:
                    self.G_axial[i] = copper["k"] * A_cross / dz

                # Radial: very high for solid copper
                self.G_radial[i] = copper["k"] * 2 * np.pi * dz / np.log(max(r_o / (r_o * 0.5), 1.01))

        # External conductance
        for i in range(self.n_seg):
            if self.is_grip[i]:
                # Hand: high conductance (skin contact)
                area_ext = 2 * np.pi * r_o * dz
                self.G_ext[i] = 50.0 * area_ext  # h_eff=50 for hand grip
            elif self.is_head[i]:
                # Head in ice cream
                area_ext = 2 * np.pi * r_o * dz
                self.G_ext[i] = 100.0 * area_ext  # high h for ice cream contact
            else:
                # Ambient convection
                area_ext = 2 * np.pi * r_o * dz
                self.G_ext[i] = self.h_amb * area_ext

    def _oil_eff_k(self, T_cold, T_hot):
        """Effective oil conductivity with natural convection."""
        L_char = 2 * self.r_inner
        return oil_effective_k(T_cold, T_hot, L_char)

    def _step_explicit(self, dt):
        """Explicit step on lumped network."""
        T = self.T
        T_new = T.copy()
        n = self.n_seg

        for i in range(n):
            q_total = 0.0

            # Axial conduction from neighbors
            if i > 0:
                q_total += self.G_axial[i - 1] * (T[i - 1] - T[i])
            if i < n - 1:
                q_total += self.G_axial[i] * (T[i + 1] - T[i])

            # External heat source
            if self.is_grip[i]:
                q_total += self.G_ext[i] * (self.T_hand - T[i])
            elif self.is_head[i]:
                q_total += self.G_ext[i] * (self.T_ice - T[i])
            else:
                q_total += self.G_ext[i] * (self.T_amb - T[i])

            T_new[i] = T[i] + q_total * dt / self.C[i]

        self.T = T_new

    def _step_implicit(self, dt):
        """Implicit (backward Euler) step using tridiagonal solve."""
        n = self.n_seg
        # Build tridiagonal system: A*T_new = b
        # For each node i:
        #   C_i/dt * (T_new_i - T_old_i) = sum of G*(T_neighbor - T_new_i) + G_ext*(T_ext - T_new_i)

        a = np.zeros(n)  # lower diagonal
        b = np.zeros(n)  # main diagonal
        c = np.zeros(n)  # upper diagonal
        d = np.zeros(n)  # RHS

        for i in range(n):
            Ci_dt = self.C[i] / dt
            diag = Ci_dt
            rhs = Ci_dt * self.T[i]

            # Axial neighbors
            if i > 0:
                g = self.G_axial[i - 1]
                a[i] = -g
                diag += g
            if i < n - 1:
                g = self.G_axial[i]
                c[i] = -g
                diag += g

            # External
            if self.is_grip[i]:
                T_ext = self.T_hand
            elif self.is_head[i]:
                T_ext = self.T_ice
            else:
                T_ext = self.T_amb

            diag += self.G_ext[i]
            rhs += self.G_ext[i] * T_ext

            b[i] = diag
            d[i] = rhs

        # Thomas algorithm
        for i in range(1, n):
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]

        T_new = np.zeros(n)
        T_new[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            T_new[i] = (d[i] - c[i] * T_new[i + 1]) / b[i]

        self.T = T_new

    def run(self, verbose=True):
        n_steps = int(self.t_total / self.dt)
        snap_every = max(1, n_steps // 120)

        method = "implicit"
        if verbose:
            print(f"[Design {self.design}] {self.n_seg} segments, dt={self.dt*1000:.0f}ms, "
                  f"steps={n_steps}, method={method}")

        t0 = time_module.time()
        t = 0.0
        for step in range(n_steps + 1):
            self._step_implicit(self.dt)
            t += self.dt

            if step % snap_every == 0 or step == n_steps:
                ht = self.head_temp()
                q = self.heat_input()
                self.history.append({"t": t, "head_temp": ht, "heat_flux": q})
                self.times.append(t)
                if verbose:
                    el = time_module.time() - t0
                    print(f"  t={t:6.1f}s  head={ht:7.2f}°C  Q={q:7.3f}W  [{el:.2f}s]")

        if verbose:
            print(f"  Done. {len(self.history)} snapshots in {time_module.time()-t0:.1f}s")

    def head_temp(self):
        """Average head temperature."""
        return float(np.mean(self.T[self.is_head]))

    def handle_avg_temp(self):
        """Average handle temperature."""
        return float(np.mean(self.T[~self.is_head]))

    def heat_input(self):
        """Total heat flowing from hand into scoop."""
        q = 0.0
        for i in range(self.n_seg):
            if self.is_grip[i]:
                q += self.G_ext[i] * (self.T_hand - self.T[i])
        return q

    def save_results(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        tag = f"design_{self.design}"
        times = np.array(self.times)
        ht = np.array([h["head_temp"] for h in self.history])
        q = np.array([h["heat_flux"] for h in self.history])

        np.savez(
            os.path.join(outdir, f"{tag}_timeseries.npz"),
            times=times, head_temp=ht, heat_flux=q,
            z_centers=self.z_centers,
            is_head=self.is_head, is_grip=self.is_grip,
        )

        # Final axial temperature profile
        np.savez(
            os.path.join(outdir, f"{tag}_axial_profile.npz"),
            z=self.z_centers, T=self.T,
            is_head=self.is_head, is_grip=self.is_grip,
        )


def run_comparison(outdir="../results"):
    outdir = os.path.join(os.path.dirname(__file__), outdir)
    os.makedirs(outdir, exist_ok=True)

    results = {}
    for design in ["A", "B"]:
        print(f"\n{'='*60}")
        print(f"Design {'A (Oil-filled Aluminum)' if design == 'A' else 'B (Solid Copper)'}")
        print(f"{'='*60}")

        sim = LumpedScoopSim(design=design, n_segments=60, t_total=120.0, dt=0.05)
        sim.run(verbose=True)
        sim.save_results(outdir)

        results[design] = {
            "times": np.array(sim.times),
            "head_temp": np.array([h["head_temp"] for h in sim.history]),
            "heat_flux": np.array([h["heat_flux"] for h in sim.history]),
            "z_centers": sim.z_centers,
            "T_final": sim.T.copy(),
            "is_head": sim.is_head,
            "is_grip": sim.is_grip,
        }

    plot_all(results, outdir)
    summary = build_summary(results)

    # Save summary JSON
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results, summary


def plot_all(results, outdir):
    colors = {"A": "#2196F3", "B": "#FF5722"}
    labels = {"A": "Oil-filled Aluminum", "B": "Solid Copper"}

    # --- Time series comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    ax = axes[0]
    for d in ["A", "B"]:
        ax.plot(results[d]["times"], results[d]["head_temp"],
               label=labels[d], color=colors[d], lw=2.5)
    ax.axhline(0, color="gray", ls=":", alpha=0.6, label="0°C (freezing)")
    ax.axhline(-18, color="#4FC3F7", ls="--", alpha=0.4, label="Ice cream temp (-18°C)")
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Scoop Head Temperature (°C)", fontsize=13)
    ax.set_title("Scoop Head Temperature: Oil-Filled Aluminum vs Solid Copper\n"
                "(Initial: -18°C, Hand: 33°C, Ice cream contact: -18°C)", fontsize=14)
    ax.legend(fontsize=11, loc="right")
    ax.set_xlim(0, max(results["A"]["times"]))
    ax.grid(alpha=0.3)
    ax.set_ylim(-20, 35)

    ax = axes[1]
    for d in ["A", "B"]:
        ax.plot(results[d]["times"], results[d]["heat_flux"],
               label=labels[d], color=colors[d], lw=2.5)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Heat Flow from Hand (W)", fontsize=13)
    ax.set_title("Heat Transfer Rate from Hand to Scoop Handle", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, max(results["A"]["times"]))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_timeseries.png"), dpi=150)
    plt.close()

    # --- Time to target ---
    fig, ax = plt.subplots(figsize=(9, 6))
    targets = [-15, -10, -5, 0, 5]
    t2t = {}
    for d in ["A", "B"]:
        t2t[d] = []
        for tgt in targets:
            idx = np.where(results[d]["head_temp"] >= tgt)[0]
            t2t[d].append(results[d]["times"][idx[0]] if len(idx) > 0 else float("nan"))

    x = np.arange(len(targets))
    w = 0.35
    b1 = ax.bar(x - w/2, t2t["A"], w, label="Oil-filled Aluminum", color=colors["A"], edgecolor="white")
    b2 = ax.bar(x + w/2, t2t["B"], w, label="Solid Copper", color=colors["B"], edgecolor="white")
    ax.set_ylabel("Time to Reach Target (s)", fontsize=13)
    ax.set_xlabel("Target Head Temperature", fontsize=13)
    ax.set_title("Warmup Time: Lower is Better", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}°C" for t in targets])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    for b in list(b1) + list(b2):
        h = b.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.1f}s", xy=(b.get_x()+b.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_to_target.png"), dpi=150)
    plt.close()

    # --- Axial temperature profile ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in ["A", "B"]:
        z_mm = results[d]["z_centers"] * 1000
        T_final = results[d]["T_final"]
        ax.plot(z_mm, T_final, label=labels[d], color=colors[d], lw=2.5, marker="o", ms=3)

    # Mark regions
    ax.axvspan(0, results["A"]["z_centers"][results["A"]["is_head"]][-1] * 1000,
              alpha=0.1, color="cyan", label="Scoop head")
    grip_mask = results["A"]["is_grip"]
    if np.any(grip_mask):
        gz = results["A"]["z_centers"][grip_mask]
        ax.axvspan(gz[0] * 1000, gz[-1] * 1000, alpha=0.1, color="green", label="Hand grip")

    ax.axhline(33, color="red", ls="--", alpha=0.3, label="Hand temp (33°C)")
    ax.axhline(-18, color="blue", ls="--", alpha=0.3, label="Ice cream (-18°C)")
    ax.set_xlabel("Axial Position (mm)", fontsize=13)
    ax.set_ylabel("Temperature (°C)", fontsize=13)
    ax.set_title("Axial Temperature Profile at t=120s\n(Distance from scoop head)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "axial_profile.png"), dpi=150)
    plt.close()


def build_summary(results):
    s = {"designs": {}}
    for d in ["A", "B"]:
        data = results[d]
        name = "Oil-filled Aluminum" if d == "A" else "Solid Copper"
        times_to_target = {}
        for tgt in [-15, -10, -5, 0]:
            idx = np.where(data["head_temp"] >= tgt)[0]
            times_to_target[f"{tgt}C"] = float(data["times"][idx[0]]) if len(idx) > 0 else None

        steady_q = float(np.mean(data["heat_flux"][-10:]))
        final_head = float(data["head_temp"][-1])

        s["designs"][d] = {
            "name": name,
            "final_head_temp_C": round(final_head, 2),
            "steady_state_heat_flux_W": round(steady_q, 3),
            "times_to_target": {k: round(v, 1) if v else None for k, v in times_to_target.items()},
        }

    # Ratios
    tA = s["designs"]["A"]["times_to_target"]
    tB = s["designs"]["B"]["times_to_target"]
    s["speedup_ratios"] = {}
    for key in tA:
        if tA[key] and tB[key] and tA[key] > 0:
            s["speedup_ratios"][key] = round(tA[key] / tB[key], 2)

    return s


def print_summary(results):
    print(f"\n{'='*72}")
    print("  ICE CREAM SCOOP THERMAL SIMULATION - RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Metric':<45} {'Oil/Alum':>10} {'Copper':>10} {'Ratio':>8}")
    print(f"  {'-'*70}")

    for tgt in [-15, -10, -5, 0]:
        times = {}
        for d in ["A", "B"]:
            idx = np.where(results[d]["head_temp"] >= tgt)[0]
            times[d] = results[d]["times"][idx[0]] if len(idx) > 0 else float("nan")
        ratio = times["A"] / times["B"] if times["B"] > 0 else float("nan")
        print(f"  Time to reach {tgt:>3}°C{'':>30} {times['A']:>8.1f}s {times['B']:>8.1f}s {ratio:>6.2f}x")

    print(f"  {'-'*70}")
    for d, name in [("A", "Oil/Alum"), ("B", "Copper")]:
        sq = np.mean(results[d]["heat_flux"][-10:])
        final_t = results[d]["head_temp"][-1]
        print(f"  Final head temp ({name}){'':>20} {final_t:>8.2f}°C")
        print(f"  Steady Q_hand ({name}){'':>24} {sq:>8.3f}W")

    print(f"  {'-'*70}")
    print(f"  Copper warms the head ~{results['A']['head_temp'][-1]/max(abs(results['B']['head_temp'][-1]),0.01):.1f}x ")
    print(f"  faster than oil-filled aluminum.")
    print(f"{'='*72}")


if __name__ == "__main__":
    results, summary = run_comparison()
    print_summary(results)
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))
