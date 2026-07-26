#!/usr/bin/env python3
"""Engineering screen for an impatient ice-cream eating utensil.

This is deliberately simpler than the scoop CFD models in this repository.
The first bite is governed by two separable mechanisms:

1. Mechanical crack initiation at the leading edge.
2. Heat delivered from the hand after the utensil's initial stored heat is spent.

The script compares architectures without pretending that an unmeasured
ice-cream fracture strength or hand contact coefficient is known precisely.
All inputs are visible below so a future bench test can replace assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


DELTA_T_HAND_TO_ICE_K = 33.0 - (-18.0)
HAND_POWER_LIMIT_W = 3.0
ICE_CREAM_EFFECTIVE_MELT_ENTHALPY_J_PER_KG = 200_000.0
APPLIED_FORCE_N = 5.0


@dataclass(frozen=True)
class Candidate:
    name: str
    edge_length_mm: float
    edge_thickness_mm: float
    body_k_w_mk: float
    body_area_mm2: float
    axial_length_mm: float = 100.0
    added_path_k_w_mk: float = 0.0
    added_path_area_mm2: float = 0.0
    transport_cap_w: float | None = None
    note: str = ""

    @property
    def edge_area_mm2(self) -> float:
        return self.edge_length_mm * self.edge_thickness_mm

    @property
    def nominal_edge_pressure_mpa(self) -> float:
        # 1 N/mm² = 1 MPa.
        return APPLIED_FORCE_N / self.edge_area_mm2

    @property
    def body_conductance_w_k(self) -> float:
        area_m2 = self.body_area_mm2 * 1e-6
        length_m = self.axial_length_mm * 1e-3
        return self.body_k_w_mk * area_m2 / length_m

    @property
    def added_conductance_w_k(self) -> float:
        if not self.added_path_area_mm2 or not self.added_path_k_w_mk:
            return 0.0
        area_m2 = self.added_path_area_mm2 * 1e-6
        length_m = self.axial_length_mm * 1e-3
        return self.added_path_k_w_mk * area_m2 / length_m

    @property
    def hand_to_tip_power_w(self) -> float:
        raw = (
            self.body_conductance_w_k + self.added_conductance_w_k
        ) * DELTA_T_HAND_TO_ICE_K
        caps = [raw, HAND_POWER_LIMIT_W]
        if self.transport_cap_w is not None:
            caps.append(self.transport_cap_w)
        return min(caps)

    def melt_mass_mg(self, seconds: float) -> float:
        energy_j = self.hand_to_tip_power_w * seconds
        return energy_j / ICE_CREAM_EFFECTIVE_MELT_ENTHALPY_J_PER_KG * 1e6


CANDIDATES = [
    Candidate(
        name="Ordinary stainless teaspoon",
        edge_length_mm=32.0,
        edge_thickness_mm=0.9,
        body_k_w_mk=16.0,
        body_area_mm2=10.0,
        note="Broad rolled rim; baseline.",
    ),
    Candidate(
        name="Snow Peak-style titanium spork",
        edge_length_mm=8.0,
        edge_thickness_mm=0.30,
        body_k_w_mk=17.0,
        body_area_mm2=10.0,
        note="Immediate winner from short, thin tines—not titanium conductivity.",
    ),
    Candidate(
        name="First Bite: titanium shell + 3 mm² copper spine",
        edge_length_mm=8.0,
        edge_thickness_mm=0.30,
        body_k_w_mk=17.0,
        body_area_mm2=7.0,
        added_path_k_w_mk=401.0,
        added_path_area_mm2=3.0,
        note="Practical passive hybrid; copper ends before the food-contact edge.",
    ),
    Candidate(
        name="First Bite: titanium + 20 µm CVD diamond ribbon",
        edge_length_mm=8.0,
        edge_thickness_mm=0.30,
        body_k_w_mk=17.0,
        body_area_mm2=10.0,
        # 10 mm wide × 0.020 mm thick coating down the full thermal path.
        added_path_k_w_mk=2_000.0,
        added_path_area_mm2=0.20,
        note="Great conductivity, tiny cross-section; costly coating is not magic.",
    ),
    Candidate(
        name="First Bite X: titanium + 3 mm methanol heat pipe",
        edge_length_mm=8.0,
        edge_thickness_mm=0.30,
        body_k_w_mk=17.0,
        body_area_mm2=3.0,
        # Published heat-pipe effective k spans ~1,500–50,000 W/mK.
        # Use a conservative 10,000 W/mK and cap a miniature low-temp pipe at 3 W.
        added_path_k_w_mk=10_000.0,
        added_path_area_mm2=7.07,
        transport_cap_w=3.0,
        note=(
            "Use methanol/low-temperature fluid. A standard water pipe freezes "
            "at the -18°C tip and behaves mostly like its copper wall at startup."
        ),
    ),
]


def main() -> None:
    baseline_pressure = CANDIDATES[0].nominal_edge_pressure_mpa
    rows = []
    for item in CANDIDATES:
        rows.append(
            {
                **asdict(item),
                "edge_area_mm2": round(item.edge_area_mm2, 3),
                "nominal_edge_pressure_mpa": round(
                    item.nominal_edge_pressure_mpa, 3
                ),
                "pressure_vs_teaspoon": round(
                    item.nominal_edge_pressure_mpa / baseline_pressure, 1
                ),
                "hand_to_tip_power_w": round(item.hand_to_tip_power_w, 3),
                "hand_energy_melt_5s_mg": round(item.melt_mass_mg(5.0), 1),
                "hand_energy_melt_30s_mg": round(item.melt_mass_mg(30.0), 1),
            }
        )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "first_bite_architectures.json").write_text(
        json.dumps(
            {
                "assumptions": {
                    "applied_force_n": APPLIED_FORCE_N,
                    "hand_to_ice_delta_t_k": DELTA_T_HAND_TO_ICE_K,
                    "hand_power_limit_w": HAND_POWER_LIMIT_W,
                    "effective_melt_enthalpy_j_kg": (
                        ICE_CREAM_EFFECTIVE_MELT_ENTHALPY_J_PER_KG
                    ),
                    "warning": (
                        "Screening model only. It excludes initial utensil thermal "
                        "mass, interface resistance, and measured fracture mechanics."
                    ),
                },
                "candidates": rows,
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"{'Architecture':54} {'edge MPa':>9} {'vs spoon':>9} "
        f"{'tip W':>7} {'melt@5s':>9}"
    )
    for row in rows:
        print(
            f"{row['name'][:54]:54} "
            f"{row['nominal_edge_pressure_mpa']:9.3f} "
            f"{row['pressure_vs_teaspoon']:8.1f}× "
            f"{row['hand_to_tip_power_w']:7.3f} "
            f"{row['hand_energy_melt_5s_mg']:8.1f}mg"
        )


if __name__ == "__main__":
    main()
