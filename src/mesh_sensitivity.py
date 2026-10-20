"""
Mesh sensitivity study for the 2D pouch-cell thermal model (Appendix A, Table A.2).

Reuses the model options and production mesh (var_pts) of
src.optimisation.run_latin_hypercube_doe, the function that generated the
800-case dataset, so this checks the mesh actually used for the results.

Motivation: the in-plane mesh is a FIXED 10x4 (y, z) node grid regardless of cell
width/height, which vary by up to 6x across the DoE bounds (Table 1). This script
checks whether OF2 (thermal gradient) is converged with respect to that fixed grid at
the two geometric extremes of the DoE (smallest and largest cell footprint), under the
NT topology with tab cooling.

Load profile: this study uses a constant-current 2C discharge, matching the protocol
used for the specific-energy objective but solved on the 2D spatially-resolved model.
The transient drive cycles are not used here: mesh convergence of dT_max is a property
of the spatial discretisation, and a constant-current discharge is markedly more
numerically stable than a transient load (the WLTP interpolant variant of this study
stalled in IDAKLU without converging).

Outputs (under outputs/): mesh_sensitivity_results_Tab.csv

Usage (from the repository root): python -m src.mesh_sensitivity
"""
import time
import numpy as np
import pandas as pd
import pybamm

from src.optimisation import build_dynamic_topology
from src.config import get_base_parameters
from src.paths import output_path

# Identical to the model options in src.optimisation.run_latin_hypercube_doe
MODEL_OPTIONS = {
    "cell geometry": "pouch",
    "thermal": "x-lumped",
    "current collector": "potential pair",
    "dimensionality": 2,
}

# Mesh levels to compare. "baseline" is the production var_pts of
# src.optimisation.run_latin_hypercube_doe. The refined level only
# increases the in-plane (y, z) resolution, since x/r are not geometry-dependent
# in the DoE (thickness factor and current collector thickness are 1D-lumped
# through-plane quantities, not spatially resolved in y/z).
#
# A 4x level (y=40, z=16) was also trialled but is not included: it failed to
# converge in IDAKLU, running >11 h without completing on the smallest geometry
# while the 2x level solved in ~105 s (Appendix A.3).
MESH_LEVELS = {
    "baseline (production, y=10 z=4)": {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 10, "z": 4},
    "refined 2x (y=20 z=8)":           {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 20, "z": 8},
}

# Cell width/height extremes of the DoE bounds (Table 1): 0.10-0.60 m and
# 0.05-0.30 m respectively. Physical in-plane resolution (mm/node) varies by
# the same ~6x factor across these bounds since node count is fixed.
TEST_GEOMETRIES = {
    "small (0.10m x 0.05m)": {"cell_w": 0.10, "cell_h": 0.05},
    "large (0.60m x 0.30m)": {"cell_w": 0.60, "cell_h": 0.30},
}

# Mid-range values for the remaining continuous DoE parameters (Table 1 bounds),
# held fixed so only geometry and mesh resolution vary between runs.
FIXED_PARAMS = {
    "thick_factor": 1.0,
    "L_ccp": 20e-6,
    "L_ccn": 10e-6,
    "pos_p": 0.5,
    "pos_n": 0.5,
    "w_tab": 0.5,
}

TOPOLOGY = "NT"  # topology with the largest mean thermal gradient (Section 3.2)

# Cooling settings of the mesh study. Tab cooling: 750 W.m-2.K-1 at both tabs, with
# adiabatic current-collector surfaces.
COOLING_CONDITIONS = {
    "Surface": {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 5.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Total heat transfer coefficient [W.m-2.K-1]": 0.0,
    },
    "Tab": {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 750.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 750.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0.0,
    },
}


def run_single_case(cell_w, cell_h, var_pts, cooling_htc, temp_ambient=298.15):
    """
    Builds and solves one 2C constant-current discharge on the 2D model,
    replicating the model and parameter setup of
    src.optimisation.run_latin_hypercube_doe, then extracts OF2 (max spatial
    thermal gradient) with the same definition.
    Returns (thermal_grad_K, solve_time_s).
    """
    model = pybamm.lithium_ion.DFN(options=MODEL_OPTIONS)
    coords, _, _ = build_dynamic_topology(
        cell_w, cell_h, FIXED_PARAMS["pos_p"], FIXED_PARAMS["pos_n"], FIXED_PARAMS["w_tab"], TOPOLOGY
    )

    base_p = pybamm.ParameterValues("Ai2020")
    W_base = base_p["Electrode width [m]"]
    H_base = base_p["Electrode height [m]"]
    base_capacity_Ah = base_p["Nominal cell capacity [A.h]"]

    thick_factor = FIXED_PARAMS["thick_factor"]
    L_n_new = base_p["Negative electrode thickness [m]"] * thick_factor
    L_p_new = base_p["Positive electrode thickness [m]"] * thick_factor

    param = get_base_parameters(L_n=L_n_new, L_p=L_p_new)

    defaults = {
        "Ambient temperature [K]": temp_ambient,
        "Positive current collector thickness [m]": FIXED_PARAMS["L_ccp"],
        "Negative current collector thickness [m]": FIXED_PARAMS["L_ccn"],
    }
    defaults.update(cooling_htc)
    defaults.update(coords)
    param.update(defaults, check_already_exists=False)

    area_scaling = (cell_w * cell_h) / (W_base * H_base)
    dynamic_capacity = base_capacity_Ah * thick_factor * area_scaling
    n_parallel = param["Number of electrodes connected in parallel to make a cell"]

    # Constant-current 2C discharge. The current magnitude follows the
    # specific-energy protocol (capacity * 2.0),
    # divided across the parallel electrode layers as the 2D model expects.
    # Stated in amps rather than as a "2C" string so the rate is derived from
    # the geometry-scaled dynamic capacity, not from nominal cell capacity.
    discharge_current_A = (2.0 * dynamic_capacity) / n_parallel

    solver_2d = pybamm.IDAKLUSolver(rtol=1e-4, atol=1e-4)
    experiment = pybamm.Experiment(
        [f"Discharge at {discharge_current_A:.4f} A until 3.0V"], period="30 seconds"
    )
    sim_2d = pybamm.Simulation(model, experiment=experiment, parameter_values=param.copy(), var_pts=var_pts, solver=solver_2d)

    t0 = time.time()
    sol_2d = sim_2d.solve()
    solve_time_s = time.time() - t0

    T_3d_array = sol_2d["X-averaged cell temperature [K]"].entries
    T_max = np.max(T_3d_array, axis=(0, 1))
    T_min = np.min(T_3d_array, axis=(0, 1))
    thermal_grad = np.max(T_max - T_min)

    return float(thermal_grad), solve_time_s


def run_mesh_sensitivity_study(cooling="Tab"):
    """
    Runs every (geometry x mesh level) combination under the given cooling
    condition and the NT / 2C discharge setup, and saves a results CSV with the
    ΔT_max of each mesh, its change from the production mesh, and the solve time.
    """
    print("\n" + "=" * 70)
    print(f"MESH SENSITIVITY STUDY: NT topology, {cooling} cooling, 2C discharge")
    print("=" * 70)

    cooling_htc = COOLING_CONDITIONS[cooling]
    rows = []

    for geom_label, geom in TEST_GEOMETRIES.items():
        for mesh_label, var_pts in MESH_LEVELS.items():
            n_inplane = var_pts["y"] * var_pts["z"]
            print(f"\n-> {geom_label} | {mesh_label} (y*z = {n_inplane} nodes)")
            thermal_grad, solve_time_s = run_single_case(geom["cell_w"], geom["cell_h"], var_pts, cooling_htc)
            print(f"   dT_max = {thermal_grad:.5f} K   (solve time: {solve_time_s:.1f} s)")

            rows.append({
                "geometry": geom_label, "cell_w_m": geom["cell_w"], "cell_h_m": geom["cell_h"],
                "mesh_level": mesh_label, "y_nodes": var_pts["y"], "z_nodes": var_pts["z"],
                "thermal_grad_K": thermal_grad, "solve_time_s": solve_time_s,
            })

    df = pd.DataFrame(rows)

    # % change relative to the baseline (production) mesh, per geometry
    df["pct_change_vs_baseline"] = np.nan
    for geom_label in TEST_GEOMETRIES:
        mask = df["geometry"] == geom_label
        baseline_val = df.loc[mask & (df["mesh_level"] == list(MESH_LEVELS.keys())[0]), "thermal_grad_K"].iloc[0]
        df.loc[mask, "pct_change_vs_baseline"] = 100.0 * (df.loc[mask, "thermal_grad_K"] - baseline_val) / baseline_val

    out_csv = output_path(f"mesh_sensitivity_results_{cooling}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    print("\n" + "-" * 70)
    print(df[["geometry", "mesh_level", "thermal_grad_K", "solve_time_s", "pct_change_vs_baseline"]].to_string(index=False))
    print("-" * 70)

    return df


if __name__ == "__main__":
    # NT + tab cooling: the pairing with the lowest uncalibrated coverage (Section 3.1)
    # and the largest mean thermal gradient (Section 3.2), evaluated here on a 2C
    # discharge (see module docstring for why the drive cycles are not used).
    run_mesh_sensitivity_study(cooling="Tab")
