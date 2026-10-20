"""
Design of experiments (Section 2.1): the 200-point pack-feasible Latin hypercube, the
cell geometry builder, the vehicle model and pack sizing that convert a drive cycle into
cell current, and the 2D x-lumped PyBaMM simulations that produce the objectives.

Each (geometry, topology, cooling) case is independent and writes one shard to
outputs/shards/<cooling>/; src.compile_dataset collects the shards into the dataset files.

Usage (from the repository root):
    python -m src.optimisation --cooling Tab --topology NT --run-id 5   # one case
    python -m src.optimisation --cooling Surface                        # all 400 cases
"""
import argparse
import os
import numpy as np
import pandas as pd
import pybamm
from scipy.stats import qmc
from src.metrics import calculate_cell_mass
from src.config import get_base_parameters
from src.paths import US06_FILE, WLTP_FILE, WLTP_SOURCE, SHARD_DIR


def read_wltp_class3():
    """WLTC Class 3 sheet of the UNECE WLTP DHC document 12-07 (see README)."""
    if not os.path.exists(WLTP_FILE):
        raise FileNotFoundError(
            f"{WLTP_FILE} not found. Download WLTP-DHC-12-07e.xls (UNECE WLTP DHC, "
            f"12th session) from {WLTP_SOURCE} and save it in data/inputs/.")
    return pd.read_excel(WLTP_FILE, sheet_name="WLTC_class_3", engine="xlrd", skiprows=7, header=None)

# ==============================================================================
# 1. 8-DIMENSIONAL LATIN HYPERCUBE GENERATOR
# ==============================================================================
def get_bounded_lhs_samples(num_samples=200, max_c_rate=3.0):
    """
    Generates an over-sampled LHS pool, applies strict 80-120kWh pack sizing constraints,
    and returns a pure stratified subset without breaking the solver loop.
    """
    print(f"--- Pre-filtering LHS space for 80-120 kWh boundaries ---")

    # 1. Compute absolute peak pack power once
    _, P_pack = vehicle_power_profile("WLTP")
    max_pack_power_W = np.max(P_pack)

    # 2. Extract Baselines
    base_p = pybamm.ParameterValues("Ai2020")
    W_base, H_base = base_p["Electrode width [m]"], base_p["Electrode height [m]"]
    base_capacity_Ah = base_p["Nominal cell capacity [A.h]"]
    V_nom, N_s = 3.6, int(800.0 / 3.6)

    # 3. Generate Over-Sampled Pool
    pool_size = num_samples * 50
    sampler = qmc.LatinHypercube(d=8, seed=42)
    sample_pool = sampler.random(n=pool_size)

    l_bounds = [0.5, 10e-6, 5e-6, 0.10, 0.05, 0.15, 0.15, 0.10]
    u_bounds = [2.0, 30e-6, 15e-6, 0.60, 0.30, 0.85, 0.85, 0.99]
    scaled_pool = qmc.scale(sample_pool, l_bounds, u_bounds)

    valid_samples = []

    # 4. Filter logic
    for point in scaled_pool:
        thick_factor, _, _, cell_w, cell_h, _, _, _ = point

        area_scaling = (cell_w * cell_h) / (W_base * H_base)
        true_dynamic_cap = base_capacity_Ah * thick_factor * area_scaling

        N_p_energy = int(np.ceil((80.0 * 1000) / (true_dynamic_cap * V_nom * N_s)))
        max_cell_current = max_c_rate * true_dynamic_cap
        N_p_power = int(np.ceil(max_pack_power_W / (N_s * V_nom * max_cell_current)))

        N_p = max(N_p_energy, N_p_power)
        actual_pack_kwh = (N_s * N_p * true_dynamic_cap * V_nom) / 1000.0

        if 80.0 <= actual_pack_kwh <= 120.0:
            valid_samples.append(point)

        if len(valid_samples) == num_samples:
            break

    if len(valid_samples) < num_samples:
        raise ValueError("LHS Pool exhausted. Loosen your bounds or increase pool size.")

    return np.array(valid_samples)

# ==============================================================================
# 2. DYNAMIC GEOMETRY BUILDER
# ==============================================================================
def build_dynamic_topology(W, H, pos_frac_p, pos_frac_n, w_tab_frac, topo_label):
    """Calculates exact physical dimensions [m] based on explicit Width and Height."""
    tab_w = W * w_tab_frac

    def clip_tab(center_y):
        return np.clip(center_y, tab_w / 2, W - (tab_w / 2))

    y_p = clip_tab(W * pos_frac_p)
    y_n = clip_tab(W * pos_frac_n)

    coords = {
        "Cell width [m]": W, "Electrode width [m]": W,
        "Cell height [m]": H, "Electrode height [m]": H,
        "Positive tab width [m]": tab_w, "Negative tab width [m]": tab_w,
    }

    if topo_label == "NT":
        coords.update({
            "Positive tab centre y-coordinate [m]": y_p, "Positive tab centre z-coordinate [m]": H,
            "Negative tab centre y-coordinate [m]": y_n, "Negative tab centre z-coordinate [m]": H,
        })
    elif topo_label == "CT":
        coords.update({
            "Positive tab centre y-coordinate [m]": y_p, "Positive tab centre z-coordinate [m]": H,
            "Negative tab centre y-coordinate [m]": y_n, "Negative tab centre z-coordinate [m]": 0.0,
        })
    else:
        raise ValueError(f"Unknown topology {topo_label!r}; expected 'CT' or 'NT'.")

    return coords, W, H

# ==============================================================================
# 3. 1D LONGITUDINAL VEHICLE DYNAMICS AND PACK SIZING
# ==============================================================================
def vehicle_power_profile(cycle_type="US06"):
    """
    Pack power demand [W] of the reference vehicle over the US06 or WLTP (Class 3) speed
    trace: propulsion power divided by, and regenerative power multiplied by, the
    drivetrain efficiency. Returns (time [s], power [W]).
    """
    # 1. Load Data
    if cycle_type == "US06":
        df_cycle = pd.read_csv(US06_FILE, skiprows=2, sep=r'\s+', header=None)
        time_raw = df_cycle.iloc[:, 0].values
        speed_ms = (pd.Series(df_cycle.iloc[:, 1].values).rolling(5, min_periods=1).mean().fillna(0).values) * 0.44704

    elif cycle_type in ["WLTP", "WLTP_FastCharge"]:
        df_cycle = read_wltp_class3()
        time_raw = df_cycle.iloc[:, 2].values
        speed_ms = (pd.Series(df_cycle.iloc[:, 4].values).rolling(5, min_periods=1).mean().fillna(0).values) / 3.6

    else:
        raise ValueError(f"Unknown cycle {cycle_type!r}")

    # 2. Vehicle Physics (Medium EV SUV)
    m_veh = 2000.0
    Cd = 0.24
    A_front = 2.2
    Crr = 0.015
    rho_air = 1.225
    eta_dt = 0.90

    accel = np.gradient(speed_ms, time_raw)

    F_aero = 0.5 * rho_air * Cd * A_front * (speed_ms ** 2)
    F_roll = np.ones_like(speed_ms) * (Crr * m_veh * 9.81)
    F_roll[speed_ms == 0] = 0.0
    F_inertia = m_veh * accel

    P_wheels = (F_aero + F_roll + F_inertia) * speed_ms

    # 3. Battery Pack Power Demand [W]
    P_pack = np.zeros_like(P_wheels)
    P_pack[P_wheels > 0] = P_wheels[P_wheels > 0] / eta_dt  # Propulsion
    P_pack[P_wheels < 0] = P_wheels[P_wheels < 0] * eta_dt  # Regen

    return time_raw, P_pack

def parallel_strings(dynamic_capacity, max_pack_power_W, max_c_rate=5.0, target_pack_kwh=80.0, pack_voltage=800.0):
    """
    Dynamic pack sizing around a cell of the given capacity [A.h] (scalar or array):
    cells in series for the pack voltage, and the smallest number of parallel strings
    that meets both the pack energy target and the peak-power C-rate limit.
    Returns (N_s, N_p).
    """
    V_nom = 3.6
    N_s = int(pack_voltage / V_nom)

    # dynamic_capacity already includes the thickness and area scaling.
    N_p_energy = np.ceil((target_pack_kwh * 1000) / (dynamic_capacity * V_nom * N_s))

    max_allowable_cell_current = max_c_rate * dynamic_capacity
    N_p_power = np.ceil(max_pack_power_W / (N_s * V_nom * max_allowable_cell_current))

    return N_s, np.maximum(N_p_energy, N_p_power)

def generate_drive_cycle(cycle_type="US06", dynamic_capacity=5.0, max_c_rate=5.0, target_pack_kwh=80.0, pack_voltage=800.0, n_parallel=1):
    """
    Cell current [A] per electrode layer over the US06 or WLTP cycle, for a cell of the
    given capacity in a pack sized around it. Returns (time [s], current [A]).
    """
    time_raw, P_pack = vehicle_power_profile(cycle_type)

    # 2. Dynamic Pack Sizing
    V_nom = 3.6
    N_s, N_p = parallel_strings(dynamic_capacity, np.max(P_pack), max_c_rate, target_pack_kwh, pack_voltage)
    total_cells = N_s * N_p

    I_dynamic = (P_pack / total_cells) / (V_nom * n_parallel)

    return time_raw, I_dynamic

def cell_capacity(thick_factor, cell_w, cell_h):
    """Nominal capacity [A.h] of a design: the Ai2020 cell scaled by electrode thickness and area, as in the DoE."""
    base_p = pybamm.ParameterValues("Ai2020")
    area_scaling = (cell_w * cell_h) / (base_p["Electrode width [m]"] * base_p["Electrode height [m]"])
    return base_p["Nominal cell capacity [A.h]"] * thick_factor * area_scaling

def pack_sizing_scale(cycle_type, dynamic_capacity, max_c_rate=5.0, target_pack_kwh=80.0, pack_voltage=800.0):
    """
    RMS cell C-rate over a drive cycle for cells of the given capacity [A.h] (scalar or
    array), from the same vehicle model and pack sizing as generate_drive_cycle. The cell
    C-rate is P(t) / E_installed, so the scale is P_rms / E_installed and jumps wherever the
    number of parallel strings changes (Appendix C).
    """
    _, P_pack = vehicle_power_profile(cycle_type)
    V_nom = 3.6
    N_s, N_p = parallel_strings(dynamic_capacity, np.max(P_pack), max_c_rate, target_pack_kwh, pack_voltage)
    return np.sqrt(np.mean(P_pack ** 2)) / (N_s * N_p * V_nom * np.asarray(dynamic_capacity))

# ==============================================================================
# 3b. SWELLING OBJECTIVE (POST-PROCESSED FROM THE SOLVED CONCENTRATIONS)
# ==============================================================================
def thickness_change_from_initial(sol, param):
    """
    Through-plane thickness change of the cell relative to its own state at the first
    time step [m], resolved over the cell face: shape (y, z, t) in 2D, (z, t) in 1D.

    Each electrode's effective volumetric strain follows from the partial molar volume,
    applied to the particle-volume-averaged, electrode-averaged lithium concentration;
    the two electrodes are summed over all parallel layers. Swelling is post-processed,
    so it does not feed back into the electrochemical or thermal solution.
    """
    dL = 0.0
    for domain, Domain in (("negative", "Negative"), ("positive", "Positive")):
        c = sol[f"Average {domain} particle concentration [mol.m-3]"].entries
        strain = (float(param[f"{Domain} electrode active material volume fraction"])
                  * float(param[f"{Domain} electrode partial molar volume [m3.mol-1]"])
                  * (c - c[..., :1]))
        dL = dL + strain * float(param[f"{Domain} electrode thickness [m]"])
    return float(param["Number of electrodes connected in parallel to make a cell"]) * dL

def swelling_objective(sol, param):
    """
    Peak swelling objective: the largest |thickness change| from the initial state over
    the cell face and the whole duty cycle, and the signed change at that point
    (negative = contraction, positive = expansion).
    """
    dL = thickness_change_from_initial(sol, param)
    k = np.unravel_index(np.argmax(np.abs(dL)), dL.shape)
    return float(np.abs(dL[k])), float(dL[k])

# ==============================================================================
# 4. THE CTP AUTOMOTIVE SIMULATION LOOP
# ==============================================================================
# Heat-transfer coefficients [W.m-2.K-1] of the two cooling architectures in the DoE.
COOLING = {
    "Surface": {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 5.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0,
    },
    "Tab": {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 1.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 1.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 750.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 750.0,
    },
}

CYCLES = ["US06", "WLTP", "WLTP_FastCharge"]

def run_latin_hypercube_doe(num_samples=200, temp_ambient=298.15, topologies=None, target_run_id=None,
                            cooling="Surface"):
    if topologies is None:
        topologies = ["CT", "NT"]

    print(f"\n--- Initializing Vehicle-to-Cell DOE ({num_samples} samples) ---")
    print(f"--- Targeted Topologies: {topologies} | Cooling: {cooling} ---")

    lhs_points = get_bounded_lhs_samples(num_samples)

    options = {"cell geometry": "pouch", "thermal": "x-lumped", "current collector": "potential pair", "dimensionality": 2}

    # One case (e.g. one task of a cluster array job) or every geometry in turn
    if target_run_id is not None:
        run_id = int(target_run_id)
        if run_id >= num_samples:
            raise ValueError(f"Run ID {run_id} is out of bounds for {num_samples} samples.")
        points_to_run = [(run_id, lhs_points[run_id])]
    else:
        points_to_run = list(enumerate(lhs_points))

    shard_dir = os.path.join(SHARD_DIR, cooling)
    os.makedirs(shard_dir, exist_ok=True)

    for topo_label in topologies:
        for run_id, point in points_to_run:
            thick_factor, L_ccp, L_ccn, cell_w, cell_h, pos_p, pos_n, w_tab = point
            print(f"  Run {run_id+1}/{num_samples} | Thick: {thick_factor:.2f}x, W: {cell_w:.2f}m, H: {cell_h:.2f}m")

            output_csv = os.path.join(shard_dir, f"shard_{topo_label}_run_{run_id}.csv")

            # Every output column is pre-allocated, so a failed case still writes a complete row.
            current_row = {
                "Topology": topo_label, "Thickness_Factor": thick_factor,
                "L_ccp_[m]": L_ccp, "L_ccn_[m]": L_ccn,
                "Cell_Width_[m]": cell_w, "Cell_Height_[m]": cell_h,
                "Pos_Tab_Frac": pos_p, "Neg_Tab_Frac": pos_n, "Tab_Width_Frac": w_tab,
                "OF1_Specific_Energy_[Wh/kg]": np.nan,
                **{f"OF2_{cycle}_Thermal_Gradient_[K]": np.nan for cycle in CYCLES},
                **{f"OF3_{cycle}_Peak_Swelling_[m]": np.nan for cycle in CYCLES},
                **{f"Diag_{cycle}_Signed_Swelling_[m]": np.nan for cycle in CYCLES},
                "Diagnostic_Mass_[kg]": np.nan,
                "Solve_Time_Total_[s]": 0.0,
                "Error": "Incomplete"
            }

            try:
                model = pybamm.lithium_ion.DFN(options=options)
                coords, _, _ = build_dynamic_topology(cell_w, cell_h, pos_p, pos_n, w_tab, topo_label)

                base_p = pybamm.ParameterValues("Ai2020")

                # Extract baseline dimensions BEFORE any parameter updating occurs
                W_base = base_p["Electrode width [m]"]
                H_base = base_p["Electrode height [m]"]
                base_capacity_Ah = base_p["Nominal cell capacity [A.h]"]

                L_n_new = base_p["Negative electrode thickness [m]"] * thick_factor
                L_p_new = base_p["Positive electrode thickness [m]"] * thick_factor

                param = get_base_parameters(L_n=L_n_new, L_p=L_p_new)

                defaults = {
                    "Ambient temperature [K]": temp_ambient,
                    "Positive current collector thickness [m]": L_ccp,
                    "Negative current collector thickness [m]": L_ccn,
                    **COOLING[cooling],
                    "Edge heat transfer coefficient [W.m-2.K-1]":0.0,
                    "Total heat transfer coefficient [W.m-2.K-1]": 0.0,
                }
                defaults.update(coords)
                param.update(defaults, check_already_exists=False)

                # Production mesh (Appendix A, Table A.1)
                var_pts = {
                    "x_n": 5, "x_s": 5, "x_p": 5,
                    "r_n": 5, "r_p": 5,
                    "y": 10, "z": 4
                }

                # --- ISOLATED 1D CAPACITY CHECK FOR OF1 ---
                print("    -> Running 1D 2C Discharge for Specific Energy...", flush=True)

                area_scaling = (cell_w * cell_h) / (W_base * H_base)
                dynamic_capacity = base_capacity_Ah * thick_factor * area_scaling

                model_1d = pybamm.lithium_ion.DFN()
                param_1d = param.copy()
                param_1d.update({"Current function [A]": dynamic_capacity * 2.0}, check_already_exists=False)

                sim_1d = pybamm.Simulation(model_1d, parameter_values=param_1d)
                sol_1d = sim_1d.solve(t_eval=np.linspace(0, 3600, 500))

                time_h_1d = sol_1d["Time [h]"].entries
                power_w_1d = np.abs(sol_1d["Terminal voltage [V]"].entries * sol_1d["Current [A]"].entries)

                discharge_energy_Wh = np.trapezoid(power_w_1d, x=time_h_1d)
                mass_kg = calculate_cell_mass(param)
                specific_energy = discharge_energy_Wh / mass_kg

                current_row["OF1_Specific_Energy_[Wh/kg]"] = specific_energy
                current_row["Diagnostic_Mass_[kg]"] = mass_kg

                # ==========================================================
                # --- NESTED 2D EXECUTION FOR ALL 3 CYCLES ---
                # ==========================================================
                for cycle in CYCLES:
                    print(f"    -> Running 2D Transient: {cycle}...", flush=True)
                    n_parallel = param["Number of electrodes connected in parallel to make a cell"]

                    time_array, I_cell_array = generate_drive_cycle(cycle_type=cycle, dynamic_capacity=dynamic_capacity, max_c_rate=5.0, n_parallel=n_parallel)

                    solver_2d = pybamm.IDAKLUSolver(rtol=1e-4, atol=1e-4)

                    if cycle == "WLTP_FastCharge":
                        print("    -> Initializing PyBaMM Experiment (WLTP -> Rest -> 2C Charge)...", flush=True)
                        wltp_drive_cycle = np.column_stack([time_array, I_cell_array])
                        charge_current_layer = (2.0 * dynamic_capacity) / n_parallel

                        experiment = pybamm.Experiment(
                            [
                                pybamm.step.current(wltp_drive_cycle),
                                "Rest for 300 seconds",
                                f"Charge at {charge_current_layer:.4f} A until 4.2 V"
                            ],
                            period="1 seconds"
                        )

                        sim_2d = pybamm.Simulation(model, experiment=experiment, parameter_values=param.copy(), var_pts=var_pts, solver=solver_2d)
                        sol_2d = sim_2d.solve(initial_soc=0.60)

                    else:
                        print(f"    -> Attaching Load Interpolant for {cycle}...", flush=True)
                        interpolant = pybamm.Interpolant(time_array, I_cell_array, pybamm.t, interpolator="cubic")

                        param_std = param.copy()
                        param_std.update({"Current function [A]": interpolant}, check_already_exists=False)

                        sim_2d = pybamm.Simulation(model, parameter_values=param_std, var_pts=var_pts, solver=solver_2d)
                        sol_2d = sim_2d.solve(t_eval=time_array, initial_soc=0.60)

                    # Whitelist the valid termination strings for both standard solve() and experiment solve()
                    valid_terminations = ['final time', 'success']

                    # If it's an experiment, hitting the 4.2V cutoff is a successful completion
                    is_expected_cutoff = "Voltage > 4.2" in str(sol_2d.termination)

                    if sol_2d.termination not in valid_terminations and not is_expected_cutoff:
                        raise pybamm.SolverError(f"Physical Limit Reached: {sol_2d.termination}")

                    T_3d_array = sol_2d["X-averaged cell temperature [K]"].entries
                    T_max = np.max(T_3d_array, axis=(0, 1))
                    T_min = np.min(T_3d_array, axis=(0, 1))
                    thermal_grad = np.max(T_max - T_min)

                    # Swelling relative to this cycle's own initial state (initial_soc above),
                    # not the parameter-set initial concentrations.
                    peak_swelling, signed_swelling = swelling_objective(sol_2d, param)

                    current_row[f"OF2_{cycle}_Thermal_Gradient_[K]"] = thermal_grad
                    current_row[f"OF3_{cycle}_Peak_Swelling_[m]"] = peak_swelling
                    current_row[f"Diag_{cycle}_Signed_Swelling_[m]"] = signed_swelling
                    current_row["Solve_Time_Total_[s]"] += getattr(sol_2d.solve_time, 'value', 0.0)
                    current_row["Error"] = f"Passed {cycle}"

                current_row["Error"] = "Success"
                pd.DataFrame([current_row]).to_csv(output_csv, index=False, header=True)

            except Exception as e:
                # Keep the objectives already computed for this case and record the failure.
                print(f"  [!] Run Failed: {e}", flush=True)
                current_row["Error"] = f"Failed: {str(e)}"
                pd.DataFrame([current_row]).to_csv(output_csv, index=False, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the design of experiments; one shard is written per case.")
    parser.add_argument("--cooling", choices=sorted(COOLING), required=True)
    parser.add_argument("--topology", choices=["CT", "NT"], nargs="+", default=["CT", "NT"])
    parser.add_argument("--run-id", type=int, default=None,
                        help="index (0-199) of a single LHS geometry; omit to run all 200")
    args = parser.parse_args()
    run_latin_hypercube_doe(num_samples=200, topologies=args.topology, target_run_id=args.run_id,
                            cooling=args.cooling)
