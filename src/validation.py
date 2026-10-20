"""
Baseline validation of the Ai2020 parameterisation against the experimental 1C discharge
of Ai et al. (Appendix B): terminal voltage, temperature rise and cell thickness change.

Outputs (under outputs/):
    validation_metrics.json           Table B.1

The experimental data are downloaded by PyBaMM's DataLoader on first use.

Usage (from the repository root): python -m src.validation
"""
import json

import pybamm
import pandas as pd
import numpy as np
from .cell_model import run_dfn_swelling_simulation
from .optimisation import thickness_change_from_initial
from .paths import output_path


def load_experimental_data():
    """
    Loads experimental validation data from PyBaMM.
    Returns a dictionary organized by measurement type and C-rate.
    """
    data_loader = pybamm.DataLoader()

    val_data = {
        "Voltage": {},
        "Temperature": {},
        "Displacement": {}
    }

    # Load Voltages and Displacements
    for c in [0.1, 0.5, 1, 2]:
        v_file = f"{c}C_discharge_U.txt"
        d_file = f"{c}C_discharge_displacement.txt"

        # Using delimiter=r"\s+" to avoid pandas warnings for regex
        val_data["Voltage"][c] = pd.read_csv(data_loader.get_data(v_file), delimiter=r"\s+", header=None)
        val_data["Displacement"][c] = pd.read_csv(data_loader.get_data(d_file), delimiter=r"\s+", header=None)

    # Load Temperatures (0.1C was omitted in the original script)
    for c in [0.5, 1, 2]:
        t_file = f"{c}C_discharge_T.txt"
        val_data["Temperature"][c] = pd.read_csv(data_loader.get_data(t_file), delimiter=r"\s+", header=None)

    return val_data

def compute_signal_rmse(sim_time, sim_signal, exp_time, exp_signal, span=None):
    """
    Interpolate experimental signal onto simulation time grid and compute RMSE/MAE.
    Returns absolute RMSE/MAE and each as a percentage of `span`. If `span` is not
    given, it defaults to the span of the interpolated experimental data actually
    used in this call (i.e. of whatever time window sim_time/exp_time cover).
    """
    exp_signal_interp = np.interp(sim_time, exp_time, exp_signal, left=np.nan, right=np.nan)
    valid_mask = ~np.isnan(exp_signal_interp)
    sim_valid = sim_signal[valid_mask]
    exp_valid = exp_signal_interp[valid_mask]

    if span is None:
        span = np.max(exp_valid) - np.min(exp_valid)

    rmse = np.sqrt(np.mean((sim_valid - exp_valid) ** 2))
    rmse_pct = 100.0 * rmse / span if span > 0 else np.inf

    mae = np.mean(np.abs(sim_valid - exp_valid))
    mae_pct = 100.0 * mae / span if span > 0 else np.inf

    return rmse, rmse_pct, mae, mae_pct, span

def validate_baseline_1C(sol, param):
    """
    Validates 1C discharge simulation against Ai2020 experimental voltage,
    temperature rise, and cell thickness change.
    Computes RMSE/MAE for each signal and returns a metrics dictionary.
    """
    print("\n" + "="*70)
    print("BASELINE VALIDATION: 1C Discharge Accuracy")
    print("="*70)

    initial_temp = pybamm.ParameterValues("Ai2020")["Initial temperature [K]"]

    sim_time_h = sol["Time [h]"].entries
    sim_voltage_v = sol["Terminal voltage [V]"].entries
    sim_temp_rise_k = sol["Volume-averaged cell temperature [K]"].entries - initial_temp
    sim_thickness_m = sol["Cell thickness change [m]"].entries.mean(axis=0)

    val_data = load_experimental_data()

    # Unified comparison window applied to all three signals: excludes the near-vertical
    # end-of-discharge knee (voltage/thickness) and the post-discharge rest/cooldown
    # transition (temperature), so sim and experimental curves are compared over the
    # same, physically comparable stretch of the discharge.
    WINDOW_H = 0.9
    v_window_mask = sim_time_h <= WINDOW_H
    sim_time_w = sim_time_h[v_window_mask]

    # --- Voltage ---
    exp_v_df = val_data["Voltage"][1]
    exp_time_v_h = exp_v_df.iloc[:, 0].values / 3600.0  # experimental time columns are in seconds
    exp_voltage_v = exp_v_df.iloc[:, 1].values
    span_v = exp_voltage_v.max() - exp_voltage_v.min()  # full discharge-range span, not the windowed subset

    rmse_v, rmse_v_pct, mae_v, mae_v_pct, _ = compute_signal_rmse(
        sim_time_w, sim_voltage_v[v_window_mask], exp_time_v_h, exp_voltage_v, span=span_v
    )
    print(f"\nVoltage (0-{WINDOW_H:.1f}h window): RMSE = {rmse_v:.5f} V  ({rmse_v_pct:.2f}% of full-range span)   MAE = {mae_v_pct:.2f}%")

    # --- Temperature rise ---
    exp_t_df = val_data["Temperature"][1]
    exp_time_t_h_full = exp_t_df.iloc[:, 0].values / 3600.0
    exp_temp_rise_k_full = exp_t_df.iloc[:, 1].values
    # Span from the discharge-only portion (excludes the post-discharge rest/cooldown,
    # which would otherwise inflate the span with near-zero rest-period values)
    disch_end_h = sim_time_h[-1]
    t_disch_mask = exp_time_t_h_full <= disch_end_h
    span_t = exp_temp_rise_k_full[t_disch_mask].max() - exp_temp_rise_k_full[t_disch_mask].min()

    rmse_t, rmse_t_pct, mae_t, mae_t_pct, _ = compute_signal_rmse(
        sim_time_w, sim_temp_rise_k[v_window_mask], exp_time_t_h_full, exp_temp_rise_k_full, span=span_t
    )
    print(f"Temperature (0-{WINDOW_H:.1f}h window):    RMSE = {rmse_t:.5f} K  ({rmse_t_pct:.2f}% of discharge span)   MAE = {mae_t_pct:.2f}%")

    # --- Thickness change ---
    exp_d_df = val_data["Displacement"][1]
    exp_time_d_h = exp_d_df.iloc[:, 0].values / 3600.0
    exp_thickness_m = exp_d_df.iloc[:, 1].values - exp_d_df.iloc[0, 1]  # zero the baseline offset
    span_d = exp_thickness_m.max() - exp_thickness_m.min()

    rmse_d, rmse_d_pct, mae_d, mae_d_pct, _ = compute_signal_rmse(
        sim_time_w, sim_thickness_m[v_window_mask], exp_time_d_h, exp_thickness_m, span=span_d
    )
    print(f"Thickness (0-{WINDOW_H:.1f}h window):    RMSE = {rmse_d*1e6:.3f} um ({rmse_d_pct:.2f}% of full-range span)   MAE = {mae_d_pct:.2f}%")

    # --- Thickness change from the peak-swelling objective (optimisation.py) ---
    # The same function as the DoE, evaluated on this run and compared over the same
    # window and span as the built-in thickness variable above.
    sim_formula_m = thickness_change_from_initial(sol, param).mean(axis=0)
    rmse_f, rmse_f_pct, mae_f, mae_f_pct, _ = compute_signal_rmse(
        sim_time_w, sim_formula_m[v_window_mask], exp_time_d_h, exp_thickness_m, span=span_d
    )
    gap_m = sim_formula_m[v_window_mask] - sim_thickness_m[v_window_mask]
    print(f"Thickness, objective formula (0-{WINDOW_H:.1f}h window):    RMSE = {rmse_f*1e6:.3f} um ({rmse_f_pct:.2f}% of full-range span)   MAE = {mae_f_pct:.2f}%")
    print(f"  Formula minus built-in thickness change: max |gap| = {np.max(np.abs(gap_m))*1e6:.2f} um "
          f"({100.0*np.max(np.abs(gap_m))/span_d:.2f}% of span); "
          f"peak |change| over window {np.max(np.abs(sim_formula_m[v_window_mask]))*1e6:.2f} um (formula) vs "
          f"{np.max(np.abs(sim_thickness_m[v_window_mask]))*1e6:.2f} um (built-in)")

    metrics = {
        "c_rate": 1.0,
        "window_h": WINDOW_H,
        "rmse_v": float(rmse_v), "rmse_v_pct": float(rmse_v_pct), "mae_v_pct": float(mae_v_pct),
        "rmse_t": float(rmse_t), "rmse_t_pct": float(rmse_t_pct), "mae_t_pct": float(mae_t_pct),
        "rmse_d": float(rmse_d), "rmse_d_pct": float(rmse_d_pct), "mae_d_pct": float(mae_d_pct),
        "rmse_d_formula": float(rmse_f), "rmse_d_formula_pct": float(rmse_f_pct), "mae_d_formula_pct": float(mae_f_pct),
    }

    print("\n" + "="*70)
    print(f"Measured accuracy (0-{WINDOW_H:.1f}h window) - "
          f"Voltage: {rmse_v_pct:.2f}%  Temperature: {rmse_t_pct:.2f}%  Thickness: {rmse_d_pct:.2f}%")
    print("="*70 + "\n")

    return metrics

if __name__ == "__main__":
    print("\nStarting 1C baseline validation...\n")
    sol, param = run_dfn_swelling_simulation(C_rate=1.0)
    metrics = validate_baseline_1C(sol, param)
    metrics_path = output_path("validation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_path}")
