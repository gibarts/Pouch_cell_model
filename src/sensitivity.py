"""
Global sensitivity analysis on the final surrogates (Sections 2.4, 3.2 and 3.3):
Sobol first-order and total-effect indices with bootstrap confidence intervals for the
eight continuous design variables, and the two-way ANOVA of topology and cooling.

Requires the final models from src.model_training.

Outputs (under outputs/):
    sensitivity_indices.json / .csv   S1, ST (and S2 in the JSON) with 95% CIs
    categorical_sensitivity.csv       one-way and two-way eta-squared, raw and log scale,
                                      and condition means

Usage (from the repository root): python -m src.sensitivity
"""
import os
import json
import torch
import pandas as pd
import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

from botorch.models import MixedSingleTaskGP

from src.model_training import (CONT_COLS, TARGETS, build_X, load_pooled, target_scale,
                                make_fixed_noise_likelihood)
from src.paths import MODEL_DIR, output_path

SENSITIVITY_RESULTS_JSON = output_path("sensitivity_indices.json")
SENSITIVITY_RESULTS_CSV = output_path("sensitivity_indices.csv")
CATEGORICAL_RESULTS_CSV = output_path("categorical_sensitivity.csv")


def define_design_space():
    """Defines the 8 continuous variables for the Sobol sampling."""
    return {
        'num_vars': 8,
        'names': CONT_COLS,
        'bounds': [
            [0.5, 2.0],         # Micro: Electrode thickness scale
            [10e-6, 30e-6],     # Al current collector thickness
            [5e-6, 15e-6],      # Cu current collector thickness
            [0.10, 0.60],       # Macro: Cell Width (100mm to 600mm)
            [0.05, 0.30],       # Macro: Cell Height (50mm to 300mm)
            [0.15, 0.85],       # Tab placement parameters
            [0.15, 0.85],
            [0.1, 0.99]         # Tab width fraction
        ]
    }

def calculate_eta_squared(df_all, target):
    """Calculates the ANOVA Eta-squared for the categorical Condition variable."""
    overall_mean = df_all[target].mean()
    ss_total = ((df_all[target] - overall_mean) ** 2).sum()

    ss_between = 0
    for cond in df_all['Condition'].unique():
        cond_group = df_all[df_all['Condition'] == cond][target]
        ss_between += len(cond_group) * ((cond_group.mean() - overall_mean) ** 2)

    return ss_between / ss_total if ss_total != 0 else 0


def calculate_two_way_eta_squared(df_all, target, log_scale=False):
    """
    Two-way ANOVA decomposition of the categorical effect into Topology and
    Cooling main effects plus their interaction.

    The Condition labels are of the form '<Topology>_<Cooling>' (e.g. 'NT_Tab'),
    and the design is balanced and fully crossed -- the same evaluation sample is
    used under all four conditions -- so the sums of squares are orthogonal and
    SS_topology + SS_cooling + SS_interaction reproduces the one-way SS_between
    exactly. That identity is asserted below.

    Set log_scale=True to repeat the decomposition on log(y); this checks that
    additivity is not an artefact of the raw (additive) parameterisation. Only
    valid for strictly positive targets.

    Returns a dict of eta-squared values, each expressed as a fraction of the
    total sum of squares.
    """
    d = df_all[["Condition", target]].copy()
    if log_scale:
        # The GP predicts in raw space, so a strictly-positive target can still
        # receive occasional non-positive predictions. Dropping them per-row would
        # leave unequal counts per condition and destroy the balance the
        # decomposition relies on, so drop each offending sample from ALL
        # conditions: the evaluation sample is shared, and rows appear in the same
        # order within every condition block.
        d["_sample"] = d.groupby("Condition").cumcount()
        bad = d.loc[d[target] <= 0, "_sample"].unique()
        d = d[~d["_sample"].isin(bad)].drop(columns="_sample")
        if d.empty:
            raise ValueError(f"No strictly positive samples remain for {target}")
        d[target] = np.log(d[target])

    d["Topology"] = d["Condition"].str.split("_").str[0]
    d["Cooling"] = d["Condition"].str.split("_").str[1]

    y = d[target].values
    grand_mean = y.mean()
    ss_total = ((y - grand_mean) ** 2).sum()
    if ss_total == 0:
        return {k: 0.0 for k in
                ("eta_sq_condition", "eta_sq_topology", "eta_sq_cooling", "eta_sq_interaction")}

    cells = d.groupby(["Topology", "Cooling"])[target].agg(["mean", "count"])
    topo = d.groupby("Topology")[target].agg(["mean", "count"])
    cool = d.groupby("Cooling")[target].agg(["mean", "count"])

    ss_topology = (topo["count"] * (topo["mean"] - grand_mean) ** 2).sum()
    ss_cooling = (cool["count"] * (cool["mean"] - grand_mean) ** 2).sum()
    ss_interaction = sum(
        row["count"] * (row["mean"] - topo.loc[t, "mean"] - cool.loc[c, "mean"] + grand_mean) ** 2
        for (t, c), row in cells.iterrows()
    )
    ss_between = (cells["count"] * (cells["mean"] - grand_mean) ** 2).sum()

    # Orthogonality check: exact for a balanced, fully crossed design.
    if not np.isclose(ss_topology + ss_cooling + ss_interaction, ss_between, rtol=1e-8):
        raise ValueError(
            f"Two-way decomposition failed for {target}: design is not balanced "
            f"(SS_topo+SS_cool+SS_int={ss_topology + ss_cooling + ss_interaction:.6e} "
            f"vs SS_between={ss_between:.6e})"
        )

    return {
        "eta_sq_condition": ss_between / ss_total,
        "eta_sq_topology": ss_topology / ss_total,
        "eta_sq_cooling": ss_cooling / ss_total,
        "eta_sq_interaction": ss_interaction / ss_total,
    }


def save_sobol_indices(master_results, param_names):
    """Save Sobol first-order, total-effect, and second-order indices."""
    serializable = {}
    rows = []

    for target, by_condition in master_results.items():
        serializable[target] = {}
        for condition, indices in by_condition.items():
            s1 = np.asarray(indices["S1"], dtype=float)
            st = np.asarray(indices["ST"], dtype=float)
            s2 = np.asarray(indices.get("S2", []), dtype=float)
            s1_conf = np.asarray(indices["S1_conf"], dtype=float)
            st_conf = np.asarray(indices["ST_conf"], dtype=float)

            serializable[target][condition] = {
                "parameters": list(param_names),
                "S1": s1.tolist(),
                "ST": st.tolist(),
                "S2": s2.tolist(),
                "S1_conf": s1_conf.tolist(),
                "ST_conf": st_conf.tolist(),
            }

            for param, s1_value, st_value, s1c, stc in zip(param_names, s1, st, s1_conf, st_conf):
                rows.append({
                    "target": target,
                    "condition": condition,
                    "parameter": param,
                    "S1": s1_value,
                    "S1_conf": s1c,
                    "ST": st_value,
                    "ST_conf": stc,
                })

    with open(SENSITIVITY_RESULTS_JSON, "w") as f:
        json.dump(serializable, f, indent=2)

    pd.DataFrame(rows).to_csv(SENSITIVITY_RESULTS_CSV, index=False)


def save_categorical_sensitivity(df_global, targets):
    rows = []
    for target in targets:
        eta_sq = calculate_eta_squared(df_global, target)
        two_way = calculate_two_way_eta_squared(df_global, target)
        try:
            two_way_log = calculate_two_way_eta_squared(df_global, target, log_scale=True)
        except (ValueError, FloatingPointError):
            two_way_log = {k: np.nan for k in two_way}

        means = df_global.groupby("Condition")[target].mean()
        for condition, mean_value in means.items():
            rows.append({
                "target": target,
                "condition": condition,
                "eta_squared": eta_sq,
                "eta_sq_topology": two_way["eta_sq_topology"],
                "eta_sq_cooling": two_way["eta_sq_cooling"],
                "eta_sq_interaction": two_way["eta_sq_interaction"],
                "eta_sq_topology_log": two_way_log["eta_sq_topology"],
                "eta_sq_cooling_log": two_way_log["eta_sq_cooling"],
                "eta_sq_interaction_log": two_way_log["eta_sq_interaction"],
                "condition_mean": mean_value,
            })

    pd.DataFrame(rows).to_csv(CATEGORICAL_RESULTS_CSV, index=False)


def target_column_from_model_name(model_name):
    for target in TARGETS:
        if target.split('_[')[0] == model_name:
            return target
    raise KeyError(f"Could not map model name {model_name} to a training target column.")

if __name__ == "__main__":
    problem = define_design_space()

    # Physical states mapping to the BoTorch categorical indices (Topology_idx, Cooling_idx)
    # Topology: 0 = CT, 1 = NT
    # Cooling:  0 = Tab, 1 = Surface
    physical_conditions = {
        "CT_Tab": (0, 0),
        "CT_Surface": (0, 1),
        "NT_Tab": (1, 0),
        "NT_Surface": (1, 1)
    }

    targets_to_analyze = [
        'OF1_Specific_Energy',
        'OF2_WLTP_FastCharge_Thermal_Gradient',
        'OF2_WLTP_Thermal_Gradient',
        'OF2_US06_Thermal_Gradient',
        'OF3_WLTP_FastCharge_Peak_Swelling',
        'OF3_WLTP_Peak_Swelling',
        'OF3_US06_Peak_Swelling'
    ]

    print(f"\n\n{'*'*60}\nEXECUTING POOLED BOTORCH TENSOR SENSITIVITY\n{'*'*60}")

    # 1. Load the boundary scaling metrics saved during training
    bounds_file = os.path.join(MODEL_DIR, "scaling_bounds.npz")
    if not os.path.exists(bounds_file):
        raise FileNotFoundError(f"{bounds_file} not found. Run python -m src.model_training first.")

    bounds_data = np.load(bounds_file)
    cont_min, cont_max = bounds_data['cont_min'], bounds_data['cont_max']
    df_train = load_pooled()
    X_train = torch.tensor(build_X(df_train, cont_min, cont_max), dtype=torch.float64)

    # 2. Generate the base continuous Saltelli matrix (N=2048)
    print("  Generating continuous Saltelli base matrix (N=2048)...")
    X_saltelli_raw = sobol_sample.sample(problem, 2048, calc_second_order=True, seed=42)
    X_saltelli_scaled = (X_saltelli_raw - cont_min) / (cont_max - cont_min)

    master_sobol_results = {t: {} for t in targets_to_analyze}
    all_predictions_list = []

    cat_dims = [8, 9]

    for target in targets_to_analyze:
        print(f"\n--- Instantiating Computational Graph for: {target} ---")

        # 3. Reconstruct the exact GP with the full training inputs/targets.
        # Exact GP posterior predictions condition on these points; loading
        # hyperparameters into a dummy GP is not sufficient.
        target_column = target_column_from_model_name(target)
        # Same target scaling as training (pack-sizing scale for drive-cycle targets);
        # predictions are multiplied back below, evaluated at each Saltelli design.
        y_train = torch.tensor(df_train[[target_column]].values / target_scale(df_train, target_column)[:, None],
                               dtype=torch.float64)
        saltelli_scale = target_scale(pd.DataFrame(X_saltelli_raw, columns=problem['names']), target_column)
        gp = MixedSingleTaskGP(X_train, y_train, cat_dims=cat_dims, likelihood=make_fixed_noise_likelihood())

        # 4. Load the master state dictionary
        model_path = os.path.join(MODEL_DIR, f"{target}_final.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] Missing serialized model for {target}")

        gp.load_state_dict(torch.load(model_path))
        gp.eval()

        target_predictions = []

        # 5. Iteratively evaluate the posterior for each categorical condition
        for cond_label, (topo_idx, cool_idx) in physical_conditions.items():
            print(f"  -> Extracting sensitivity for {cond_label}...")

            cat_topo = np.full((X_saltelli_scaled.shape[0], 1), topo_idx)
            cat_cool = np.full((X_saltelli_scaled.shape[0], 1), cool_idx)

            # Construct the 10D tensor
            X_10D = np.hstack([X_saltelli_scaled, cat_topo, cat_cool])
            X_tensor = torch.tensor(X_10D, dtype=torch.float64)

            # --- MEMORY SAFE BATCHED INFERENCE ---
            batch_size = 2048
            y_pred_list = []

            with torch.no_grad():
                for i in range(0, len(X_tensor), batch_size):
                    X_batch = X_tensor[i : i + batch_size]
                    posterior = gp.posterior(X_batch)
                    y_pred_list.append(posterior.mean.squeeze(-1).numpy())

            # Concatenate the chunks back into a single 1D array
            y_pred_real = np.concatenate(y_pred_list) * saltelli_scale
            # -------------------------------------

            # Execute variance decomposition
            # Bootstrap confidence intervals (95%) are seeded so they reproduce
            # exactly; the point estimates do not depend on this seed.
            Si = sobol.analyze(problem, y_pred_real, calc_second_order=True,
                               print_to_console=False, seed=42)
            master_sobol_results[target][cond_label] = {
                'S1': Si['S1'],
                'ST': Si['ST'],
                'S2': Si['S2'],
                'S1_conf': Si['S1_conf'],
                'ST_conf': Si['ST_conf'],
            }

            # Predictions under this condition, for the two-way ANOVA of topology and cooling
            df_cond = pd.DataFrame(X_saltelli_raw, columns=problem['names'])
            df_cond['Condition'] = cond_label
            df_cond[target] = y_pred_real
            target_predictions.append(df_cond)

        # Combine across conditions for this target
        if not all_predictions_list:
            all_predictions_list = target_predictions
        else:
            for idx, df in enumerate(target_predictions):
                all_predictions_list[idx][target] = df[target]

    if all_predictions_list:
        df_global = pd.concat(all_predictions_list, ignore_index=True)

        save_sobol_indices(master_sobol_results, problem['names'])
        save_categorical_sensitivity(df_global, targets_to_analyze)
        print(f"\nSobol indices written to {SENSITIVITY_RESULTS_JSON}")
        print(f"Sobol S1/ST table written to {SENSITIVITY_RESULTS_CSV}")
        print(f"Categorical sensitivity table written to {CATEGORICAL_RESULTS_CSV}")
