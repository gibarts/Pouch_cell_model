"""
Pooled GP surrogate over all 4 topology/cooling datasets (Section 2.2).

Inputs (10 dims):
    8 continuous design variables
    Topology  (CT / NT)          -> categorical dim
    Cooling   (Tab / Surface)    -> categorical dim

Targets (7, modeled as independent MixedSingleTaskGPs sharing the same
input featurization / cat_dims):
    OF1_Specific_Energy_[Wh/kg]
    OF2_US06_Thermal_Gradient_[K]
    OF2_WLTP_Thermal_Gradient_[K]
    OF2_WLTP_FastCharge_Thermal_Gradient_[K]
    OF3_US06_Peak_Swelling_[m]
    OF3_WLTP_Peak_Swelling_[m]
    OF3_WLTP_FastCharge_Peak_Swelling_[m]

Validation: 5-fold CV grouped by Run_ID (i.e. all 4 topology/cooling rows
for a given geometry stay in the same fold), so held-out folds are
genuinely unseen geometries, not just an unseen topology/cooling label
for an already-seen geometry.

Noise: fixed at a near-zero nugget (1e-4 in standardized-output units).
The underlying simulator has no known stochastic sub-model -- there is no
replicate/aleatoric noise to fit, so all uncertainty here is epistemic.

Model space: all targets are modeled in raw space, except that the four
drive-cycle targets are first divided by the known pack-sizing scale (see
PACK_SCALED_TARGETS and Appendix C) and multiplied back after prediction.

Outputs (under outputs/):
    cv_results.log            human-readable per-target CV summary
    cv_metrics.jsonl          one record per (target, fold): metrics + fitted noise
    predictions.npz           per-test-point y_true/y_pred/y_std per target, used by
                              src.uq_calibration
    models/<target>_final.pth final models fitted on all 800 cases, used by
                              src.sensitivity
    models/scaling_bounds.npz input scaling of the final models

Usage (from the repository root):
    python -m src.model_training             # cross-validation, then final models
    python -m src.model_training --skip-cv   # final models only
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from src.optimisation import cell_capacity, pack_sizing_scale
from src.paths import DATA_DIR, MODEL_DIR, output_path

torch.set_default_dtype(torch.float64)

FILES = {
    ("CT", "Tab"): "Master_CT_Optimization_Dataset.csv",
    ("CT", "Surface"): "Master_CT_Optimization_Dataset_Surface.csv",
    ("NT", "Tab"): "Master_NT_Optimization_Dataset.csv",
    ("NT", "Surface"): "Master_NT_Optimization_Dataset_Surface.csv",
}

CONT_COLS = [
    "Thickness_Factor", "L_ccp_[m]", "L_ccn_[m]", "Cell_Width_[m]",
    "Cell_Height_[m]", "Pos_Tab_Frac", "Neg_Tab_Frac", "Tab_Width_Frac",
]

TARGETS = [
    "OF1_Specific_Energy_[Wh/kg]",
    "OF2_US06_Thermal_Gradient_[K]",
    "OF2_WLTP_Thermal_Gradient_[K]",
    "OF2_WLTP_FastCharge_Thermal_Gradient_[K]",
    "OF3_US06_Peak_Swelling_[m]",
    "OF3_WLTP_Peak_Swelling_[m]",
    "OF3_WLTP_FastCharge_Peak_Swelling_[m]",
]

FIXED_NOISE_LEVEL = 1e-4  # in standardized-output units (Standardize -> unit variance)

# The drive-cycle targets are driven by a cell current set through pack sizing, so they
# are modelled as y = s(x)^k * g(x): s is the RMS cell C-rate of the pack-sized cycle
# (optimisation.pack_sizing_scale), k = 2 for the thermal gradient (ohmic heat) and
# k = 1 for swelling (charge throughput). The GP models the smooth part g, and its mean
# and std are multiplied back by s^k, so the pack-sizing discontinuities are exact.
PACK_SCALED_TARGETS = {
    "OF2_US06_Thermal_Gradient_[K]": ("US06", 2),
    "OF2_WLTP_Thermal_Gradient_[K]": ("WLTP", 2),
    "OF3_US06_Peak_Swelling_[m]": ("US06", 1),
    "OF3_WLTP_Peak_Swelling_[m]": ("WLTP", 1),
}

logger = logging.getLogger("gp_surrogate")
logger.setLevel(logging.INFO)


def setup_file_logging():
    """Attach the log-file handler. Only called from __main__ -- importing
    this module (e.g. from src.uq_calibration) must not truncate
    cv_results.log as a side effect."""
    logger.handlers.clear()
    file_handler = logging.FileHandler(output_path("cv_results.log"), mode="w")
    stream_handler = logging.StreamHandler(sys.stdout)
    for h in (file_handler, stream_handler):
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)


def load_pooled():
    frames = []
    for (topo, cool), fname in FILES.items():
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        df["Cooling"] = cool
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["Topology_idx"] = (full["Topology"] == "NT").astype(int)
    full["Cooling_idx"] = (full["Cooling"] == "Surface").astype(int)
    return full


def target_scale(df, target):
    """s(x)^k for each row of df (design columns in physical units); ones for targets not pack-scaled."""
    if target not in PACK_SCALED_TARGETS:
        return np.ones(len(df))
    cycle, k = PACK_SCALED_TARGETS[target]
    Q = cell_capacity(df["Thickness_Factor"].values, df["Cell_Width_[m]"].values, df["Cell_Height_[m]"].values)
    return pack_sizing_scale(cycle, Q) ** k


def build_X(df, cont_min, cont_max):
    cont = (df[CONT_COLS].values - cont_min) / (cont_max - cont_min)
    cat = df[["Topology_idx", "Cooling_idx"]].values
    return np.concatenate([cont, cat], axis=1)


def make_fixed_noise_likelihood():
    lik = GaussianLikelihood()
    lik.noise = FIXED_NOISE_LEVEL
    lik.raw_noise.requires_grad_(False)
    return lik


def run_cv(df, n_splits=5):
    groups = df["Run_ID"].values
    gkf = GroupKFold(n_splits=n_splits)
    cat_dims = [len(CONT_COLS), len(CONT_COLS) + 1]

    jsonl_records = []
    predictions = {}

    for target in TARGETS:
        y_true_all, y_pred_all, y_std_all = [], [], []
        scale = target_scale(df, target)

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups)):
            train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]

            cont_min = train_df[CONT_COLS].values.min(axis=0)
            cont_max = train_df[CONT_COLS].values.max(axis=0)

            X_train = torch.tensor(build_X(train_df, cont_min, cont_max))
            X_test = torch.tensor(build_X(test_df, cont_min, cont_max))
            y_train = torch.tensor(train_df[[target]].values / scale[tr_idx, None])

            gp = MixedSingleTaskGP(X_train, y_train, cat_dims=cat_dims,
                                    likelihood=make_fixed_noise_likelihood())
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            gp.eval()
            with torch.no_grad():
                post = gp.posterior(X_test)
                mean = post.mean.squeeze(-1).numpy() * scale[te_idx]
                std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy() * scale[te_idx]

            y_true_fold = test_df[target].values
            r2_fold = r2_score(y_true_fold, mean) if len(y_true_fold) > 1 else float("nan")
            rmse_fold = np.sqrt(mean_squared_error(y_true_fold, mean))
            z_fold = (y_true_fold - mean) / std
            coverage_fold = float(np.mean(np.abs(z_fold) <= 1.96))

            jsonl_records.append(dict(
                target=target, fold=fold,
                n_train=len(tr_idx), n_test=len(te_idx),
                r2=float(r2_fold), rmse=float(rmse_fold), coverage95=coverage_fold,
                fitted_noise=float(gp.likelihood.noise.item()),
            ))

            y_true_all.append(y_true_fold)
            y_pred_all.append(mean)
            y_std_all.append(std)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_std_all = np.concatenate(y_std_all)
        predictions[target] = dict(y_true=y_true_all, y_pred=y_pred_all, y_std=y_std_all)

        r2 = r2_score(y_true_all, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        rel_rmse = rmse / (y_true_all.max() - y_true_all.min())
        z = (y_true_all - y_pred_all) / y_std_all
        coverage95 = np.mean(np.abs(z) <= 1.96)

        logger.info(
            f"{target:45s}  R2={r2:6.3f}  RMSE={rmse:.3e}  RelRMSE={rel_rmse:6.3%}  "
            f"95%CI-coverage={coverage95:.2%}"
        )

    metrics_path = output_path("cv_metrics.jsonl")
    with open(metrics_path, "w") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"\nPer-fold metrics written to {metrics_path}")

    flat = {}
    for target, arrs in predictions.items():
        safe = target.split("_[")[0]
        for k, v in arrs.items():
            flat[f"{safe}__{k}"] = v
    predictions_path = output_path("predictions.npz")
    np.savez(predictions_path, **flat)
    logger.info(f"Raw per-point predictions written to {predictions_path}")

def fit_and_save_final_models(df):
    """Trains final models on 100% of the data for downstream sensitivity analysis."""
    logger.info("\n=== Fitting and serializing final models on full dataset ===")
    os.makedirs(MODEL_DIR, exist_ok=True)

    cont_min = df[CONT_COLS].values.min(axis=0)
    cont_max = df[CONT_COLS].values.max(axis=0)
    X_full = torch.tensor(build_X(df, cont_min, cont_max))
    cat_dims = [len(CONT_COLS), len(CONT_COLS) + 1]

    # Save the continuous scaling boundaries for downstream reconstruction
    np.savez(os.path.join(MODEL_DIR, "scaling_bounds.npz"), cont_min=cont_min, cont_max=cont_max)

    for target in TARGETS:
        logger.info(f"  Serializing {target}...")
        y_full = torch.tensor(df[[target]].values / target_scale(df, target)[:, None])

        gp = MixedSingleTaskGP(X_full, y_full, cat_dims=cat_dims, likelihood=make_fixed_noise_likelihood())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        safe_target = target.split('_[')[0]
        torch.save(gp.state_dict(), os.path.join(MODEL_DIR, f"{safe_target}_final.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grouped cross-validation and final surrogate fits.")
    parser.add_argument("--skip-cv", action="store_true", help="fit the final models only")
    args = parser.parse_args()

    setup_file_logging()
    df = load_pooled()
    logger.info(f"Pooled dataset: {df.shape[0]} rows, {df['Run_ID'].nunique()} unique geometries\n")
    if not args.skip_cv:
        logger.info("=== 5-fold grouped CV (grouped by Run_ID), fixed near-zero noise ===")
        run_cv(df, n_splits=5)
    fit_and_save_final_models(df)
