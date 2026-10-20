"""
Uncertainty analysis and grouped conformal calibration of the pooled GP surrogate
(Sections 2.3 and 3.1; Table 3).

Runs on the CV predictions saved by src.model_training in outputs/predictions.npz -- no GPs are
refit here. GroupKFold(shuffle=False) is deterministic, so re-running the
same split against the same (unshuffled) pooled dataframe reproduces the
exact fold assignment / concatenation order used when predictions.npz was
written, letting us recover which Run_ID/Topology/Cooling each saved
prediction belongs to without touching the model.

Two questions are answered for every target:

1. WHERE does the GP's own uncertainty (posterior std) fail to match
   reality? Two candidate explanations are checked:
     - Mahalanobis distance in the 8D continuous design space (is
       miscalibration explained by geometrically extreme/sparse points?)
     - Topology x Cooling condition (is it explained by a specific
       physical regime the model handles worse?)

2. Given the answer (condition, not distance), what does a grouped
   (Mondrian) split-conformal correction look like, and does it actually
   fix calibration on held-out points?

Outputs (under outputs/):
    uq_report.json               Mahalanobis correlations, per-condition
                                  breakdown, NLPD, PIT KS-test, per target
    conformal_calibration.json   per-condition q_group multipliers +
                                  validated (held-out) coverage (Table 3)

Usage (from the repository root): python -m src.uq_calibration
"""

import os
import json

import numpy as np
import pandas as pd
from scipy.stats import kstest, spearmanr, norm
from sklearn.model_selection import GroupKFold

from src.model_training import load_pooled, CONT_COLS, TARGETS
from src.paths import OUTPUT_DIR, output_path

ALPHA = 0.05
SEED = 0


# --------------------------------------------------------------------------
# Data plumbing
# --------------------------------------------------------------------------

def recover_fold_identifiers(df, n_splits=5):
    groups = df["Run_ID"].values
    gkf = GroupKFold(n_splits=n_splits)
    ids = []
    for _, te_idx in gkf.split(df, groups=groups):
        ids.append(df.iloc[te_idx][["Run_ID", "Topology", "Cooling"]])
    return pd.concat(ids, ignore_index=True)


def mahalanobis_by_geometry(df):
    geo = df.drop_duplicates("Run_ID").set_index("Run_ID")[CONT_COLS]
    mu = geo.mean(axis=0).values
    sigma = geo.std(axis=0).values
    Z = (geo.values - mu) / sigma
    inv_cov = np.linalg.inv(np.cov(Z, rowvar=False))
    d2 = np.einsum("ij,jk,ik->i", Z, inv_cov, Z)
    return pd.Series(np.sqrt(d2), index=geo.index, name="mahalanobis")


def conformal_quantile(scores, alpha=ALPHA):
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def diagnose(ids, preds):
    report = {}
    for target in TARGETS:
        safe = target.split("_[")[0]
        y_true, y_pred, y_std = preds[f"{safe}__y_true"], preds[f"{safe}__y_pred"], preds[f"{safe}__y_std"]
        z = (y_true - y_pred) / y_std
        outside95 = (np.abs(z) > 1.96).astype(int)
        d = ids["mahalanobis"].values

        rho_absz, p_absz = spearmanr(d, np.abs(z))
        quart = pd.qcut(d, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        cov_by_quart = pd.Series(1 - outside95, index=range(len(d))).groupby(quart, observed=True).mean()

        grp = pd.DataFrame({"Topology": ids["Topology"], "Cooling": ids["Cooling"],
                             "z": z, "outside95": outside95, "sq_err": (y_true - y_pred) ** 2})
        by_cond = grp.groupby(["Topology", "Cooling"]).agg(
            n=("z", "size"), coverage95=("outside95", lambda x: 1 - x.mean()),
            rmse=("sq_err", lambda x: float(np.sqrt(x.mean()))),
        )
        by_cond.index = [f"{t}_{c}" for t, c in by_cond.index]

        nlpd = float(np.mean(0.5 * np.log(2 * np.pi * y_std ** 2) + 0.5 * z ** 2))
        ks_stat, ks_p = kstest(norm.cdf(z), "uniform")

        report[target] = dict(
            mahalanobis_vs_abs_z_spearman=dict(rho=float(rho_absz), p=float(p_absz)),
            coverage95_by_mahalanobis_quartile=cov_by_quart.round(4).to_dict(),
            by_condition=by_cond.round(4).to_dict(orient="index"),
            nlpd=nlpd, pit_ks_stat=float(ks_stat), pit_ks_p=float(ks_p),
        )
    return report


def grouped_conformal(ids, preds, rng):
    conditions = ids["Topology"] + "_" + ids["Cooling"]
    report = {}
    for target in TARGETS:
        safe = target.split("_[")[0]
        y_true, y_pred, y_std = preds[f"{safe}__y_true"], preds[f"{safe}__y_pred"], preds[f"{safe}__y_std"]
        abs_z = np.abs(y_true - y_pred) / y_std

        d = pd.DataFrame({"condition": conditions, "abs_z": abs_z})
        d["is_calib"] = False
        for cond, grp_idx in d.groupby("condition").groups.items():
            grp_idx = np.array(grp_idx)
            calib_idx = rng.choice(grp_idx, size=len(grp_idx) // 2, replace=False)
            d.loc[calib_idx, "is_calib"] = True
        calib, val = d[d["is_calib"]], d[~d["is_calib"]]

        q_global = conformal_quantile(calib["abs_z"].values)
        by_cond = {}
        for cond in sorted(d["condition"].unique()):
            c_calib = calib.loc[calib["condition"] == cond, "abs_z"].values
            c_val = val.loc[val["condition"] == cond, "abs_z"].values
            q_group = conformal_quantile(c_calib)
            by_cond[cond] = dict(
                n_calib=len(c_calib), n_val=len(c_val), q_group=q_group,
                coverage_default=float(np.mean(c_val <= 1.96)),
                coverage_global_conformal=float(np.mean(c_val <= q_global)),
                coverage_grouped_conformal=float(np.mean(c_val <= q_group)),
            )
        report[target] = dict(q_global=q_global, by_condition=by_cond)
    return report


# --------------------------------------------------------------------------

def main():
    df = load_pooled()
    ids = recover_fold_identifiers(df)
    mdist = mahalanobis_by_geometry(df)
    ids = ids.merge(mdist.rename("mahalanobis"), left_on="Run_ID", right_index=True)

    predictions_path = os.path.join(OUTPUT_DIR, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"{predictions_path} not found. Run python -m src.model_training first.")
    preds = np.load(predictions_path)

    uq_report = diagnose(ids, preds)
    uq_path = output_path("uq_report.json")
    with open(uq_path, "w") as f:
        json.dump(uq_report, f, indent=2)
    print(f"UQ diagnostics written to {uq_path}")

    rng = np.random.default_rng(SEED)
    conformal_report = grouped_conformal(ids, preds, rng)
    conformal_path = output_path("conformal_calibration.json")
    with open(conformal_path, "w") as f:
        json.dump(conformal_report, f, indent=2)
    print(f"Conformal calibration written to {conformal_path}")

if __name__ == "__main__":
    main()
