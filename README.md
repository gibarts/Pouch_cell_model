# Surrogate-based design of large-format CTP pouch cells

This repository contains the dataset and Python code for the article *Surrogate-Based
Multi-Objective Design of Large-Format CTP Pouch Cells: Sensitivity Analysis and Conformal
Uncertainty Quantification Across Fast-Charge and Drive-Cycle Conditions*.

The workflow generates 800 two-dimensional PyBaMM simulation cases (200 cell geometries
under four tab-topology and cooling configurations), trains a pooled mixed-kernel Gaussian
process surrogate, calibrates its uncertainty with grouped (Mondrian) split-conformal
prediction, and runs a Sobol sensitivity analysis and a two-factor ANOVA on the calibrated
surrogate.

## Contents

| Folder | Contents |
|---|---|
| `data/` | The 800-case dataset (four files, described below) and the drive-cycle inputs in `data/inputs/` |
| `src/` | The pipeline; each step is run as `python -m src.<module>` from the repository root |
| `results/` | Reference outputs of each step, for comparison |
| `outputs/` | Created when the scripts run; everything they write goes here |

## Installation

Python 3.12 was used. Install the pinned dependencies:

```
pip install -r requirements.txt
```

The US06 speed trace is included (`data/inputs/us06_epa.txt`, from the US EPA
[dynamometer drive schedules](https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules)).
The WLTP speed trace is not redistributed: download document **WLTP-DHC-12-07e.xls** from the
UNECE WLTP DHC 12th session page, <https://unece.org/dhc-12th-session>, and save it as
`data/inputs/WLTP-DHC-12-07e.xls`. Its `WLTC_class_3` sheet is used. Every step except the
validation and the mesh study needs this file.

## Reproducing the results from the dataset

Run from the repository root, in this order:

| Step | Command | Run time* | Produces |
|---|---|---|---|
| 1 | `python -m src.model_training` | ~30 min | Grouped five-fold cross-validation of the seven objective surrogates: R², RMSE and 95% interval coverage per fold (`cv_metrics.jsonl`, `cv_results.log`) and the out-of-fold predictions (`predictions.npz`). The final surrogates, fitted on all 800 cases (`models/`) |
| 2 | `python -m src.uq_calibration` | <1 min | Uncertainty diagnostics: interval coverage against Mahalanobis distance in the design space and per topology/cooling condition, negative log predictive density and a PIT uniformity test (`uq_report.json`). Grouped conformal calibration: the interval multiplier and the held-out coverage before and after calibration, per condition (`conformal_calibration.json`) |
| 3 | `python -m src.sensitivity` | ~3 min | Sobol first-order, total-effect and second-order indices with 95% bootstrap intervals for the eight design variables, per objective and condition (`sensitivity_indices.json`, `sensitivity_indices.csv`). One-way and two-way ANOVA of topology and cooling on the raw and log scale, with the condition means (`categorical_sensitivity.csv`) |
| 4 | `python -m src.validation` | ~10 min | RMSE and MAE of the terminal voltage, temperature rise and cell thickness change of a 1C discharge against the measurements of Ai et al. (2020) (`validation_metrics.json`) |
| 5 | `python -m src.mesh_sensitivity` | ~5 min | Maximum thermal gradient on the production mesh (10 × 4 in-plane nodes) and a twice-refined mesh (20 × 8) for the smallest and largest cell, NT topology, tab cooling, 2C discharge, with the change between meshes and the solve times (`mesh_sensitivity_results_Tab.csv`) |

\*On a 14-thread desktop CPU.

Step 2 reads the cross-validation predictions written by step 1, and step 3 loads the final
surrogates from step 1. Steps 4 and 5 are independent of the others. The validation data of
Ai et al. (2020) are downloaded by PyBaMM on first use.

Outputs are written to `outputs/`; compare them with `results/`.

## Regenerating the dataset

Each case (geometry, topology, cooling) is an independent simulation:

```
python -m src.optimisation --cooling Tab --topology NT --run-id 5
```

`--cooling` is `Tab` or `Surface`; `--topology` takes `CT`, `NT` or both; `--run-id` selects
one of the 200 Latin-hypercube geometries (0–199) and runs all of them if omitted. Each case
writes one shard to `outputs/shards/<cooling>/`. The 2D solves take about 20 minutes per case
under surface cooling and 45 minutes under tab cooling, so the 800 cases need roughly
430 CPU-hours; running cases in parallel is advisable. Then compile the shards into dataset
files in `outputs/datasets/`:

```
python -m src.compile_dataset --cooling Tab
```

The dataset was generated on Linux with PyBaMM 25.8.0. On Windows, the same version raised
`NotImplementedError: Shape not recognized` at a step transition of the WLTP fast-charge
experiment; the surrogate, calibration, sensitivity, validation and mesh steps run on both.

## Dataset

| File | Topology | Cooling |
|---|---|---|
| `Master_CT_Optimization_Dataset.csv` | Counter-Tab | Tab |
| `Master_CT_Optimization_Dataset_Surface.csv` | Counter-Tab | Surface |
| `Master_NT_Optimization_Dataset.csv` | Normal-Tab | Tab |
| `Master_NT_Optimization_Dataset_Surface.csv` | Normal-Tab | Surface |

Each file has 200 rows, one per geometry; `Run_ID` identifies the same geometry in all four.

| Columns | Meaning |
|---|---|
| `Thickness_Factor` | Electrode stack thickness scaling factor (–) |
| `L_ccp_[m]`, `L_ccn_[m]` | Positive and negative current-collector foil thickness |
| `Cell_Width_[m]`, `Cell_Height_[m]` | Active-area width and height |
| `Pos_Tab_Frac`, `Neg_Tab_Frac` | Tab centre positions along the tab edge, as fractions of the cell width (–) |
| `Tab_Width_Frac` | Tab width as a fraction of the cell width (–) |
| `OF1_Specific_Energy_[Wh/kg]` | Specific energy from a 1D isothermal 2C discharge to 3.0 V |
| `OF2_<cycle>_Thermal_Gradient_[K]` | Maximum in-plane temperature difference over the cycle, ΔT_max |
| `OF3_<cycle>_Peak_Swelling_[m]` | Largest absolute through-plane thickness change from the start of the cycle |
| `Diag_<cycle>_Signed_Swelling_[m]` | The same thickness change with its sign: negative is contraction, positive is expansion |
| `Diagnostic_Mass_[kg]` | Cell mass used for the specific energy |
| `Solve_Time_Total_[s]` | Solver time of the three 2D simulations |
| `Error` | `Success` for every published case |

`<cycle>` is `US06`, `WLTP` or `WLTP_FastCharge` (the WLTP cycle, a 300 s rest and a 2C
constant-current charge to 4.2 V). The drive-cycle current of each cell follows from a
vehicle model (2000 kg, drag coefficient 0.24, frontal area 2.2 m², rolling-resistance
coefficient 0.015, drivetrain efficiency 0.90) and a pack sized around the cell: 222 cells
in series for 800 V, and the smallest number of parallel strings that installs 80 kWh
within a 5C peak-power limit.

## Reproducibility

The Latin hypercube (seed 42), the Saltelli sample and its bootstrap (seed 42) and the
calibration/validation split of the conformal step (seed 0) are fixed. On the same machine,
re-running steps 1–3 and 5 reproduces `results/` exactly, apart from the solve times of the
mesh study. The validation metrics of step 4 vary between runs by less than one part in 10⁹,
from the PyBaMM solver. On a different platform or library build, expect differences in the
last digits of the surrogate fits, and of the simulations (about 0.1–0.3% was observed
between Linux and Windows).
