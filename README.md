# Replication of Papers

A research repository for reproducing and studying machine-learning papers, with the current strongest project focused on battery state-of-health prediction from electrochemical impedance spectroscopy data.

## Current Highlight: Battery SOH Prediction

Path: `化学/`

This reproduction studies lithium-ion battery state-of-health prediction using capacity data, EIS features, classical ML models, XGBoost, leave-one-battery-out validation, and SHAP interpretability.

## Problem

Battery health prediction is difficult because degradation signals are hidden inside noisy measurement curves. The project turns raw capacity and EIS files into engineered features, trains multiple models, evaluates generalization across batteries, and visualizes feature importance.

## What The Code Does

File: `化学/reproduce_battery_soh_v2.py`

- Loads capacity files from `Capacity data/`.
- Loads EIS files from `EIS data/`.
- Aligns EIS cycles with capacity-derived SOH labels.
- Extracts statistical EIS features from semicircle and tail regions.
- Adds sliding-window SOH features.
- Compares Lasso, Random Forest, SVR, MLP, and XGBoost.
- Uses leave-one-battery-out cross validation.
- Optimizes XGBoost with Optuna.
- Reports MAE, MSE, RMSE, and MAPE.
- Generates SOH prediction curves.
- Runs SHAP analysis for interpretability.

## Evidence Outputs

Generated artifacts:

- `化学/soh_prediction_curves.png`
- `化学/eis_split_check.png`
- `化学/shap_25C.png`
- `化学/shap_35C.png`
- `化学/shap_45C.png`

These outputs show both model behavior and feature-level explanations.

## Tech Stack

- Python
- NumPy, pandas
- scikit-learn
- XGBoost
- Optuna
- SHAP
- Matplotlib

## Research Notes

The repository also includes academic-writing notes under `如何写出一篇高质量论文/`, covering research proposals, writing style, presentation skills, and literature reading strategy.

## How To Run The Main Reproduction

```powershell
cd 化学
python -m venv .venv
.\.venv\Scripts\python -m pip install numpy pandas scikit-learn xgboost shap optuna matplotlib
.\.venv\Scripts\python reproduce_battery_soh_v2.py
```

## What This Repository Demonstrates

- Reading and transforming raw experimental data.
- Reproducing an ML pipeline from a research idea.
- Comparing baselines instead of only reporting one model.
- Using cross-validation suited to battery-level generalization.
- Explaining model behavior with SHAP rather than treating the model as a black box.
