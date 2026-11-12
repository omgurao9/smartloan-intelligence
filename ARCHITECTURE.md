# System Architecture

## Overview
SmartLoan Intelligence is a local, single-user educational prototype. It has no hosted service dependency.

```text
CSV dataset
   |
   v
Data cleaning -> EDA -> Stratified 70/15/15 split
   |
   v
scikit-learn preprocessing pipeline
   |
   +--> Baseline model comparison (train -> validation)
   |
   +--> Selected estimator -> 5-fold sigmoid calibration on train
                                  |
                                  v
                        Validation threshold selection
                                  |
                                  v
                          One final test evaluation
                                  |
                  +---------------+----------------+
                  |               |                |
                  v               v                v
              SHAP report    Fairness audit    Model artifacts
                  |               |                |
                  +---------------+----------------+
                                  |
                                  v
                         FastAPI local backend
                          |       |       |
                       Predict  What-if  Reports
                          |
                          v
                     SQLite history
                          |
                          v
                    React + Vite UI
```

## Data and ML layer
- `ml/preprocessing/pipeline.py` owns cleaning, feature creation, numeric scaling, categorical one-hot encoding, and optional feature exclusion for ablation.
- `scripts/01_clean_data.py` creates the cleaned dataset from the immutable raw CSV.
- `scripts/02_eda.py` creates descriptive findings and plots.
- `scripts/03_split_data.py` creates stratified train/validation/test splits with seed 42.
- `scripts/04_train_models.py` trains baseline classifiers and compares validation PR-AUC. It does not load the test split.
- `scripts/05_ablation.py` measures the impact of previous-default history on validation metrics.
- `scripts/06_calibration.py` selects the best baseline by validation PR-AUC, calibrates it using 5-fold CV on training data, selects a threshold using validation F1, and evaluates test once.
- `scripts/07_shap.py` creates global and local SHAP summaries for the XGBoost fold estimators.
- `scripts/08_fairness.py` calculates descriptive subgroup metrics on the held-out test split at the frozen threshold.

## Backend
FastAPI loads the final calibrated model and metadata once during application startup. Main routes:
- `GET /api/health`
- `POST /api/predict`
- `POST /api/what-if`
- `GET /api/model-performance`
- `GET /api/final-evaluation`
- `GET /api/explainability/global`
- `GET /api/fairness`
- `GET /api/predictions`

The backend derives `loan_percent_income = loan_amnt / person_income`; clients do not submit this engineered feature separately. SQLite stores local prediction history.

## Frontend
React + Vite provides the applicant form and displays the model's class-1 estimate. The derived ratio is shown as an automatically calculated value. The UI is an educational interface and must not imply real underwriting authority.

## Local artifacts
Generated locally and intentionally excluded from Git:
- `data/raw/*.csv`, processed datasets and splits
- `models/*.joblib`
- `database/*.db`
- Python virtual environments and frontend build/dependency folders

## Key design constraints
- Never fit preprocessing on validation or test data.
- Never use test labels to select a model or threshold.
- Do not describe SHAP as causal.
- Do not modify protected or immutable attributes in what-if scenarios.
- The target is `loan_status`; its provenance and semantics require confirmation before any approval/default interpretation.
