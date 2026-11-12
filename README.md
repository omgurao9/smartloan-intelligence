# SmartLoan Intelligence

**An Explainable and Responsible AI Platform for Loan-Status Assessment**

A local, educational ML application that compares classification models, calibrates probabilities, exposes model evaluation, and provides an API/UI for exploring predictions.

> **Important limitation:** The supplied dataset contains a `loan_status` label. The project has not established that this label represents future repayment or default. Outputs must be described as estimates of the dataset label, not real lending decisions or future-default predictions.

## Stack
- Python, Pandas, NumPy, scikit-learn, XGBoost
- FastAPI, SQLite
- React, Vite
- SHAP and fairness analysis are planned follow-on modules

## Local setup
1. Put the supplied dataset at `data/raw/loan_data.csv` (this file is intentionally not tracked by Git).
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the pipeline from the repository root:
   ```bash
   python scripts/01_clean_data.py
   python scripts/02_eda.py
   python scripts/03_split_data.py
   python scripts/04_train_models.py
   python scripts/06_calibration.py
   python -m pytest -q
   ```
5. Start the API: `uvicorn backend.main:app --reload`.
6. Start the frontend in another terminal: `cd frontend && npm install && npm run dev`.

The training scripts create local data splits, model binaries, and evaluation reports. These generated artifacts are ignored by Git and must be created locally before using the prediction API.

## Evaluation protocol
- Stratified 70/15/15 train/validation/test split.
- Preprocessing is part of the sklearn pipeline and is fit on training data.
- Baselines are compared on validation PR-AUC.
- Probability calibration uses stratified cross-validation within the training split.
- Threshold selection uses validation F1.
- The held-out test split is evaluated once at the selected threshold.

See [DATA_AUDIT.md](DATA_AUDIT.md) for the verified dataset audit and known limitations. See [HANDOFF.md](HANDOFF.md) for current development status.
