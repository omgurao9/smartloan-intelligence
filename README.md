# SmartLoan Intelligence

**An Explainable and Responsible AI Platform for Loan-Status Assessment**

A local educational ML application that compares classification models, calibrates probabilities, and provides prediction, explainability, subgroup-audit, and what-if views through FastAPI and React.

> **Important limitation:** The supplied dataset contains a `loan_status` label. The project has not established that this label represents future repayment or default. Outputs are estimates of the dataset label, not real lending decisions or future-default predictions.

## Stack
- Python, Pandas, NumPy, scikit-learn, XGBoost
- SHAP and subgroup fairness metrics
- FastAPI and SQLite
- React, Vite and Recharts

## Local setup (Windows)
1. Put the original dataset at `data/raw/loan_data.csv`. Do not commit the CSV.
2. Open PowerShell in the repository root.
3. Create and activate a virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install Python packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Generate cleaned data, splits, models and reports:
   ```powershell
   python -m scripts.01_clean_data
   python -m scripts.02_eda
   python -m scripts.03_split_data
   python -m scripts.04_train_models
   python -m scripts.06_calibration
   python -m scripts.05_ablation
   python -m scripts.07_shap
   python -m scripts.08_fairness
   python -m pytest -q
   ```
6. Start the API in the first terminal:
   ```powershell
   uvicorn backend.main:app --reload
   ```
7. In a second PowerShell terminal:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
8. Open the Vite URL printed in the terminal (normally `http://127.0.0.1:5173`).

The API expects generated model files and reports to exist locally. If the dashboard says the API/model is offline, confirm the Python pipeline completed and the API terminal is still running.

## Evaluation protocol
- Stratified 70/15/15 train/validation/test split.
- Preprocessing is inside the sklearn pipeline and fitted on training data.
- Baselines are compared on validation PR-AUC.
- Probability calibration uses stratified cross-validation within the training split.
- Threshold selection uses validation F1.
- The held-out test split is evaluated once at the selected threshold.

## Project documentation
- [Architecture](ARCHITECTURE.md)
- [Data audit](DATA_AUDIT.md)
- [Experiment report](EXPERIMENT_REPORT.md)
- [Model card](MODEL_CARD.md)
- [Viva preparation](VIVA_PREPARATION.md)
- [Development handoff](HANDOFF.md)

## Privacy and limitations
The repository is public by choice. Do not commit applicant records, the dataset, trained model binaries, SQLite history, API keys, or other private information. The app is an academic prototype, not a production credit-scoring service.
