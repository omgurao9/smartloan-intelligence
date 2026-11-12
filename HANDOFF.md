# SmartLoan Intelligence — Handoff

## Repository
https://github.com/omgurao9/smartloan-intelligence
- Default branch: main
- Visibility: public
- Keep datasets, applicant records, secrets, local databases, and model artifacts out of Git.

## Project goal
Build an explainable and responsible local loan-risk assessment platform with a strong ML evaluation component and a polished dark fintech dashboard.

## Constraints
- Zero budget; no paid APIs or hosted services.
- Windows 11, 16 GB RAM; CPU-first implementation.
- Local demonstration.
- Prefer small, independently verifiable tasks.
- Claude Web Chat Free should produce working deliverables early and stop optional work before usage is exhausted. ChatGPT handles architecture, data audit, ML review, debugging, integration, tests, and documentation where possible.

## Planned components
1. Data validation and leakage-safe preprocessing.
2. Model training, comparison, evaluation, and calibration.
3. SHAP global/local explanations.
4. Constrained what-if/counterfactual analysis.
5. Group fairness metrics and limitations.
6. FastAPI backend and local history if useful.
7. React/Vite/Tailwind dashboard with charts.
8. End-to-end testing and academic documentation.

## Dataset status
Earlier project notes mention:
- loan_data(1).csv: 45,000 rows, 14 columns.
- loan_data_clean.csv: 44,991 rows, 14 columns.
- The previously cleaned version excluded nine unusual age/experience records.
These are historical notes only. Reinspect the actual files and validate target meaning, feature availability, outliers, and preprocessing before relying on these findings. Never commit either dataset.

## Current repository state
- README.md now contains Windows local setup, evaluation commands, and links to architecture, experiment report, model card, and viva preparation.
- ARCHITECTURE.md, MODEL_CARD.md, EXPERIMENT_REPORT.md, and VIVA_PREPARATION.md are committed.
- CLAUDE.md contains concise output-first working rules.
- DATA_AUDIT.md contains dataset findings, code review, and measured model results.
- Python ML preprocessing, cleaning, EDA, split, training, calibration, SHAP and subgroup fairness scripts, FastAPI backend, tests, and a five-view React/Vite dashboard are in the repository.
- API includes prediction using neutral class-0/class-1 terminology, constrained financial what-if scenarios, model metrics, final evaluation, SHAP global report, fairness report, and local history endpoints.
- Frontend package dependencies are pinned for reproducibility. The dashboard connects to prediction, model-performance, evaluation, SHAP, fairness, and what-if endpoints. Dashboard styles use local CSS without remote font dependencies.
- Generated CSVs, model binaries, and SQLite history remain local and are not committed.

## Verified local run
- Cleaned dataset: 44,991 rows.
- Stratified split: train 31,493; validation 6,749; test 6,749.
- Eight baselines trained; XGBoost selected by validation PR-AUC (0.9336).
- Sigmoid calibration fitted with 5-fold stratified CV on training only.
- Threshold selected on validation: 0.28.
- Final holdout: accuracy 0.9255, balanced accuracy 0.8990, precision 0.8202, recall 0.8513, F1 0.8355, ROC-AUC 0.9770, PR-AUC 0.9351, Brier 0.0517.
- `python -m compileall -q backend scripts ml tests`: passed.
- `python -m pytest -q`: 5 passed, including API prediction, local history, and constrained what-if test.
- `python -m scripts.05_ablation`: completed; removing previous-default history reduced validation ROC-AUC by 0.0547 and PR-AUC by 0.0748 in the Random Forest ablation.
- `python -m scripts.01_clean_data`, `python -m scripts.03_split_data`, and `python -m scripts.05_ablation` verified successfully using module execution.
- `PYTHONPATH=. python scripts/07_shap.py`: completed; global and local SHAP JSON reports generated for 300 validation examples across 5 folds.
- `PYTHONPATH=. python scripts/08_fairness.py`: completed; gender, education, and age-band subgroup metrics generated on the held-out test split.
- Frontend build has not yet been run in this environment. Claude's previous frontend response did not implement the requested dashboard views; ChatGPT replaced main.jsx and style.css with the complete dashboard. The user must run npm install and npm run build locally to verify it.

## Important data limitation
The target is `loan_status`; the data does not prove future repayment/default outcomes. Describe the app as a loan-status classifier. A strong one-way relationship exists between previous-default history and target; see DATA_AUDIT.md.

## Next action
User next action: run the Windows local setup in README.md, including npm install and npm run build in frontend. Fix any build/runtime errors from the actual laptop environment. Do not retrain or tune against held-out test results.
