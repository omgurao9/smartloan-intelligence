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
- README.md: minimal starter text.
- .gitignore: excludes CSV/XLSX datasets, common Python artifacts, secrets, local databases, and frontend build/dependency folders.
- CLAUDE.md: project working rules.
- DATA_AUDIT.md: verified comparison of raw/clean datasets and starter-code review.
- Existing ZIP scaffold: useful but not production-ready; see DATA_AUDIT.md for concrete defects.

## Verified checks
- Raw CSV copies are byte-identical; ZIP cleaned CSV matches separately uploaded cleaned CSV row-for-row.
- Raw: 45,000 × 14, 0 missing, 0 duplicates.
- Clean: 44,991 × 14, 0 missing, 0 duplicates; nine records excluded by the documented age/experience rule.
- `python -m compileall -q scripts ml backend tests`: passed.
- `python -m pytest -q`: 2 passed.
- Bare `pytest -q` failed import discovery in the review environment; use `python -m pytest` or add test configuration.
- No trained model artifacts were present in the ZIP.

## Important data limitation
The dataset has an approval/status label, not a post-loan repayment/default outcome. The project must describe approval/status classification, not future default prediction. The previous-default feature has a very strong one-way relationship with the target; see DATA_AUDIT.md.

## Next action
Repair the training and calibration/evaluation scripts before model training. Fix SVM sampling, create a leakage-safe calibration strategy, keep the test split untouched until final evaluation, and generate actual artifacts/metrics. Do not ask Claude to redo the dataset audit.
