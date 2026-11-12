# Dataset and Starter-Code Audit

## Scope
Audited the uploaded `SmartLoan-AI-complete.zip` and compared its raw/cleaned CSVs with the separately uploaded dataset files. Findings below are from direct inspection of those files and source code, not from unverified model runs.

## Dataset identity and integrity
- Raw dataset: 45,000 rows × 14 columns.
- The two separately uploaded raw CSVs are byte-identical to the ZIP's `data/raw/loan_data.csv`.
- The separately uploaded cleaned CSV is row-for-row identical to the ZIP's `data/processed/loan_data_clean.csv`.
- Raw data has 0 missing values and 0 duplicate rows.
- Cleaned data has 44,991 rows, 0 missing values and 0 duplicate rows.
- Target counts in raw data: `loan_status=0`: 35,000; `loan_status=1`: 10,000.
- Target counts after cleaning: `loan_status=0`: 34,991; `loan_status=1`: 10,000.
- Project context labels 0 as rejected and 1 as approved. This is the starter project's stated interpretation; retain that caveat until the original dataset source/provenance is documented.

## Cleaning rule
The existing cleaning code removes rows where `person_age >= 84` OR `person_emp_exp > person_age - 18`. Exactly nine rows are removed. These are extreme-age/experience records (ages 84–144, experience 61–125). All nine have `loan_status=0`. The cleaned file matches this rule exactly.

This is a transparent rule, but its impact should be documented. Keep the raw file unchanged and reproduce cleaning from code; do not silently overwrite the raw dataset.

## Important target-feature relationship
Cross-tabulation of `previous_loan_defaults_on_file` against `loan_status` in the raw data:

| Previous default on file | Status 0 | Status 1 |
|---|---:|---:|
| No | 12,142 | 10,000 |
| Yes | 22,858 | 0 |

Thus, no record with a previous default marked Yes has target status 1. This feature is available in the dataset and may be a legitimate historical underwriting input, but the perfect one-way relationship is unusually strong and may dominate model behavior. It must be highlighted in the report and ablation study. It does not establish that the model predicts repayment/default.

## Target limitation
The dataset contains an approval/status label, not a post-origination repayment outcome. The project can model the dataset's recorded loan-status/approval label (subject to source verification). It must not claim to estimate probability of future default, credit loss, or actual borrower repayment risk.

## Starter code review
The ZIP is useful as a scaffold, not a finished implementation.

Verified:
- Python source compiles successfully with `python -m compileall -q scripts ml backend tests`.
- Existing tests pass with `python -m pytest -q`: 2 passed.
- Running the bare `pytest -q` command in this review environment failed to import the local `ml` package; use `python -m pytest` or add explicit project test configuration.

Issues to fix before relying on the pipeline:
1. `scripts/04_train_models.py` passes unsupported `stratify=` to `DataFrame.sample()` for the SVM subsample.
2. `scripts/06_calibration.py` calibrates on validation data and reports performance/threshold on that same data. Use a separate calibration subset or leakage-safe cross-validation, use validation for threshold selection, and reserve test data for one final evaluation.
3. The backend asks applicants to submit `loan_percent_income`; derive it from loan amount and income on the server.
4. The backend reloads the serialized model on every prediction and hardcodes the decision threshold. Load once at startup and read the approved threshold from a model metadata/report artifact.
5. The UI currently contains nonfunctional navigation items and manually exposes the derived loan-to-income ratio.
6. `09_q_learning.py` is an isolated toy reward simulation, not a trained loan-risk model. Keep it out of the core prediction claims; include it only if there is a clear educational justification.
7. SHAP, calibration and clustering scripts need end-to-end verification after the training pipeline is repaired. No model artifacts or validated model metrics were included in the ZIP.

## Approved next steps
1. Repair and test data preparation/model training.
2. Establish a leakage-safe calibration, threshold-selection and final test-evaluation procedure.
3. Generate model artifacts and actual evaluation reports locally.
4. Repair backend request validation and model loading.
5. Improve the frontend after the API contract is stable.
6. Add explainability/fairness features only after their outputs can be tested and accurately described.

## Privacy and repository handling
The GitHub repository is intentionally public by user choice. Never commit applicant-level CSVs, local databases, model binaries, secrets, or other generated private artifacts. The repository's existing `.gitignore` should be retained and checked before local files are pushed.
