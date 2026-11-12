# Model Card — SmartLoan Intelligence

## Model identity
- Selected estimator: XGBoost classifier.
- Probability calibration: sigmoid calibration with 5-fold stratified cross-validation on training data.
- Classification threshold: 0.28, selected by maximum validation F1.
- Target: `loan_status`, class 1.

## Intended use
Educational demonstration of tabular classification, model comparison, probability calibration, explainability, subgroup evaluation, and API integration using the supplied dataset.

## Out-of-scope use
- Real loan underwriting or credit eligibility decisions.
- Prediction of future repayment, delinquency, loss, or default.
- Use as a production financial service.
- Automated adverse decisions affecting real applicants.

## Dataset
- 45,000 raw rows; 44,991 after the project's documented age/experience cleaning rule.
- 14 columns including target.
- Class counts before cleaning: 35,000 class 0 and 10,000 class 1.
- Source provenance and exact target-label semantics have not yet been independently established.

## Evaluation
Stratified train/validation/test split: 31,493 / 6,749 / 6,749.

Held-out test metrics at the validation-selected threshold:
- Accuracy: 0.9255
- Balanced accuracy: 0.8990
- Precision: 0.8202
- Recall: 0.8513
- F1: 0.8355
- ROC-AUC: 0.9770
- PR-AUC: 0.9351
- Brier score: 0.0517

See `reports/models/final_evaluation.json` and `EXPERIMENT_REPORT.md`.

## Important feature behavior
Previous-default history has a very strong one-way relationship with the target. In the inspected dataset, all class-1 records are in the "No previous default" group. A validation-only Random Forest ablation showed ROC-AUC decreasing by 0.0547 and PR-AUC decreasing by 0.0748 when this feature was removed. This dependence requires careful discussion and source verification.

## Explainability
SHAP is used to describe how the underlying XGBoost fold estimators contribute to model scores. It does not establish causality and the reported values are not direct explanations of calibrated probabilities.

## Fairness
Subgroup metrics are descriptive sample statistics for gender, education, and age bands. They do not certify fairness, legal compliance, or absence of discrimination. Small groups are suppressed in the generated report.

## Limitations
1. Target-label provenance is not verified.
2. Dataset may contain historical process patterns and proxy relationships.
3. Strong dependence on previous-default history may inflate apparent performance.
4. No external or temporal validation dataset is available.
5. The project cleaning rule excludes nine records using an assumed age-18 employment start.
6. Metrics do not demonstrate real-world generalization.

## Warnings
Outputs are class-label estimates for academic demonstration only. They are not financial advice, causal recommendations, or real credit decisions.
