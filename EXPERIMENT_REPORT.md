# SmartLoan Intelligence — Experiment Report

## 1. Objective and scope
This experiment compares supervised classifiers for the supplied `loan_status` target. The dataset does not establish a post-origination repayment/default outcome. Therefore, the model is a dataset-label classifier, not a validated default-risk model and not a real credit decision system.

## 2. Data
- Raw observations: 45,000.
- Cleaned observations: 44,991.
- Features: 13 input columns plus the target.
- Missing values and duplicate rows: none in the inspected raw and cleaned files.
- Target counts before cleaning: class 0 = 35,000; class 1 = 10,000.
- Cleaning excludes nine observations using the project rule age >= 84 or employment experience > age - 18. This is a transparent but assumption-dependent rule and should be challenged in a limitations discussion.
- A strong one-way relationship was observed: all 10,000 records with target class 1 have `previous_loan_defaults_on_file = No`, while the Yes group contains no class-1 records. This may dominate model behavior and must be discussed.

## 3. Experimental protocol
- Reproducible random seed: 42.
- Stratified split: 70% train (31,493), 15% validation (6,749), 15% test (6,749).
- Feature creation and preprocessing are in the scikit-learn pipeline; preprocessing is fitted on training data.
- Baselines are compared on validation data. The test split is not read by the baseline-training script.
- SVM uses a reproducible stratified 12,000-row training subset for computational practicality.
- The selected model is calibrated with sigmoid calibration using 5-fold stratified cross-validation within the training split.
- Threshold is selected by maximum validation F1. The test split is evaluated once at that fixed threshold.

## 4. Baseline comparison on validation
| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9326 | 0.8764 | 0.9079 | 0.7753 | 0.8364 | 0.9754 | 0.9336 |
| Random Forest | 0.9161 | 0.8963 | 0.7834 | 0.8607 | 0.8202 | 0.9720 | 0.9249 |
| Gradient Boosting | 0.9227 | 0.8617 | 0.8826 | 0.7520 | 0.8121 | 0.9693 | 0.9185 |
| Decision Tree | 0.8855 | 0.8845 | 0.6892 | 0.8827 | 0.7740 | 0.9629 | 0.8935 |
| SVM | 0.8785 | 0.8838 | 0.6700 | 0.8933 | 0.7657 | 0.9573 | 0.8778 |
| KNN | 0.9007 | 0.8081 | 0.8793 | 0.6413 | 0.7417 | 0.9421 | 0.8671 |
| Logistic Regression | 0.8558 | 0.8716 | 0.6213 | 0.9000 | 0.7351 | 0.9503 | 0.8370 |
| Naive Bayes | 0.7403 | 0.8254 | 0.4603 | 0.9787 | 0.6261 | 0.9343 | 0.7991 |

## 5. Calibration and final test results
XGBoost was selected by validation PR-AUC (0.9336). Sigmoid calibration was fitted using 5-fold stratified CV within training. The threshold maximizing validation F1 was 0.28.

| Metric | Held-out test |
|---|---:|
| Accuracy | 0.9255 |
| Balanced accuracy | 0.8990 |
| Precision | 0.8202 |
| Recall | 0.8513 |
| Specificity | 0.9467 |
| F1 | 0.8355 |
| ROC-AUC | 0.9770 |
| PR-AUC | 0.9351 |
| Brier score | 0.0517 |

Confusion matrix (actual rows, predicted columns; labels 0 then 1):
```
[[4969, 280],
 [ 223, 1277]]
```

## 6. Explainability
Global SHAP was computed over 300 validation examples and averaged across the five XGBoost fold estimators. The largest mean absolute contribution was `previous_loan_defaults_on_file_Yes`, followed by interest rate, loan-to-income ratio, and income. This aligns with the strong target relationship identified in the data audit.

SHAP values here explain the underlying XGBoost fold-model scores, not calibrated probabilities and not causal effects. Local examples are illustrative, not recommendations.

## 7. Subgroup audit
Descriptive subgroup metrics were calculated on the held-out test split at the already-frozen threshold:
- Gender TPR: female 0.8474; male 0.8542.
- Age-band TPR: 18–25 = 0.8771; 26–35 = 0.8231; 36–45 = 0.8417; 46–60 = 0.8824. The 61–83 group has only 10 observations and is suppressed.
- Education subgroup metrics are included in `reports/fairness/subgroup_metrics.json`.

These are sample-specific descriptive metrics. They do not establish legal compliance or fairness. The held-out results must not be used for threshold tuning.

## 8. Limitations and next experiments
1. Verify dataset source and exact semantics of `loan_status` before interpreting class 1 as approval.
2. Investigate whether previous-default history is a legitimate point-in-time feature and why it perfectly separates one target subgroup.
3. Run an ablation without previous-default history using validation only; report the performance change without retuning on test.
4. Examine calibration plots and calibration by subgroup.
5. Validate performance on an independent dataset or later time period if one becomes available.
6. Do not claim future default prediction, causality, fairness certification, or real-world lending suitability.

## 9. Reproducibility
Run the scripts in README.md from the repository root. The final machine-readable report is `reports/models/final_evaluation.json`. Model binaries, datasets, and local prediction history are generated locally and are not committed.


## 10. Feature ablation: previous-default history
A validation-only Random Forest ablation was performed with identical model settings and split:
- With `previous_loan_defaults_on_file`: ROC-AUC 0.9720; PR-AUC 0.9249.
- Without it: ROC-AUC 0.9173; PR-AUC 0.8501.
- Difference after removing it: ROC-AUC -0.0547; PR-AUC -0.0748.

This indicates strong dependence on previous-default history in this dataset. It does not establish that the feature is inappropriate, but the feature's provenance, timing, and target relationship deserve explicit scrutiny.
