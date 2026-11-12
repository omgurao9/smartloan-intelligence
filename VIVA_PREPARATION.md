# SmartLoan Intelligence — Viva Preparation

## Project explanation
SmartLoan Intelligence is a local educational machine-learning platform. It compares classifiers for the dataset's `loan_status` label, calibrates the selected model, evaluates it on a held-out test split, and exposes prediction, explainability, fairness, and what-if functionality through a FastAPI backend and React frontend.

## Likely viva questions

### 1. What problem does your project solve?
It demonstrates a full tabular ML workflow for classifying a loan-status label, including model comparison, calibration, explanations, subgroup analysis, and a local application interface.

### 2. Why did you use multiple algorithms?
Different algorithms capture different patterns and have different trade-offs. Comparing them on the same stratified validation split provides a more defensible model-selection process than choosing one arbitrarily.

### 3. Why is accuracy not enough?
The target is imbalanced. Accuracy can hide poor performance on the minority class. Precision, recall, F1, balanced accuracy, ROC-AUC, PR-AUC, and calibration provide complementary information.

### 4. Why did you choose XGBoost?
It achieved the highest validation PR-AUC among the tested baselines (0.9336). Random Forest had higher validation balanced accuracy and recall at the default threshold, so the selection criterion and trade-off must be explained.

### 5. What is data leakage?
Leakage occurs when information unavailable at prediction time or information from validation/test data influences training. The preprocessing is kept inside the sklearn pipeline and fitted on training folds only; the baseline training script does not load the test split.

### 6. Why calibrate probabilities?
A classifier can rank examples well while its probability estimates are poorly calibrated. Sigmoid calibration maps model scores to better-behaved probabilities using cross-validated training predictions.

### 7. How did you choose the threshold?
The threshold maximizing F1 on the validation split was selected (0.28). The held-out test split was evaluated once at that fixed threshold.

### 8. What is SHAP?
SHAP assigns feature contribution values based on a game-theoretic framework. Here it explains the underlying XGBoost fold-model scores; it does not prove causality or directly explain the calibrated probability.

### 9. What is fairness analysis?
The project compares selection rate, true-positive rate, false-positive rate, precision, and accuracy across selected demographic groups. These are descriptive metrics, not proof of fairness or legal compliance.

### 10. What is the purpose of the what-if simulator?
It shows how the model output changes under hypothetical changes to selected financial or loan-term inputs. It is not a causal model, a recommendation, or a guarantee of approval.

### 11. What is the biggest limitation?
The dataset target is `loan_status`, not a verified future repayment/default outcome. Also, previous-default history has an unusually strong one-way relationship with the target and substantially affects model performance.

### 12. What would you improve next?
Verify dataset provenance and target semantics, use external or time-based validation, investigate feature availability at decision time, assess subgroup calibration, and improve robustness and monitoring before any real-world use.

## Metrics to remember
- Selected baseline: XGBoost.
- Validation PR-AUC: 0.9336.
- Threshold: 0.28.
- Final test accuracy: 0.9255.
- Final test F1: 0.8355.
- Final test ROC-AUC: 0.9770.
- Final test PR-AUC: 0.9351.
- Final test Brier score: 0.0517.
- Test split: 6,749 records.

## Be precise
Say "classification of the supplied `loan_status` label." Do not claim "future loan default prediction" unless the original dataset documentation establishes that target meaning.
