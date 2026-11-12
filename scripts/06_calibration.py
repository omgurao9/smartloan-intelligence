"""Calibrate the validation-selected model and evaluate once on held-out test data."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

def classification_report(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def main():
    model_dir = ROOT / "models"
    report_dir = ROOT / "reports/models"
    comparison_path = report_dir / "baseline_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Run scripts/04_train_models.py first.")
    comparison = pd.read_csv(comparison_path)
    if comparison.empty or "model" not in comparison or "pr_auc" not in comparison:
        raise ValueError("Baseline comparison is empty or missing model/pr_auc columns.")
    selected_name = str(comparison.sort_values("pr_auc", ascending=False).iloc[0]["model"])
    baseline_path = model_dir / f"{selected_name}_baseline.joblib"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Selected baseline model not found: {baseline_path}")
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    val = pd.read_csv(ROOT / "data/processed/val.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    X_train, y_train = train.drop(columns="loan_status"), train["loan_status"]
    X_val, y_val = val.drop(columns="loan_status"), val["loan_status"]
    X_test, y_test = test.drop(columns="loan_status"), test["loan_status"]
    calibrated = CalibratedClassifierCV(
        estimator=joblib.load(baseline_path), method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        ensemble=True,
    )
    calibrated.fit(X_train, y_train)
    val_probabilities = calibrated.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_val, val_probabilities >= threshold, zero_division=0) for threshold in thresholds]
    selected_threshold = float(thresholds[int(np.argmax(scores))])
    test_probabilities = calibrated.predict_proba(X_test)[:, 1]
    model_path = model_dir / "final_calibrated_model.joblib"
    joblib.dump(calibrated, model_path)
    report = {
        "selected_model": selected_name,
        "selection_metric": "validation PR-AUC from baseline_comparison.csv",
        "calibration": "sigmoid; 5-fold stratified CV on training split only",
        "threshold_selection": "maximum validation F1; test labels not used",
        "selected_threshold": selected_threshold,
        "validation_metrics_at_selected_threshold": classification_report(y_val, val_probabilities, selected_threshold),
        "final_test_metrics": classification_report(y_test, test_probabilities, selected_threshold),
        "split_sizes": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "target_semantics_note": "Metrics describe the dataset's loan_status label only. They do not establish prediction of future repayment or default. Verify source label provenance.",
    }
    (report_dir / "final_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (model_dir / "model_metadata.json").write_text(json.dumps({
        "model_file": model_path.name, "selected_model": selected_name,
        "threshold": selected_threshold, "positive_class": 1, "target": "loan_status"
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved calibrated model: {model_path}")
    print("Important: test results are final holdout results; do not tune further against them.")

if __name__ == "__main__":
    main()
