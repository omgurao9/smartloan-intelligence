"""Validation-only ablation of the previous-default-history feature."""
from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from ml.preprocessing.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
FEATURE = "previous_loan_defaults_on_file"

def evaluate(estimator, drop_feature):
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    validation = pd.read_csv(ROOT / "data/processed/val.csv")
    X_train, y_train = train.drop(columns="loan_status"), train["loan_status"]
    X_val, y_val = validation.drop(columns="loan_status"), validation["loan_status"]
    pipeline = make_pipeline(estimator, drop_features=[FEATURE] if drop_feature else None)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_val)[:, 1]
    return {"roc_auc": float(roc_auc_score(y_val, probabilities)), "pr_auc": float(average_precision_score(y_val, probabilities))}

def main():
    estimator = RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    full = evaluate(estimator, False)
    without = evaluate(estimator, True)
    rows = [{
        "model": "random_forest",
        "roc_auc_with_feature": full["roc_auc"],
        "roc_auc_without_feature": without["roc_auc"],
        "roc_auc_change": without["roc_auc"] - full["roc_auc"],
        "pr_auc_with_feature": full["pr_auc"],
        "pr_auc_without_feature": without["pr_auc"],
        "pr_auc_change": without["pr_auc"] - full["pr_auc"],
    }]
    output = ROOT / "reports/models"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "previous_default_ablation.csv", index=False)
    (output / "previous_default_ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
