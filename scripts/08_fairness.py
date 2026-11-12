"""Audit subgroup performance on the held-out test split at the frozen threshold."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MIN_GROUP_SIZE = 50

def group_metrics(groups, y_true, probabilities, threshold):
    prediction = (probabilities >= threshold).astype(int)
    rows = []
    for group_value, indices in groups.groupby(groups, dropna=False, observed=True).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        actual = np.asarray(y_true)[idx]
        predicted = prediction[idx]
        if len(idx) < MIN_GROUP_SIZE:
            rows.append({"group": str(group_value), "n": int(len(idx)), "suppressed": True})
            continue
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        rows.append({
            "group": str(group_value), "n": int(len(idx)), "suppressed": False,
            "selection_rate": float(predicted.mean()),
            "true_positive_rate": float(tp / (tp + fn)) if tp + fn else None,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "accuracy": float((tp + tn) / len(idx)),
        })
    return rows

def main():
    model = joblib.load(ROOT / "models/final_calibrated_model.joblib")
    metadata = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    X, y = test.drop(columns="loan_status"), test["loan_status"].to_numpy()
    probabilities = model.predict_proba(X)[:, 1]
    age_band = pd.cut(test["person_age"], bins=[17, 25, 35, 45, 60, 84], labels=["18-25", "26-35", "36-45", "46-60", "61-83"])
    report = {
        "threshold": float(metadata["threshold"]),
        "evaluation_split": "held-out test; descriptive audit only",
        "minimum_group_size": MIN_GROUP_SIZE,
        "groups": {
            "person_gender": group_metrics(test["person_gender"], y, probabilities, metadata["threshold"]),
            "person_education": group_metrics(test["person_education"], y, probabilities, metadata["threshold"]),
            "age_band": group_metrics(age_band, y, probabilities, metadata["threshold"]),
        },
        "limitations": [
            "Group metrics describe this dataset and sample only; they do not prove legal or ethical fairness.",
            "Small groups are suppressed below the configured minimum count.",
            "The dataset label is not established as future repayment or default.",
            "The threshold was selected before test evaluation; do not tune it using these subgroup results."
        ],
    }
    output = ROOT / "reports/fairness"
    output.mkdir(parents=True, exist_ok=True)
    (output / "subgroup_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved subgroup audit: {output / 'subgroup_metrics.json'}")

if __name__ == "__main__":
    main()
