"""Generate global and local SHAP explanations for the calibrated XGBoost ensemble."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
from ml.preprocessing.pipeline import add_features

ROOT = Path(__file__).resolve().parents[1]

def main():
    model_path = ROOT / "models/final_calibrated_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run scripts/06_calibration.py first.")
    model = joblib.load(model_path)
    validation = pd.read_csv(ROOT / "data/processed/val.csv")
    sample = validation.drop(columns="loan_status").sample(n=min(300, len(validation)), random_state=42)
    X = add_features(sample)
    fold_importances, fold_values = [], []
    feature_names = None
    for calibrated_fold in model.calibrated_classifiers_:
        pipeline = calibrated_fold.estimator
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()
        values = shap.TreeExplainer(estimator).shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        fold_values.append(values)
        fold_importances.append(np.abs(values).mean(axis=0))
    importance = np.mean(fold_importances, axis=0)
    order = np.argsort(importance)[::-1]
    global_rows = [{"feature": str(feature_names[i]), "mean_abs_shap": float(importance[i]), "rank": int(rank + 1)} for rank, i in enumerate(order)]
    local_values = np.mean(np.stack(fold_values, axis=0), axis=0)
    probabilities = model.predict_proba(sample)[:, 1]
    local_rows = []
    for row_index in range(min(5, len(sample))):
        top = np.argsort(np.abs(local_values[row_index]))[::-1][:10]
        local_rows.append({
            "sample_index": int(row_index),
            "calibrated_class_1_probability": float(probabilities[row_index]),
            "top_features": [{"feature": str(feature_names[i]), "mean_signed_shap": float(local_values[row_index, i])} for i in top],
        })
    output = ROOT / "reports/models"
    output.mkdir(parents=True, exist_ok=True)
    (output / "shap_global.json").write_text(json.dumps({
        "explanation_scope": "Mean absolute SHAP for underlying XGBoost fold estimators; not calibrated probability or causality.",
        "sample_size": int(len(sample)), "ensemble_folds": int(len(fold_values)), "features": global_rows,
    }, indent=2), encoding="utf-8")
    (output / "shap_local_examples.json").write_text(json.dumps({
        "explanation_scope": "Mean signed SHAP across fold estimators; descriptive local feature contributions, not causal effects.",
        "examples": local_rows,
    }, indent=2), encoding="utf-8")
    print(f"Saved SHAP reports for {len(sample)} validation examples and {len(fold_values)} folds.")

if __name__ == "__main__":
    main()
