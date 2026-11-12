"""Train baseline classifiers and compare them on the validation split.

The test split is intentionally not loaded or evaluated in this script.
"""
from pathlib import Path
import warnings
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ml.preprocessing.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports/models"
RANDOM_STATE = 42

def evaluate(model, features, target):
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def build_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
        "svm": SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
        "naive_bayes": GaussianNB(),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
    except ImportError:
        warnings.warn("xgboost is not installed; skipping the XGBoost baseline.")
    return models

def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    val = pd.read_csv(ROOT / "data/processed/val.csv")
    if "loan_status" not in train or "loan_status" not in val:
        raise ValueError("Both train.csv and val.csv must contain loan_status.")
    X_train, y_train = train.drop(columns="loan_status"), train["loan_status"]
    X_val, y_val = val.drop(columns="loan_status"), val["loan_status"]
    results = []
    for name, estimator in build_models().items():
        X_fit, y_fit = X_train, y_train
        if name == "svm" and len(X_train) > 12000:
            X_fit, _, y_fit, _ = train_test_split(X_train, y_train, train_size=12000, stratify=y_train, random_state=RANDOM_STATE)
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_fit, y_fit)
        row = evaluate(pipeline, X_val, y_val)
        row["model"] = name
        row["training_rows"] = int(len(X_fit))
        results.append(row)
        joblib.dump(pipeline, MODELS / f"{name}_baseline.joblib")
        print(f"{name}: PR-AUC={row['pr_auc']:.4f}, ROC-AUC={row['roc_auc']:.4f}, F1={row['f1']:.4f}", flush=True)
    comparison = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    comparison.to_csv(REPORTS / "baseline_comparison.csv", index=False)
    print(f"\nSaved validation comparison to {REPORTS / 'baseline_comparison.csv'}")
    print("Test data was not accessed by this script.")

if __name__ == "__main__":
    main()
