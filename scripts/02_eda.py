from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/loan_data_clean.csv"
OUT = ROOT / "reports/eda"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    findings = {
        "rows": len(df), "columns": len(df.columns),
        "target_distribution": {str(k): int(v) for k, v in df["loan_status"].value_counts().items()},
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "approval_by_previous_default": {str(k): float(v) for k, v in df.groupby("previous_loan_defaults_on_file")["loan_status"].mean().items()},
        "numeric_correlations": {str(k): float(v) for k, v in df.select_dtypes("number").corr()["loan_status"].sort_values(ascending=False).items()},
        "interpretation_warning": "Verify target label provenance; this dataset label is not demonstrated to be future repayment/default."
    }
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    plt.figure(figsize=(7, 5))
    df["loan_status"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("loan_status"); plt.ylabel("Records"); plt.title("Target Label Distribution")
    plt.tight_layout(); plt.savefig(OUT / "target_distribution.png", dpi=150); plt.close()
    rates = df.groupby("previous_loan_defaults_on_file")["loan_status"].mean()
    plt.figure(figsize=(7, 5)); rates.plot(kind="bar")
    plt.ylabel("Mean loan_status"); plt.title("Target Rate by Previous Default History")
    plt.tight_layout(); plt.savefig(OUT / "target_by_default_history.png", dpi=150); plt.close()
    numeric = df.select_dtypes("number")
    plt.figure(figsize=(10, 7)); sns.heatmap(numeric.corr(), cmap="vlag", center=0)
    plt.title("Numeric Correlation Heatmap"); plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=150); plt.close()
    print(f"EDA complete: {OUT}")

if __name__ == "__main__":
    main()
