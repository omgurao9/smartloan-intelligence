from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/loan_data_clean.csv"
OUT = ROOT / "data/processed"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    train, temp = train_test_split(df, test_size=0.30, stratify=df["loan_status"], random_state=42)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["loan_status"], random_state=42)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows; class counts {part['loan_status'].value_counts().to_dict()}")
    rng = pd.Series(df["loan_status"].to_numpy()).sample(frac=1, random_state=42).to_numpy()
    print(f"Shuffled-label sanity AUC (should be near 0.5): {roc_auc_score(df['loan_status'], rng):.3f}")

if __name__ == "__main__":
    main()
