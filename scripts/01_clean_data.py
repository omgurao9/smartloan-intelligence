from pathlib import Path
import pandas as pd
from ml.preprocessing.pipeline import clean_dataframe

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/loan_data.csv"
OUT = ROOT / "data/processed/loan_data_clean.csv"
REPORT = ROOT / "reports/data_cleaning_log.md"

def main():
    df = pd.read_csv(RAW)
    before = len(df)
    invalid = (df["person_age"] >= 84) | (df["person_emp_exp"] > df["person_age"] - 18)
    removed = int(invalid.sum())
    cleaned = clean_dataframe(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT, index=False)
    REPORT.write_text(
        f"# Data Cleaning Log\n\n- Raw rows: {before}\n- Removed invalid age/experience rows: {removed}\n"
        f"- Clean rows: {len(cleaned)}\n- Missing values: {int(cleaned.isna().sum().sum())}\n"
        f"- Duplicate rows: {int(cleaned.duplicated().sum())}\n\n"
        "The rule excludes records with age >= 84 or employment experience inconsistent with an assumed working age of 18. "
        "This is a project cleaning assumption and its limitations must be disclosed. The raw file is unchanged.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(cleaned)} rows to {OUT}")

if __name__ == "__main__":
    main()
