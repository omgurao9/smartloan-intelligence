import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

TARGET = "loan_status"
CATEGORICAL = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
NUMERIC = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "person_income_log"]

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["person_income_log"] = np.log1p(np.clip(x["person_income"].astype(float), 0, None))
    return x

def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x = x.loc[(x["person_age"] < 84) & (x["person_emp_exp"] <= x["person_age"] - 18)].copy()
    return x.reset_index(drop=True)

def make_preprocessor(scale_numeric=True, drop_features=None):
    dropped = set(drop_features or [])
    numeric_columns = [column for column in NUMERIC if column not in dropped]
    categorical_columns = [column for column in CATEGORICAL if column not in dropped]
    numeric = Pipeline([("scaler", StandardScaler())]) if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_columns),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def make_pipeline(estimator, drop_features=None):
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", make_preprocessor(True, drop_features)),
        ("model", estimator),
    ])
