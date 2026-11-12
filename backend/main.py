"""Local FastAPI service for the SmartLoan educational classifier."""
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sqlite3
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/final_calibrated_model.joblib"
METADATA_PATH = ROOT / "models/model_metadata.json"
DB_PATH = ROOT / "database/smartloan.db"
_loaded_model = None
_model_metadata = None

class Applicant(BaseModel):
    """Applicant inputs; loan-to-income ratio is derived by the API."""
    model_config = ConfigDict(str_strip_whitespace=True)
    person_age: float = Field(gt=17, lt=84)
    person_gender: str
    person_education: str
    person_income: float = Field(gt=0)
    person_emp_exp: int = Field(ge=0)
    person_home_ownership: str
    loan_amnt: float = Field(gt=0)
    loan_intent: str
    loan_int_rate: float = Field(gt=0)
    cb_person_cred_hist_length: float = Field(ge=0)
    credit_score: int = Field(ge=300, le=900)
    previous_loan_defaults_on_file: str

    @model_validator(mode="after")
    def validate_experience(self):
        if self.person_emp_exp > self.person_age - 18:
            raise ValueError("Employment experience is inconsistent with applicant age.")
        return self

    def model_features(self):
        values = self.model_dump()
        values["loan_percent_income"] = self.loan_amnt / self.person_income
        return values

class WhatIfScenario(BaseModel):
    """Hypothetical changes to limited financial/loan terms only."""
    person_income: float | None = Field(default=None, gt=0)
    loan_amnt: float | None = Field(default=None, gt=0)
    loan_int_rate: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_change(self):
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("Provide at least one supported what-if change.")
        return self

class WhatIfRequest(BaseModel):
    applicant: Applicant
    scenarios: list[WhatIfScenario] = Field(min_length=1, max_length=5)

def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        approval_probability REAL NOT NULL, decision TEXT NOT NULL, payload TEXT NOT NULL
    )""")
    connection.commit()
    return connection

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _loaded_model, _model_metadata
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        _loaded_model = joblib.load(MODEL_PATH)
        _model_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    get_db().close()
    yield
    _loaded_model = None
    _model_metadata = None

app = FastAPI(
    title="SmartLoan Intelligence API", version="1.2.0",
    description="Educational classification of the dataset's loan_status label; not a future-default or real lending decision service.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)

@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": _loaded_model is not None}

@app.post("/api/predict")
def predict(applicant: Applicant):
    if _loaded_model is None or _model_metadata is None:
        raise HTTPException(503, "Model artifact unavailable. Run training and calibration first.")
    features = applicant.model_features()
    probability = float(_loaded_model.predict_proba(pd.DataFrame([features]))[0, 1])
    threshold = float(_model_metadata["threshold"])
    decision = "APPROVED" if probability >= threshold else "NOT_APPROVED"
    record = applicant.model_dump()
    record["loan_percent_income"] = features["loan_percent_income"]
    connection = get_db()
    try:
        connection.execute(
            "INSERT INTO predictions(approval_probability, decision, payload) VALUES (?, ?, ?)",
            (probability, decision, json.dumps(record)),
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "approval_probability": probability, "threshold": threshold,
        "decision": decision, "target_label": "loan_status",
        "disclaimer": "Educational estimate of the dataset label; not a real lending decision or future-default prediction.",
    }

@app.post("/api/what-if")
def what_if(request: WhatIfRequest):
    if _loaded_model is None or _model_metadata is None:
        raise HTTPException(503, "Model artifact unavailable. Run training and calibration first.")
    baseline_features = request.applicant.model_features()
    baseline_probability = float(_loaded_model.predict_proba(pd.DataFrame([baseline_features]))[0, 1])
    results = []
    for index, scenario in enumerate(request.scenarios, start=1):
        features = dict(baseline_features)
        changes = {key: value for key, value in scenario.model_dump().items() if value is not None}
        features.update(changes)
        features["loan_percent_income"] = features["loan_amnt"] / features["person_income"]
        probability = float(_loaded_model.predict_proba(pd.DataFrame([features]))[0, 1])
        results.append({
            "scenario": index, "changes": changes,
            "derived_loan_percent_income": features["loan_percent_income"],
            "class_1_probability": probability,
            "difference_from_baseline": probability - baseline_probability,
        })
    return {
        "baseline_class_1_probability": baseline_probability,
        "threshold": float(_model_metadata["threshold"]),
        "scenarios": results,
        "disclaimer": "Hypothetical model sensitivity only. These changes are not causal recommendations, do not guarantee approval, and do not represent a real lending decision.",
    }

@app.get("/api/model-performance")
def model_performance():
    path = ROOT / "reports/models/baseline_comparison.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

@app.get("/api/final-evaluation")
def final_evaluation():
    path = ROOT / "reports/models/final_evaluation.json"
    if not path.exists():
        raise HTTPException(404, "Final evaluation report not found.")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/explainability/global")
def explainability_global():
    path = ROOT / "reports/models/shap_global.json"
    if not path.exists():
        raise HTTPException(404, "Run scripts/07_shap.py to generate SHAP explanations.")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/fairness")
def fairness_report():
    path = ROOT / "reports/fairness/subgroup_metrics.json"
    if not path.exists():
        raise HTTPException(404, "Run scripts/08_fairness.py to generate the subgroup audit.")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/predictions")
def predictions():
    connection = get_db()
    try:
        rows = connection.execute(
            "SELECT id, created_at, approval_probability, decision FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
    finally:
        connection.close()
    return [{"id": r[0], "created_at": r[1], "approval_probability": r[2], "decision": r[3]} for r in rows]
