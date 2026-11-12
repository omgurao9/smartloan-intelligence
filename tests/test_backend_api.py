from fastapi.testclient import TestClient
from backend import main

def test_health_reports_trained_model():
    with TestClient(main.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_available"] is True

def test_prediction_derives_income_ratio_and_persists(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "test.db")
    applicant = {
        "person_age": 32, "person_gender": "male", "person_education": "Bachelor",
        "person_income": 60000, "person_emp_exp": 8, "person_home_ownership": "RENT",
        "loan_amnt": 10000, "loan_intent": "PERSONAL", "loan_int_rate": 11,
        "cb_person_cred_hist_length": 6, "credit_score": 680,
        "previous_loan_defaults_on_file": "No",
    }
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json=applicant)
        history = client.get("/api/predictions")
        expected_threshold = main._model_metadata["threshold"]
    assert response.status_code == 200, response.text
    result = response.json()
    assert 0 <= result["approval_probability"] <= 1
    assert result["threshold"] == expected_threshold
    assert "loan_percent_income" not in applicant
    assert history.status_code == 200
    assert len(history.json()) == 1
