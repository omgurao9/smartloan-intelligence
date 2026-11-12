import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const defaults = {
  person_age: 32, person_gender: "male", person_education: "Bachelor",
  person_income: 60000, person_emp_exp: 8, person_home_ownership: "RENT",
  loan_amnt: 10000, loan_intent: "PERSONAL", loan_int_rate: 11,
  cb_person_cred_hist_length: 6, credit_score: 680,
  previous_loan_defaults_on_file: "No",
};

const numericFields = [
  ["person_age", "Age", 18, 83],
  ["person_income", "Annual income", 1, 10000000],
  ["person_emp_exp", "Employment experience (years)", 0, 65],
  ["loan_amnt", "Loan amount", 1, 1000000],
  ["loan_int_rate", "Interest rate (%)", 0.01, 100],
  ["cb_person_cred_hist_length", "Credit history length (years)", 0, 70],
  ["credit_score", "Credit score", 300, 900],
];

const selectFields = [
  ["person_gender", "Gender", ["male", "female"]],
  ["person_education", "Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"]],
  ["person_home_ownership", "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]],
  ["loan_intent", "Loan intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]],
  ["previous_loan_defaults_on_file", "Previous default on file", ["No", "Yes"]],
];

function App() {
  const [form, setForm] = useState(defaults);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(payload.detail)
          ? payload.detail.map((item) => item.msg).join("; ")
          : payload.detail || "Prediction request failed.";
        throw new Error(detail);
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "API unavailable. Start FastAPI and try again.");
    } finally {
      setBusy(false);
    }
  }

  const ratio = form.person_income > 0 ? form.loan_amnt / form.person_income : 0;

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">SL<small>SMARTLOAN INTELLIGENCE</small></div>
        <div className="nav"><button className="active">Assessment</button><button disabled>Model Lab · coming next</button><button disabled>Explainability · coming next</button></div>
        <div className="rail-note">LOCAL ML WORKSPACE<br />EDUCATIONAL DEMO</div>
      </aside>
      <main className="content">
        <header className="top">
          <div><div className="eyebrow">CREDIT INTELLIGENCE / MODEL WORKSPACE</div><h1>Loan status assessment</h1><p className="sub">Explore how the trained classifier responds to applicant and loan attributes.</p></div>
          <div className="status"><span /> LOCAL ENVIRONMENT</div>
        </header>
        <div className="grid">
          <section className="panel">
            <div className="section-head"><div><div className="eyebrow">INPUT DATA</div><h2>Applicant profile</h2></div><span className="step">01 / PROFILE</span></div>
            <form className="form" onSubmit={submit}>
              {numericFields.map(([key, label, min, max]) => (
                <div className="field" key={key}>
                  <label htmlFor={key}>{label}</label>
                  <input id={key} type="number" min={min} max={max} step="any" required value={form[key]} onChange={(event) => update(key, Number(event.target.value))} />
                </div>
              ))}
              {selectFields.map(([key, label, options]) => (
                <div className="field" key={key}>
                  <label htmlFor={key}>{label}</label>
                  <select id={key} value={form[key]} onChange={(event) => update(key, event.target.value)}>
                    {options.map((option) => <option key={option} value={option}>{option}</option>)}
                  </select>
                </div>
              ))}
              <div className="derived"><span>Derived loan / income ratio</span><strong>{(ratio * 100).toFixed(1)}%</strong><small>Calculated automatically. It is not submitted as a separate applicant input.</small></div>
              <button className="button" disabled={busy}>{busy ? "Evaluating model…" : "Run assessment"}</button>
            </form>
          </section>
          <section className="panel result-panel">
            <div className="section-head"><div><div className="eyebrow">MODEL OUTPUT</div><h2>Assessment result</h2></div><span className="step">02 / RESULT</span></div>
            {!result && !error && <div className="empty"><div className="empty-mark">SL</div><h3>Awaiting applicant data</h3><p>Complete the profile and run the model to see its estimate for the dataset label.</p></div>}
            {error && <div className="error"><strong>Assessment unavailable</strong><p>{error}</p></div>}
            {result && <div className="result">
              <div className="eyebrow">PREDICTED DATASET LABEL</div>
              <div className="decision">{result.decision.replace("_", " ")}</div>
              <div className="prob">{(result.class_1_probability * 100).toFixed(1)}<span>%</span></div>
              <p className="sub">Estimated probability of class 1 in this dataset.</p>
              <div className="meter"><span style={{ width: `${result.approval_probability * 100}%` }} /></div>
              <div className="threshold"><span>Classification threshold</span><strong>{(result.threshold * 100).toFixed(0)}%</strong></div>
              <div className="notice">{result.disclaimer}</div>
            </div>}
          </section>
        </div>
        <footer className="disclaimer">RESEARCH PROTOTYPE · This model estimates the dataset's loan_status label. It is not a credit decision, lending recommendation, or verified future-default prediction.</footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
