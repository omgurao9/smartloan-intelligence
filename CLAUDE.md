# SmartLoan Intelligence — Claude Working Rules

## Project
SmartLoan Intelligence: An Explainable and Responsible AI Platform for Loan Risk Assessment.

## Environment and constraints
- Windows 11, 16 GB RAM; prefer CPU-friendly solutions.
- Python, FastAPI, React + Vite, Tailwind CSS.
- Local laptop demo only.
- Zero budget: no paid APIs, subscriptions, or hosted services.
- Repository is public. Never commit applicant-level data, datasets, secrets, local databases, or trained model artifacts.

## Priority
Build a reliable, academically defensible end-to-end project. Prioritize working core features over optional polish.

## Required workflow for every task
1. Start by creating the most important usable deliverable. Do not spend the response on a long plan.
2. Keep each task tightly scoped; normally touch no more than 1–3 related files.
3. Provide complete files, not fragments, and state exact paths.
4. Do not overwrite accepted work without a clear reason.
5. Do not add unnecessary abstractions, dependencies, or features.
6. Never leave TODOs, pass statements, or fake implementations in core functionality.
7. Never claim a command or test passed unless it was actually run; distinguish suggested commands from executed tests.
8. If usage/context is getting low, stop optional work immediately and provide completed files, remaining work, known issues, and the exact next step.
9. Update HANDOFF.md after each task with what changed and the next action.
10. Do not start a future phase without approval.

## ML and data integrity
- Inspect the actual dataset schema and target-label meaning before implementation.
- Preserve the original dataset; investigate unusual records before excluding them.
- Use reproducible seeds and stratified splits.
- Fit all preprocessing only on training data.
- Prevent target leakage and keep the final test set untouched during model selection.
- Report metrics beyond accuracy, including class-sensitive metrics and ROC-AUC/PR-AUC where appropriate.
- Check calibration and threshold trade-offs where practical.
- Treat SHAP as model explanation, not causal explanation.
- Counterfactuals are hypothetical, not guaranteed approval advice. Do not suggest changing immutable/sensitive personal attributes.
- Audit group metrics carefully and explain limitations; fairness metrics do not prove a model is fair.
- Calculate loan_percent_income internally from loan amount and income; do not ask an applicant to enter it separately.

## Communication
Be concise and output-first. For each task, state:
- files created/changed;
- dependencies;
- exact run/test commands;
- actual test results (or clearly say not run);
- known limitations;
- exact next step.
