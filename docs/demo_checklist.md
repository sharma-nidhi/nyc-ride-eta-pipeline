# Demo Checklist (5–7 minutes)

Goal: show the system end to end, mapping each part to a graded module. Have two terminals ready
(both in the repo root, venv active) and the browser open.

## Before you start
- [ ] `venv` activated in both terminals.
- [ ] `data/processed/features.parquet`, `models/eta-v1.joblib`, `models/feature_schema.json` exist.
- [ ] MLflow screenshot saved at `docs/images/mlflow_compare.png`.
- [ ] Docker Desktop running (if demoing the container).

## Flow

**0:00 — Intro (30s).** "Flavor A: predict NYC taxi ETA. End-to-end pipeline: data → model →
API → monitoring." Show the README architecture diagram.

**0:30 — M2 Data (1 min).**
- `python data/validate.py` → point out the quality report (99.45% kept; 7,970 junk rows dropped).
- Mention DVC: `git tag` shows `data-v1` (dataset is versioned).

**1:30 — M3 Experiments (1.5 min).**
- Open MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) → the 3 runs.
- Compare view: baseline RMSLE 0.543 → best 0.397. State the choice + justification
  (gradient boosting wins; the two GBMs tie, so the lighter one is a valid production pick).
- Mention reproducibility: fixed seed → identical metrics on re-run.

**3:00 — M4 Serving (1.5 min).**
- API already running. Open `/docs`, run a valid `/predict` → ETA returned.
- Run `python serving/sample_requests.py` → show valid 200 + invalid **422** (validation).
- Show it runs in Docker: `docker run --rm -p 8000:8000 nyc-eta-api`.

**4:30 — M5 Monitoring (1.5 min).**
- `python monitoring/simulate_drift.py` → normal + surge batches.
- `python monitoring/check_drift.py` → **DRIFT DETECTED** table (σ-shift + PSI).
- Point to `docs/retraining_design.md`: the trigger rule + validation gate + rollback.

**6:00 — Wrap (30s).**
- `pytest -q` → tests pass.
- Show the commit history (weekly, incremental). "Everything versioned: DVC / Git / MLflow / Docker."

## Talking points if asked
- **Why log-target?** RMSLE is the metric; log space makes RMSE == RMSLE.
- **How is skew prevented?** Serving imports the same `build_features()` as training.
- **What fires a retrain?** PSI > 0.2 or σ-shift > 0.8 sustained, or rolling RMSLE > 0.44, or the 4-week schedule.
