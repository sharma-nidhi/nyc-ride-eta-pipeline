# Plan of Action — NYC Ride ETA Pipeline (28 days)

Flavor A. Maps to modules M2→M3→M4→M5. Each week ends with a tagged milestone commit.
Rubric weight is split evenly (20% each) across M2, M3, M4, M5, and Documentation.

## Phase 0 — Setup & connections (Day 1–2)
- [ ] Confirm Python 3.10+, create venv, `pip install -r requirements.txt`.
- [ ] Connect Git remote, merge existing README, push scaffold (`docs/SETUP.md`).
- [ ] Kaggle token + accept competition rules + `python data/download_data.py`.
- [ ] `dvc init`; agree branch/commit conventions with teammate.
- **Milestone:** repo initialized, dataset downloaded.

## Week 1 — M2: Data engineering & versioning (Day 3–7)
- [ ] `data/` ingestion: load raw CSV, parse pickup/dropoff datetimes, log summary.
- [ ] `data/validate.py`: column contract; `trip_duration > 0` + upper bound; NYC lat/lon
      bounds; passenger_count range; dropoff > pickup; nulls/dups; fail loudly.
- [ ] `features/build_features.py` (shared): temporal (hour, weekday, weekend, season),
      haversine distance, optional zone/borough, optional weather; `trip_duration -> log1p`.
- [ ] Feature store + `feature_schema.json` contract.
- [ ] DVC add + remote + `dvc push`; `git tag data-v1`.
- **Milestone (End W1):** validation + feature pipeline complete; dataset version tagged.

## Week 2 — M3: Experimentation & reproducibility (Day 8–14)
- [ ] MLflow experiment `nyc_ride_eta`; log params/metrics/model/schema per run.
- [ ] Baseline: LinearRegression (reference RMSLE/MAE).
- [ ] XGBoost/LightGBM + ≥2–3 runs varying n_estimators / max_depth / learning_rate.
- [ ] Compare in MLflow UI; pick best with written justification; prove reproducibility
      (re-run logged config → same metrics). Export comparison report + screenshots.
- **Milestone (End W2):** ≥2 tracked experiments; best model chosen + justified.

## Week 3 — M4: Packaging & deployment (Day 15–21)
- [ ] Serialize best model + freeze schema into `models/` (tag `eta-v1`).
- [ ] `serving/api.py` + `schemas.py`: Pydantic validation; load model at startup;
      reuse shared feature logic; `/health` + `/predict` returning ETA + model_version.
- [ ] Handle malformed input (422); note basic latency/throughput.
- [ ] Dockerfile: build + run; test with curl + Postman collection.
- **Milestone (End W3):** model deployed; endpoint tested with sample inputs.

## Week 4 — M5: Monitoring, drift & retraining (Day 22–28)
- [ ] `monitoring/logger.py`: log every prediction (inputs, ETA, version) to SQLite.
- [ ] Drift simulation: rush-hour / festival surge / longer-distance batches.
- [ ] `monitoring/check_drift.py`: mean-shift (σ) + PSI/KS; error tracking if actuals;
      report + plots.
- [ ] Retraining trigger design: thresholds, data window, validation gate, promotion/rollback.
- **Milestone (Day 28):** monitoring + drift complete; submission package + demo ready.

## Submission checklist (rubric)
1. [ ] Versioned dataset + pipeline code (repo link, weekly commit history).
2. [ ] Experiment logs + model comparison report (MLflow screenshots/exports).
3. [ ] Deployed API + sample request/response (curl / Postman).
4. [ ] Monitoring log + drift report + documented retraining trigger.
5. [ ] README with architecture diagram + setup + 5–7 min demo/presentation.

## Suggested split (2 people)
- **Person A:** M2 data/validation/features + DVC; M4 Docker.
- **Person B:** M3 training/MLflow; M4 FastAPI; M5 monitoring.
- Both: README/docs, demo, code review of each other's PRs.

## Key design decisions to justify in docs (rubric requires this)
- Why the chosen model (metrics vs latency/complexity).
- Feature choices (why haversine, why log-target for RMSLE).
- Drift-detection method (σ shift vs PSI/KS) and thresholds.
- Retraining trigger (what fires it, and the validation/promotion gate).
