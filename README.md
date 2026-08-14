# NYC Ride ETA Prediction — End-to-End ML Pipeline

**Course:** Machine Learning Engineering (PCAM ZC412) · Mini-Project (EC-1, 40%)
**Flavor A:** Delivery / Ride ETA Prediction (Tabular · Modules M2 → M3 → M4 → M5)
**Repo:** https://github.com/sharma-nidhi/nyc-ride-eta-pipeline

Predict NYC taxi trip duration (ride ETA) from trip distance, time-of-day, location, and
weather. This repo is a production-style ML system: reliable data ingestion + validation,
tracked experiments, a packaged REST API, and monitoring with drift detection and a
retraining strategy.

---

## Architecture (target)

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                    TRAINING PATH                          │
  Kaggle NYC Taxi   │  data/download → data/validate → features/build_features │
  Trip Duration ───▶│      (raw)          (schema)          (shared logic)      │
                    │                          │                                │
                    │                          ▼                                │
                    │              feature store + feature_schema.json          │
                    │                          │                                │
                    │        training/train.py  ──▶  MLflow (params/metrics)    │
                    │                          │                                │
                    │                          ▼                                │
                    │               models/eta-v1.joblib (best)                 │
                    └──────────────────────────┬───────────────────────────────┘
                                               │  (same feature logic)
                    ┌──────────────────────────▼───────────────────────────────┐
                    │                    SERVING PATH                           │
   Client ────────▶ │  serving/api.py (FastAPI)  /health   /predict            │
   (trip details)   │     Pydantic validation ──▶ predict ETA ──▶ response      │
                    │                          │                                │
                    │                          ▼                                │
                    │        monitoring/logger.py  →  predictions.db            │
                    │        monitoring/check_drift.py → drift signals          │
                    └──────────────────────────────────────────────────────────┘
```

DVC versions the dataset; Git versions the code; MLflow versions the experiments;
Docker packages the service.

---

## Repository layout

```
nyc-ride-eta-pipeline/
├── config/            # config.yaml — paths, params, thresholds
├── data/
│   ├── raw/           # Kaggle CSVs (DVC-tracked, git-ignored)
│   ├── processed/     # cleaned/engineered data (git-ignored)
│   ├── download_data.py   # pull dataset from Kaggle
│   └── validate.py        # schema + data-quality checks
├── features/
│   └── build_features.py  # SHARED feature logic (train == serve)
├── training/
│   └── train.py           # MLflow-tracked training + model comparison
├── serving/
│   ├── api.py             # FastAPI inference service
│   └── schemas.py         # Pydantic request/response contracts
├── monitoring/
│   ├── logger.py          # log every prediction to SQLite
│   └── check_drift.py     # distribution drift + monitoring signals
├── models/                # serialized model + schema (git-ignored/DVC)
├── tests/                 # pytest (feature pipeline, API)
├── notebooks/             # EDA
├── docs/                  # PLAN.md, SETUP.md, architecture.md
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

> **Status:** scaffold + plan. Module files are placeholders with TODOs — implemented
> week by week per `docs/PLAN.md`.

---

## Quickstart

See **`docs/SETUP.md`** for full setup (Python env, Git remote, Kaggle, DVC).

```bash
# 1. environment
python -m venv venv
venv\Scripts\activate           # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. data  (needs Kaggle token — see docs/SETUP.md)
python data/download_data.py
python data/validate.py

# 3. features + train (Week 1–2)
python features/build_features.py
python training/train.py
mlflow ui                        # http://127.0.0.1:5000

# 4. serve (Week 3)
uvicorn serving.api:app --reload --port 8000   # http://127.0.0.1:8000/docs

# 5. monitor (Week 4)
python monitoring/check_drift.py
```

---

## Weekly milestones

| Week | Module | Deliverable |
|------|--------|-------------|
| 1 | M2 | Ingestion, validation, feature pipeline; dataset versioned (DVC tag `data-v1`) |
| 2 | M3 | ≥2 tracked experiments; best model selected + justified |
| 3 | M4 | Model packaged; FastAPI endpoint tested with sample inputs |
| 4 | M5 | Prediction logging, drift simulation, monitoring + retraining trigger |

## Team

| Name | Role | Owner of |
|------|------|----------|
| Ronak | | |
| Nidhi Sharma | Repo owner | |

## References
- T1: *Machine Learning Production Systems*, Robert Crowe et al., O'Reilly, 2024.
- T2: *Machine Learning Engineering*, Andriy Burkov, 2020.
- R1: *Machine Learning Engineering with Python* (2e), A. P. McMahon, Packt, 2023.
- Dataset: NYC Taxi Trip Duration — Kaggle (cite competition page).
