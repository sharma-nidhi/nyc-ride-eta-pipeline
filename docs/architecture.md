# Architecture

See the diagram in the root `README.md`. Replace it with an exported image
(draw.io / Excalidraw) before final submission.

## Components & tooling
| Layer | Component | Tool |
|-------|-----------|------|
| Data source | Kaggle NYC Taxi Trip Duration | kaggle CLI |
| Ingestion | `data/` load + parse | pandas |
| Validation | `data/validate.py` | pandas (fail-loud) |
| Features (shared) | `features/build_features.py` | pandas / numpy |
| Feature store | `data/processed/feature_store.db` | SQLite / parquet |
| Data versioning | raw + processed | DVC |
| Experiments | `training/train.py` | scikit-learn, XGBoost, MLflow |
| Model artifact | `models/eta-v1.joblib` + schema | joblib |
| Serving | `serving/api.py` | FastAPI + Pydantic + uvicorn |
| Packaging | container | Docker |
| Monitoring | `monitoring/` | SQLite + custom drift checks |

## Contracts (prevent skew / breakage)
- **feature_schema.json** — exact feature columns/order; written at training, read at serving.
- **Pydantic models** (`serving/schemas.py`) — validated request/response for the API.
- **DVC + Git tags** — dataset (`data-v1`) and model (`eta-v1`) versions are reproducible.

## Known limitations (state honestly in README governance section)
- SQLite feature/prediction store — fine for the project, not high-concurrency production.
- Concept drift needs ground-truth trip_duration to measure error directly.
- Single-process serving — no HA/autoscaling.
