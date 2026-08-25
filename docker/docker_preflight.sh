#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Verifying champion serving artifact exists..."
if [[ ! -f "models/serving/model.pkl" ]]; then
  echo "ERROR: models/serving/model.pkl not found."
  echo "Run: python -m src.models.registry"
  exit 1
fi

echo "[2/5] Validating docker compose configuration..."
docker compose -f docker/docker-compose.yml config >/dev/null

echo "[3/5] Verifying Python dependency consistency..."
if command -v python3 &>/dev/null; then
  python3 -m pip check >/dev/null || echo "Warning: pip check found issues (non-fatal)"
else
  echo "Skipping: python3 not found."
fi

echo "[4/5] Verifying critical imports in local environment..."
if command -v python3 &>/dev/null; then
  python3 - <<'PY' || echo "Warning: local import check skipped (non-fatal)"
import importlib
modules = ["fastapi", "uvicorn", "joblib", "lightgbm", "xgboost", "catboost", "mlflow"]
for m in modules:
    importlib.import_module(m)
print("Local import check passed")
PY
else
  echo "Skipping: python3 not found."
fi

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[5/5] Running container smoke test for native ML libs..."
  docker build -f docker/Dockerfile.api -t eta-api:preflight . >/dev/null
  docker run --rm eta-api:preflight python - <<'PY'
import lightgbm, xgboost, catboost
print("Container native import check passed")
PY
else
  echo "[5/5] Skipping container smoke test (pass --smoke to enable)."
fi

echo "Docker preflight checks passed."
