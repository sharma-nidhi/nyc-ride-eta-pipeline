# Retraining Trigger Design — NYC Ride ETA (Week 4 / M5)

How the deployed model is kept healthy: what we monitor, what fires a retrain, and how a new
model is validated and promoted.

## Signals we monitor
1. **Feature drift (leading, label-free).** `monitoring/check_drift.py` compares live traffic
   (logged in `predictions.db`) to the `data-v1` training baseline. Per feature it computes the
   **mean shift in σ** and the **PSI**. Thresholds: **σ-shift > 0.8** or **PSI > 0.2**.
2. **Prediction error (confirming, needs labels).** Once a trip finishes, its real
   `trip_duration` is known. We join actuals back to logged predictions and track a **rolling
   RMSLE**. The production baseline is **0.397**; we alert if rolling RMSLE exceeds **0.44**
   (~10% worse).
3. **Traffic sanity.** Sudden spikes in 422 rejection rate or request volume — a sign the input
   distribution moved outside the trained/validated range.

## Trigger rule
Retrain when **any** of the following holds:
- **A.** ≥1 key feature (`haversine_km`, `hour`, `passenger_count`) breaches PSI > 0.2 or
  σ-shift > 0.8, **sustained** across a monitoring window (e.g., 2 consecutive daily checks or
  ≥ 500 requests) — so we react to real shifts, not noise.
- **B.** Rolling RMSLE on labeled predictions > 0.44 (10% above baseline).
- **C.** **Scheduled safety net:** retrain at least every 4 weeks regardless, to absorb slow
  seasonal drift that never trips a single threshold.

*Why this mix:* feature drift (A) is a fast early warning but can't see concept drift; error (B)
is the ground truth but lags until actuals arrive; the schedule (C) catches gradual drift. Today's
simulation fired rule A (haversine_km, hour, passenger_count all flagged).

## Retraining workflow
1. **Assemble data** — append recent trips to `data/raw`, re-run `data/validate.py` +
   `features/build_features.py`, and tag the new dataset `data-v2` with DVC.
2. **Train** — run `training/train.py` (same pipeline, same `random_state`) → candidate model,
   fully tracked in MLflow.
3. **Validation gate** — evaluate the candidate on a fresh hold-out. **Promote only if RMSLE ≤
   current production RMSLE** (no regression); otherwise keep the current model and alert.
4. **Promotion** — serialize as `eta-v2`, point serving at it, and **keep `eta-v1` for rollback**.
   Optionally shadow-serve v2 alongside v1 first and compare before full cutover.
5. **Rollback** — if post-deployment monitoring worsens, revert `MODEL_PATH` to `eta-v1`.

## Ownership & cadence
- `check_drift.py` runs on a **daily schedule**. A fired trigger produces `drift_report.txt`, which
  the **ML engineer reviews** before approving a retrain (human-in-the-loop, not fully automatic).
- Everything stays versioned and reproducible: data via **DVC tags**, experiments via **MLflow**,
  model artifacts as **`eta-vN`**.

## Known limitations
- **Concept drift** (the feature→duration relationship itself changing) needs labels to detect;
  feature-drift alone can miss it — which is why signal B exists.
- SQLite logging + manual approval fit this project's scale. Production would use a streaming/
  warehouse store, an orchestrated pipeline (Airflow/Prefect), and automated shadow deployment.
