"""
Drift detection + monitoring signals (Week 4 / M5).

PLACEHOLDER — compare live prediction traffic to the training baseline.

Plan:
  - Load training feature distribution + logged production inputs.
  - Per numeric feature: mean shift in σ (flag > drift_sigma_threshold) and/or PSI/KS.
  - If actuals available: track RMSLE/MAE over time.
  - Print an actionable report + simple plots; this feeds the retraining trigger.
"""


def main() -> None:
    raise NotImplementedError("TODO Week 4: implement drift comparison + signals")


if __name__ == "__main__":
    main()
