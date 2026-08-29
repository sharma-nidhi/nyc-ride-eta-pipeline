"""Monitoring, drift detection, and retraining subsystem.

All imports are lazy to avoid dragging in heavy dependencies (evidently, optuna)
at API startup time. Use explicit imports like:
    from src.monitoring.drift_report import generate_drift_report
"""