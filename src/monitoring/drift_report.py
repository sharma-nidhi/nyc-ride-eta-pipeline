# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

"""
Drift detection and reporting using Evidently AI (v0.7+).
Compares production logs against the original training dataset in model feature space.
Also provides a decision-only retraining policy check based on drift thresholds.
"""
import pathlib
import logging
import json
import warnings
import pandas as pd
from evidently import Report, presets

from src.features.feature_pipeline import load_pipeline

logger = logging.getLogger(__name__)

REF_PATH = pathlib.Path("data/processed/X_train.parquet")
LOG_PATH = pathlib.Path("data/monitoring/production_logs.jsonl")
REPORT_DIR = pathlib.Path("reports/drift")

# Decision policy thresholds
DRIFT_SEVERITY_THRESHOLD = 0.30


RAW_REQUIRED_COLS = [
    "pickup_datetime",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "vendor_id",
    "store_and_fwd_flag",
]


def _extract_feature_drift_rows(snapshot_dict: dict) -> list[tuple[str, float, float, bool]]:
    """
    Extract per-feature drift rows from Evidently snapshot metrics.

    Returns tuples as:
        (feature_name, drift_score, threshold, drift_detected)
    """
    rows: list[tuple[str, float, float, bool]] = []
    for metric in snapshot_dict.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        if not name.startswith("ValueDrift("):
            continue

        config = metric.get("config", {})
        feature = str(config.get("column", "unknown"))

        try:
            threshold = float(config.get("threshold", 0.0))
            score = float(metric.get("value", 0.0))
        except (TypeError, ValueError):
            continue

        rows.append((feature, score, threshold, score > threshold))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _extract_drift_summary(snapshot_dict: dict) -> tuple[float | None, int | None]:
    """Extract drifted feature share/count from Evidently snapshot metrics."""
    try:
        for metric in snapshot_dict.get("metrics", []):
            if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
                value = metric.get("value", {})
                return value.get("share"), value.get("count")
    except Exception:
        # Keep report generation resilient even if metric schema changes.
        pass
    return None, None


def _build_feature_drift_html(rows: list[tuple[str, float, float, bool]]) -> str:
    """Build an HTML section with per-feature drift diagnostics."""
    if not rows:
        return ""

    table_rows = []
    for feature, score, threshold, is_drifted in rows:
        status = "Yes" if is_drifted else "No"
        table_rows.append(
            "<tr>"
            f"<td>{feature}</td>"
            f"<td>{score:.6f}</td>"
            f"<td>{threshold:.6f}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )

    return (
        "<section id=\"feature-drift-summary\" "
        "style=\"margin:24px auto;max-width:1100px;padding:16px;border:1px solid #ddd;border-radius:8px;\">"
        "<h2 style=\"margin-top:0;\">Per-feature drift summary</h2>"
        "<p style=\"margin-top:0;\">"
        "Each feature is drifted when drift score is greater than threshold."
        "</p>"
        "<div style=\"overflow-x:auto;\">"
        "<table style=\"width:100%;border-collapse:collapse;font-family:Arial,sans-serif;\">"
        "<thead><tr>"
        "<th style=\"text-align:left;border-bottom:1px solid #ccc;padding:8px;\">Feature</th>"
        "<th style=\"text-align:left;border-bottom:1px solid #ccc;padding:8px;\">Drift score</th>"
        "<th style=\"text-align:left;border-bottom:1px solid #ccc;padding:8px;\">Threshold</th>"
        "<th style=\"text-align:left;border-bottom:1px solid #ccc;padding:8px;\">Drifted</th>"
        "</tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def _inject_feature_drift_html(report_path: pathlib.Path, rows: list[tuple[str, float, float, bool]]) -> None:
    """Append per-feature drift summary into Evidently HTML report."""
    if not rows or not report_path.exists():
        return

    html = report_path.read_text(encoding="utf-8")
    marker = "id=\"feature-drift-summary\""
    if marker in html:
        return

    section = _build_feature_drift_html(rows)
    if "</body>" in html:
        html = html.replace("</body>", section + "</body>", 1)
    else:
        html += section

    report_path.write_text(html, encoding="utf-8")


def _build_current_feature_frame(curr_raw: pd.DataFrame, ref_columns: list[str]) -> pd.DataFrame:
    """
    Convert raw production logs to the same engineered feature space as training.

    This is the most reliable drift view for this model because the model consumes
    engineered features (hour_sin/cos, haversine, etc.), not raw JSON request fields.
    """
    missing = [c for c in RAW_REQUIRED_COLS if c not in curr_raw.columns]
    if missing:
        raise ValueError(
            "Production logs are missing required fields for feature transformation: "
            f"{missing}"
        )

    raw = curr_raw[RAW_REQUIRED_COLS].copy()

    # Coerce numeric fields and datetime; drop malformed rows for robustness.
    numeric_cols = [
        "passenger_count",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "vendor_id",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Logs may contain mixed datetime formats (naive + timezone-aware ISO strings).
    # Pandas in this environment fails one-pass parsing on mixed formats, so we do two passes:
    # 1) default parser, 2) strict ISO8601 fallback for NaT values.
    dt_default = pd.to_datetime(raw["pickup_datetime"], errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        dt_iso = pd.to_datetime(raw["pickup_datetime"], errors="coerce", format="ISO8601")
    dt = dt_default.fillna(dt_iso)

    # Normalize to naive UTC timestamps for feature extraction consistency.
    def _to_naive_utc(x):
        if pd.isna(x):
            return pd.NaT
        ts = pd.Timestamp(x)
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").tz_localize(None)
        return ts

    raw["pickup_datetime"] = dt.map(_to_naive_utc)

    before = len(raw)
    raw = raw.dropna(subset=RAW_REQUIRED_COLS)
    dropped = before - len(raw)
    if dropped > 0:
        logger.warning("Dropped %d malformed production rows before drift transform", dropped)
    if raw.empty:
        raise ValueError("No valid production rows remain after type coercion")

    pipeline = load_pipeline()
    transformed = pipeline.transform(raw)

    curr_features = pd.DataFrame(transformed, columns=ref_columns)
    return curr_features


def generate_drift_report(limit: int | None = None, include_summary: bool = False) -> str | dict:
    """
    Generate an HTML drift report comparing recent production logs
    to the reference training data.

    Parameters
    ----------
    limit : int | None
        Number of recent production samples to analyze.
        If None, analyze all available production samples.

    Returns
    -------
    str | dict
        Path to the generated HTML report.
        When include_summary=True, returns a dict with report path and drift summary.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Reference Data
    if not REF_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found at {REF_PATH}")
    ref = pd.read_parquet(REF_PATH)
    logger.info("Loaded reference dataset (%d samples)", len(ref))

    # Load Current Production Data (raw logs)
    rows = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("Production logs are empty. Run some predictions first!")
        if limit is not None:
            lines = lines[-limit:]
        for line in lines:
            rows.append(json.loads(line))

    curr_raw = pd.DataFrame(rows)
    logger.info("Loaded current production dataset (%d samples)", len(curr_raw))

    # Transform raw logs into engineered model features for apples-to-apples drift.
    curr = _build_current_feature_frame(curr_raw, ref.columns.tolist())

    # Align to exact same feature set and order.
    common_cols = [c for c in ref.columns if c in curr.columns]
    ref = ref[common_cols].copy()
    curr = curr[common_cols].copy()
    logger.info("Comparing %d model features: %s", len(common_cols), common_cols)

    # Build report with Evidently presets (presets are MetricContainers accepted by Report)
    report = Report([
        presets.DataDriftPreset(),
        presets.DataSummaryPreset(),
    ])

    snapshot = report.run(current_data=curr, reference_data=ref)
    snapshot_dict = snapshot.dict()
    output_path = REPORT_DIR / "drift_report.html"
    snapshot.save_html(str(output_path))

    # Print a concise drift summary to make quick validation easier in CLI demos.
    total_features = len(common_cols)
    drift_share, drift_count = _extract_drift_summary(snapshot_dict)

    if drift_share is not None and drift_count is not None:
        logger.info(
            "Drift summary: %.3f%% of columns drifted (%d/%d)",
            float(drift_share) * 100,
            int(drift_count),
            total_features,
        )

    # Show per-feature drift diagnostics for quick CLI interpretation.
    feature_rows = _extract_feature_drift_rows(snapshot_dict)
    if feature_rows:
        logger.info("Per-feature drift scores (score > threshold => drift):")
        for feature, score, threshold, is_drifted in feature_rows:
            logger.info(
                "  - %-20s score=%8.4f threshold=%6.3f drifted=%s",
                feature,
                score,
                threshold,
                str(is_drifted),
            )

    _inject_feature_drift_html(output_path, feature_rows)

    logger.info("Drift report saved to %s", output_path)

    if include_summary:
        return {
            "report_path": str(output_path.resolve()),
            "drift_share": float(drift_share) if drift_share is not None else None,
            "drift_count": int(drift_count) if drift_count is not None else None,
            "total_features": total_features,
        }

    return str(output_path.resolve())


def evaluate_retrain_need(
    drift_share: float | None = None,
    drift_threshold: float = DRIFT_SEVERITY_THRESHOLD,
) -> dict:
    """Evaluate drift-based retraining recommendation."""

    diagnostics = {
        "drift": "",
    }
    reasons: list[str] = []

    if drift_share is None:
        diagnostics["drift"] = "Skipped: drift severity unavailable from current logs/reference data."
    elif drift_share >= drift_threshold:
        reasons.append(
            f"Data drift detected: {drift_share:.0%} of features drifted (threshold {drift_threshold:.0%})"
        )
        diagnostics["drift"] = f"Triggered: {drift_share:.2%} >= {drift_threshold:.2%} threshold"
    else:
        diagnostics["drift"] = f"Not triggered: {drift_share:.2%} < {drift_threshold:.2%} threshold"

    if reasons:
        return {
            "action": "triggered",
            "drift_share": drift_share,
            "reasons": reasons,
            "diagnostics": diagnostics,
        }

    return {
        "action": "skipped",
        "drift_share": drift_share,
        "reasons": [],
        "diagnostics": diagnostics,
        "message": "No retrain threshold crossed",
    }


def _print_decision_summary(result: dict) -> None:
    """Print a human-readable summary of trigger checks and final decision."""
    drift_share = result.get("drift_share")
    diagnostics = result.get("diagnostics", {})

    print("\n=== Retraining Policy Check ===")
    if drift_share is None:
        print("Drift context: unavailable")
    else:
        print(f"Drift context: {drift_share:.2%} drifted features")

    print("\nTrigger evaluation:")
    print(f"- Data drift severity: {diagnostics.get('drift', 'N/A')}")

    if result.get("action") == "triggered":
        print("\nDecision: RETRAINING NEEDED")
        print(f"Reason(s): {'; '.join(result.get('reasons', []))}")
        print("Next step: schedule/approve a training run in your pipeline workflow.")
    else:
        print("\nDecision: NO RETRAINING NEEDED")
        print(f"Summary: {result.get('message', 'No retrain threshold crossed')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate drift report and evaluate decision-only retraining triggers"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of most recent production samples to analyze (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = generate_drift_report(limit=args.limit, include_summary=True)
    print(f"\nOpen your browser to view: file:///{summary['report_path']}")

    decision = evaluate_retrain_need(
        drift_share=summary.get("drift_share"),
    )
    _print_decision_summary(decision)
