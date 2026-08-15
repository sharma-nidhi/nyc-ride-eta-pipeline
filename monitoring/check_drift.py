import sqlite3
import numpy as np
import pandas as pd

TRAIN = "data/processed/features.parquet"
PRED_DB = "monitoring/predictions.db"
REPORT = "monitoring/drift_report.txt"
SIGMA_THRESHOLD = 0.8      # flag if the mean moved > 0.8 std devs
PSI_THRESHOLD = 0.2        # PSI > 0.2 = significant distribution shift
FEATURES = ["haversine_km", "hour", "passenger_count"]


def psi(expected, actual, bins=10):
    """Population Stability Index between two distributions."""
    cuts = np.percentile(expected, np.linspace(0, 100, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    cuts = np.unique(cuts)
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    train = pd.read_parquet(TRAIN)
    prod = pd.read_sql("SELECT * FROM predictions", sqlite3.connect(PRED_DB))
    prod["hour"] = pd.to_datetime(prod["pickup_datetime"]).dt.hour

    lines = [f"Training rows: {len(train):,} | Production rows: {len(prod):,}", ""]
    lines.append(f"{'feature':16s}{'train_mean':>12s}{'prod_mean':>12s}{'shift(sig)':>12s}{'PSI':>8s}  status")

    drift = False
    for f in FEATURES:
        t, p = train[f].astype(float), prod[f].astype(float)
        shift = abs(p.mean() - t.mean()) / (t.std() + 1e-9)
        pstat = psi(t.values, p.values)
        flagged = shift > SIGMA_THRESHOLD or pstat > PSI_THRESHOLD
        drift = drift or flagged
        lines.append(f"{f:16s}{t.mean():12.2f}{p.mean():12.2f}{shift:12.2f}{pstat:8.2f}  {'DRIFT' if flagged else 'ok'}")

    lines.append("")
    lines.append("RESULT: " + ("DRIFT DETECTED -> retraining recommended" if drift else "No significant drift"))

    text = "\n".join(lines)
    print(text)
    with open(REPORT, "w", encoding="utf-8") as fp:
        fp.write(text + "\n")
    print(f"\nSaved -> {REPORT}")


if __name__ == "__main__":
    main()