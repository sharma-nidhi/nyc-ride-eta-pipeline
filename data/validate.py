"""
Schema + data-quality validation for the NYC taxi dataset (Week 1 / M2).

PLACEHOLDER — implement the checks below, then `sys.exit(1)` on any failure so the
pipeline stops loudly instead of training on bad data.

Planned checks:
  - Required columns present (id, pickup/dropoff datetime + coords, passenger_count,
    trip_duration).
  - trip_duration > 0 and within a sane upper bound (drop multi-day outliers).
  - pickup/dropoff lat-lon inside NYC bounds (see config.yaml: nyc_bounds).
  - passenger_count in a plausible range.
  - Valid, parseable timestamps; dropoff after pickup.
  - No missing GPS pings / nulls in required columns; report duplicates.
  - Emit a validation report (counts of rows dropped per rule).
"""
import sys


def validate() -> None:
    raise NotImplementedError("TODO Week 1: implement schema + quality checks")


if __name__ == "__main__":
    validate()
    print("PASS: validation complete")
