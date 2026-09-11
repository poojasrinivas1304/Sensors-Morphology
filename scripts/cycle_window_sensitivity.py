#!/usr/bin/env python3
"""Assess late-stage A80 sensitivity to cycle-window onset and period.

This analysis is restricted to the 15 original 500-cycle records for which
retained 0.05-s timestamps are available.  It does not reconstruct missing
timestamps for Phase 1 or Phase 2.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data" / "phase1_3"
OUTPUT = ROOT / "frozen_inputs" / "phase1_3" / "cycle_window_sensitivity.csv"

SOURCES = [
    ("Phase3 inward", RAW / "Phase 3 master sheet.xlsx"),
    ("Phase3 outward", RAW / "Concave_sensor_tests_Origin_simple.xlsx"),
    ("Fabric", RAW / "Fabric_sensor_tests_Origin_simple.xlsx"),
]

PERIODS_S = (1.19, 1.20, 1.21)
ONSET_SHIFTS_S = (-0.10, 0.00, 0.10)
LATE_START_S = 540.0
MIN_OBSERVATIONS = 19


def iter_records(stage: str, path: Path):
    frame = pd.read_excel(path, header=None)
    for col in range(0, frame.shape[1], 5):
        if col + 3 >= frame.shape[1]:
            continue
        if str(frame.iloc[1, col + 1]).strip() != "Time_s":
            continue
        name = str(frame.iloc[0, col]).strip()
        time_s = pd.to_numeric(frame.iloc[2:, col + 1], errors="coerce").to_numpy(float)
        response = pd.to_numeric(frame.iloc[2:, col + 3], errors="coerce").to_numpy(float)
        valid = np.isfinite(time_s) & np.isfinite(response)
        yield stage, name, time_s[valid], response[valid]


def summarize(time_s: np.ndarray, response: np.ndarray, period_s: float, onset_s: float):
    window = np.floor((time_s - onset_s) / period_s).astype(int)
    rows = []
    for index in np.unique(window):
        start = onset_s + index * period_s
        stop = start + period_s
        if start < LATE_START_S or stop > time_s.max() + 0.051:
            continue
        values = response[window == index]
        if len(values) < MIN_OBSERVATIONS:
            continue
        rows.append(float(np.percentile(values, 90) - np.percentile(values, 10)))
    values = np.asarray(rows, dtype=float)
    if values.size == 0:
        return 0, np.nan, np.nan
    cv = 100.0 * values.std(ddof=1) / values.mean() if values.size > 1 else np.nan
    return int(values.size), float(np.median(values)), float(cv)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for stage, path in SOURCES:
        for _, record, time_s, response in iter_records(stage, path):
            for period_s in PERIODS_S:
                for onset_s in ONSET_SHIFTS_S:
                    n_cycles, median_a80, cv_a80 = summarize(
                        time_s, response, period_s, onset_s
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "record": record,
                            "source_file": path.name,
                            "period_s": period_s,
                            "onset_shift_s": onset_s,
                            "eligible_late_windows": n_cycles,
                            "median_late_A80_percentage_points": median_a80,
                            "late_A80_CV_percent": cv_a80,
                        }
                    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} sensitivity rows for {len(rows) // 9} records: {OUTPUT}")


if __name__ == "__main__":
    main()
