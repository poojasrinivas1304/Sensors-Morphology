#!/usr/bin/env python3
"""Reproduce the Phase 1 specimen-level factorial analysis.

The independently fabricated sensor is the experimental unit.  The input
contains one endpoint row per canonical Phase 1 sensor.  Models use treatment
coding with CL2 and BL0 as reference levels, an interaction term, HC3 robust
covariance, and Student-t intervals with the residual degrees of freedom.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "frozen_inputs" / "phase1_3" / "Phase1_specimen_level_metrics.csv"
OUTDIR = ROOT / "generated" / "phase1"

ENDPOINTS = {
    "Late_A80_pct": "Late-stage A80 (percentage points)",
    "Resistance_evolution_pp": "Early-to-late resistance evolution (percentage points)",
    "Late_A80_CV_pct": "Late-stage within-sensor A80 CV (%)",
}

TERMS = [
    "Intercept (CL2-BL0)",
    "CL3-CL2 at BL0",
    "BL1-BL0 at CL2",
    "BL2-BL0 at CL2",
    "CL3 x BL1 interaction",
    "CL3 x BL2 interaction",
]

CONTRASTS = {
    "Marginal CL3-CL2": np.array([0, 1, 0, 0, 1 / 3, 1 / 3], float),
    "Marginal BL1-BL0": np.array([0, 0, 1, 0, 1 / 2, 0], float),
    "Marginal BL2-BL0": np.array([0, 0, 0, 1, 0, 1 / 2], float),
    "CL3-CL2 within BL0": np.array([0, 1, 0, 0, 0, 0], float),
    "CL3-CL2 within BL1": np.array([0, 1, 0, 0, 1, 0], float),
    "CL3-CL2 within BL2": np.array([0, 1, 0, 0, 0, 1], float),
}


def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    max_iter, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_cdf(t_value: float, df: int) -> float:
    x = df / (df + t_value * t_value)
    tail = 0.5 * regularized_beta(x, df / 2.0, 0.5)
    return 1.0 - tail if t_value >= 0 else tail


def t_quantile(probability: float, df: int) -> float:
    lo, hi = -50.0, 50.0
    for _ in range(160):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    cl3 = (frame["CL"] == 3).astype(float).to_numpy()
    bl1 = (frame["BL"] == 1).astype(float).to_numpy()
    bl2 = (frame["BL"] == 2).astype(float).to_numpy()
    return np.column_stack(
        [np.ones(len(frame)), cl3, bl1, bl2, cl3 * bl1, cl3 * bl2]
    )


def fit_hc3(frame: pd.DataFrame, endpoint: str) -> dict:
    x = design_matrix(frame)
    y = frame[endpoint].to_numpy(float)
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    residual = y - fitted
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    adjusted_sq = (residual / (1.0 - leverage)) ** 2
    meat = x.T @ (x * adjusted_sq[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    n, p = x.shape
    df = n - p
    tcrit = t_quantile(0.975, df)
    sse = float(residual @ residual)
    sst = float(((y - y.mean()) ** 2).sum())
    mse = sse / df
    cooks = (residual**2 / (p * mse)) * leverage / (1.0 - leverage) ** 2

    coefficients = []
    for name, estimate, se2 in zip(TERMS, beta, np.diag(covariance)):
        se = math.sqrt(max(float(se2), 0.0))
        t_value = float(estimate / se) if se else math.nan
        p_value = 2 * (1 - t_cdf(abs(t_value), df)) if se else math.nan
        coefficients.append(
            {
                "term": name,
                "estimate": float(estimate),
                "robust_se": se,
                "ci_low": float(estimate - tcrit * se),
                "ci_high": float(estimate + tcrit * se),
                "t_value": t_value,
                "p_value": p_value,
            }
        )

    contrasts = []
    for name, vector in CONTRASTS.items():
        estimate = float(vector @ beta)
        se = math.sqrt(max(float(vector @ covariance @ vector), 0.0))
        t_value = estimate / se if se else math.nan
        p_value = 2 * (1 - t_cdf(abs(t_value), df)) if se else math.nan
        contrasts.append(
            {
                "contrast": name,
                "estimate": estimate,
                "robust_se": se,
                "ci_low": estimate - tcrit * se,
                "ci_high": estimate + tcrit * se,
                "t_value": t_value,
                "p_value": p_value,
            }
        )

    diagnostics = {
        "n": n,
        "parameters": p,
        "residual_df": df,
        "t_critical_0.975": tcrit,
        "r_squared": 1.0 - sse / sst if sst else math.nan,
        "adjusted_r_squared": 1.0 - (sse / df) / (sst / (n - 1)) if sst else math.nan,
        "rmse": math.sqrt(mse),
        "condition_number": float(np.linalg.cond(x)),
        "max_leverage": float(leverage.max()),
        "max_leverage_sample": int(frame.iloc[int(leverage.argmax())]["Sample"]),
        "max_cooks_distance": float(cooks.max()),
        "max_cooks_sample": int(frame.iloc[int(cooks.argmax())]["Sample"]),
    }
    return {
        "beta": beta,
        "covariance": covariance,
        "coefficients": coefficients,
        "contrasts": contrasts,
        "diagnostics": diagnostics,
    }


def leave_one_out(frame: pd.DataFrame, endpoint: str, full: dict) -> list[dict]:
    reference = {x["contrast"]: x["estimate"] for x in full["contrasts"]}
    rows = []
    for index, specimen in frame.iterrows():
        reduced = frame.drop(index=index).reset_index(drop=True)
        result = fit_hc3(reduced, endpoint)
        for contrast in result["contrasts"]:
            name = contrast["contrast"]
            rows.append(
                {
                    "omitted_sample": int(specimen["Sample"]),
                    "contrast": name,
                    "estimate": contrast["estimate"],
                    "change_from_full": contrast["estimate"] - reference[name],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_canonical_endpoints(phase1: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for record in phase1.to_dict("records"):
        rows.append(
            {
                "specimen_id": str(int(record["Sample"])),
                "phase": "Phase 1",
                "design": record["Architecture"],
                "orientation": "Conductive-inward",
                "replication_role": "Factorial experiment; n=3 per architecture",
                "late_A80_pp": record["Late_A80_pct"],
                "resistance_evolution_pp": record["Resistance_evolution_pp"],
                "late_A80_CV_pct": record["Late_A80_CV_pct"],
                "source": "MATLAB/Phase1_specimen_level_metrics.csv",
            }
        )

    inputs = ROOT / "frozen_inputs" / "phase1_3"
    phase2 = pd.read_csv(inputs / "Phase2_specimen_level_metrics.csv")
    for record in phase2.to_dict("records"):
        rows.append(
            {
                "specimen_id": str(int(record["Sample"])),
                "phase": "Phase 2",
                "design": f"{record['Pattern']} {int(record['Infill_pct'])}%",
                "orientation": "Conductive-inward",
                "replication_role": "Exploratory screen; n=1 per condition",
                "late_A80_pp": record["Late_A80_pct"],
                "resistance_evolution_pp": record["Resistance_evolution_pp"],
                "late_A80_CV_pct": record["Late_A80_CV_pct"],
                "source": "MATLAB/Phase2_specimen_level_metrics.csv",
            }
        )

    with (inputs / "phase3_metrics.json").open(encoding="utf-8") as handle:
        inward = json.load(handle)["metrics"]
    for record in inward:
        rows.append(
            {
                "specimen_id": str(record["sample"]),
                "phase": "Phase 3 inward confirmation",
                "design": record["condition"],
                "orientation": "Conductive-inward",
                "replication_role": "Independent confirmation; n=3 per finalist",
                "late_A80_pp": record["late_A80_pct"],
                "resistance_evolution_pp": record["resistance_evolution_pp"],
                "late_A80_CV_pct": record["late_A80_CV_pct"],
                "source": "phase3_audit/phase3_metrics.json",
            }
        )

    with (inputs / "outward_metrics.json").open(encoding="utf-8") as handle:
        outward = json.load(handle)["metrics"]
    for record in outward:
        rows.append(
            {
                "specimen_id": str(record["sample"]),
                "phase": "Phase 3 outward pilot",
                "design": record["condition"],
                "orientation": "Conductive-outward",
                "replication_role": "Exploratory orientation pilot; n=1 per condition",
                "late_A80_pp": record["late_A80_pct"],
                "resistance_evolution_pp": record["resistance_evolution_pp"],
                "late_A80_CV_pct": record["late_A80_CV_pct"],
                "source": "phase3_audit/outward_metrics.json",
            }
        )

    fabric_map = {
        "Grid80_standard": ("F-G80-I", "Grid 80%", "Conductive-inward"),
        "Grid80_concave": ("F-G80-O", "Grid 80%", "Conductive-outward"),
        "Rect80_standard": ("F-R80-I", "Rectilinear 80%", "Conductive-inward"),
        "Rect80_concave": ("F-R80-O", "Rectilinear 80%", "Conductive-outward"),
        "Rect100_convex": ("F-S100-I", "Solid 100%", "Conductive-inward"),
        "Rect100_concave": ("F-S100-O", "Solid 100%", "Conductive-outward"),
    }
    with (inputs / "fabric_metrics.json").open(encoding="utf-8") as handle:
        fabric = json.load(handle)
    for record in fabric:
        specimen_id, design, orientation = fabric_map[record["record"]]
        rows.append(
            {
                "specimen_id": specimen_id,
                "phase": "Fabric feasibility",
                "design": design,
                "orientation": orientation,
                "replication_role": "Independent fabric sensor; n=1 per design-orientation",
                "late_A80_pp": record["late_median_a80_pct_points"],
                "resistance_evolution_pp": record["resistance_evolution_pct_points"],
                "late_A80_CV_pct": record["late_amplitude_cv_pct"],
                "source": f"Fabric workbook label: {record['record']}",
            }
        )
    return rows


def data_dictionary() -> list[dict]:
    return [
        {"field": "specimen_id", "definition": "Stable identifier for one independently fabricated sensor.", "unit_or_levels": "Samples 1-36 or fabric IDs F-G80-I to F-S100-O", "analysis_role": "Experimental-unit identifier"},
        {"field": "phase", "definition": "Sequential experimental stage.", "unit_or_levels": "Phase 1; Phase 2; Phase 3 inward; Phase 3 outward; Fabric", "analysis_role": "Separates factorial, screening, confirmation and feasibility evidence"},
        {"field": "CL", "definition": "Number of conductive TPU layers.", "unit_or_levels": "2 or 3", "analysis_role": "Phase 1 categorical factor"},
        {"field": "BL", "definition": "Number of nonconductive backing TPU layers.", "unit_or_levels": "0, 1 or 2", "analysis_role": "Phase 1 categorical factor"},
        {"field": "design", "definition": "Architecture or conductive-region morphology and nominal infill.", "unit_or_levels": "CLx-BLy; morphology 40%, 60%, 80%; solid 100%", "analysis_role": "Design condition"},
        {"field": "orientation", "definition": "Position of the conductive region relative to the bent surface.", "unit_or_levels": "Conductive-inward or conductive-outward", "analysis_role": "Descriptive orientation label; not inferred for n=1 pilots"},
        {"field": "time_s", "definition": "Electrical acquisition time.", "unit_or_levels": "s", "analysis_role": "Raw/processed record coordinate"},
        {"field": "cycle", "definition": "Cycle number assigned from waveform onset using the 1.2-s nominal period.", "unit_or_levels": "1-500", "analysis_role": "Repeated observation within a sensor"},
        {"field": "R_ohm", "definition": "Resistance calculated from the voltage-divider reading.", "unit_or_levels": "ohm", "analysis_role": "Measured electrical response"},
        {"field": "R0_ohm", "definition": "Median resistance during the first 5 s of the stationary pre-cycling interval.", "unit_or_levels": "ohm", "analysis_role": "Single specimen-specific normalization reference"},
        {"field": "dR_R0_pct", "definition": "100 x (R-R0)/R0.", "unit_or_levels": "%", "analysis_role": "Normalized response"},
        {"field": "cycle_center_pct", "definition": "Median normalized response within an eligible cycle.", "unit_or_levels": "percentage points", "analysis_role": "Cycle-centre trajectory"},
        {"field": "A80_cycle_pp", "definition": "Within-cycle P90 minus P10 of normalized response.", "unit_or_levels": "percentage points", "analysis_role": "Robust cyclic excursion"},
        {"field": "late_A80_pp", "definition": "Median A80 across eligible Cycles 451-500.", "unit_or_levels": "percentage points", "analysis_role": "Primary specimen-level response magnitude"},
        {"field": "resistance_evolution_pp", "definition": "Median cycle centre in Cycles 451-500 minus that in Cycles 1-50.", "unit_or_levels": "percentage points", "analysis_role": "Specimen-level resistance evolution"},
        {"field": "late_A80_CV_pct", "definition": "100 times sample SD divided by arithmetic mean of A80 in eligible Cycles 451-500.", "unit_or_levels": "%", "analysis_role": "Within-sensor late-stage cycle variability"},
        {"field": "eligible_cycle", "definition": "Cycle with at least 19 measured observations.", "unit_or_levels": "TRUE/FALSE", "analysis_role": "Primary cycle-level inclusion rule"},
        {"field": "replication_role", "definition": "Role of the record in screening, confirmation or feasibility testing.", "unit_or_levels": "Text", "analysis_role": "Prevents cycles or exploratory specimens being treated as independent confirmatory replicates"},
        {"field": "source", "definition": "Canonical processed file or original workbook label used for the row.", "unit_or_levels": "Path or source label", "analysis_role": "Audit trail"},
    ]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    split = frame["Architecture"].str.extract(r"CL(?P<CL>[23])-BL(?P<BL>[012])")
    frame[["CL", "BL"]] = split.astype(int)

    coefficient_rows: list[dict] = []
    contrast_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    loo_rows: list[dict] = []
    results = {}
    for endpoint, label in ENDPOINTS.items():
        fit = fit_hc3(frame, endpoint)
        results[endpoint] = {
            "label": label,
            "coefficients": fit["coefficients"],
            "contrasts": fit["contrasts"],
            "diagnostics": fit["diagnostics"],
        }
        coefficient_rows.extend(
            [{"endpoint": label, **row} for row in fit["coefficients"]]
        )
        contrast_rows.extend(
            [{"endpoint": label, **row} for row in fit["contrasts"]]
        )
        diagnostic_rows.append({"endpoint": label, **fit["diagnostics"]})
        loo_rows.extend(
            [{"endpoint": label, **row} for row in leave_one_out(frame, endpoint, fit)]
        )

    write_csv(OUTDIR / "phase1_coefficients_hc3.csv", coefficient_rows)
    write_csv(OUTDIR / "phase1_planned_contrasts_hc3.csv", contrast_rows)
    write_csv(OUTDIR / "phase1_model_diagnostics.csv", diagnostic_rows)
    write_csv(OUTDIR / "phase1_leave_one_out_contrasts.csv", loo_rows)
    write_csv(OUTDIR / "canonical_specimen_endpoints.csv", build_canonical_endpoints(frame))
    write_csv(OUTDIR / "data_dictionary.csv", data_dictionary())
    with (OUTDIR / "phase1_factorial_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Analysed {len(frame)} canonical Phase 1 sensors from {INPUT.name}")
    for endpoint, result in results.items():
        print(f"\n{ENDPOINTS[endpoint]}")
        for row in result["contrasts"]:
            print(
                f"  {row['contrast']}: {row['estimate']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            )


if __name__ == "__main__":
    main()
