from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "raw_data" / "hysteresis"
OUTPUT = ROOT / "generated" / "hysteresis"
LEVELS = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def segments(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    starts = np.r_[idx[0], idx[1:][np.diff(idx) > 1]]
    ends = np.r_[idx[:-1][np.diff(idx) > 1], idx[-1]]
    return list(zip(starts, ends))


def load_pair(sample):
    esp_path = Path(glob.glob(str(SOURCE / f"hysteresis{sample}*esp.csv"))[0])
    utm_path = Path(glob.glob(str(SOURCE / "utm" / f"hysteresis{sample}_*.csv"))[0])
    esp_raw = pd.read_csv(esp_path)
    esp = esp_raw.apply(pd.to_numeric, errors="coerce").dropna(subset=["R_ohm"])
    utm = pd.read_csv(utm_path, skiprows=[1]).apply(pd.to_numeric, errors="coerce")
    return esp_path, utm_path, esp_raw, esp, utm


def detect_plateaus(utm):
    rows = []
    for level in LEVELS:
        mask = (np.abs(utm["Displacement"].to_numpy() - level) <= 0.005)
        for start_idx, end_idx in segments(mask):
            start = float(utm["Time"].iloc[start_idx])
            end = float(utm["Time"].iloc[end_idx])
            if end - start > 5:
                rows.append({"start_s": start, "end_s": end, "displacement_mm": float(level),
                             "duration_s": end - start})
    return pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)


def lag_scores(esp, utm, end_time):
    t = np.arange(0, end_time, 0.1)
    disp = np.interp(t, utm["Time"], utm["Displacement"])
    d_disp = pd.Series(disp).rolling(10, center=True, min_periods=1).mean().diff().abs().to_numpy()
    scores = []
    for lag in np.arange(0, 61, 0.1):
        resistance = np.interp(t + lag, esp["elapsed_s"], esp["R_ohm"])
        d_resistance = pd.Series(resistance).rolling(10, center=True, min_periods=1).mean().diff().abs().to_numpy()
        valid = np.isfinite(d_disp) & np.isfinite(d_resistance)
        score = np.corrcoef(d_disp[valid], d_resistance[valid])[0, 1]
        scores.append((float(score), float(lag)))
    ranked = []
    for score, lag in sorted(scores, reverse=True):
        if all(abs(lag - existing[1]) > 2 for existing in ranked):
            ranked.append((score, lag))
        if len(ranked) == 6:
            break
    return ranked


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = []
    all_plateaus = []
    all_cycles = []
    all_specimens = []
    all_mechanical = []
    all_timing = []
    for sample in range(1, 6):
        esp_path, utm_path, esp_raw, esp, utm = load_pair(sample)
        plateaus = detect_plateaus(utm)
        high = plateaus[plateaus["displacement_mm"] == 3.0]
        complete_end = float(plateaus.loc[plateaus["displacement_mm"] == 0.1, "end_s"].iloc[-1])
        ranked = lag_scores(esp, utm, min(834, float(utm["Time"].iloc[-1])))
        lag = ranked[0][1]
        audit.append({
            "sample": sample,
            "esp_file": esp_path.name,
            "utm_file": utm_path.name,
            "utm_duration_s": float(utm["Time"].iloc[-1]),
            "esp_duration_s": float(esp["elapsed_s"].iloc[-1]),
            "R0_ohm": float(esp["R0_ohm"].dropna().iloc[0]),
            "valid_resistance_rows": int(esp["R_ohm"].notna().sum()),
            "missing_resistance_rows": int(esp_raw["R_ohm"].isna().sum()),
            "adc_3900_plus": int((pd.to_numeric(esp_raw["adc1"], errors="coerce") >= 3900).sum()),
            "complete_upper_plateaus": int(len(high)),
            "last_low_plateau_end_s": complete_end,
            "lag_candidates": ranked,
        })
        plateaus.to_csv(OUTPUT / f"sample_{sample}_utm_plateaus.csv", index=False)

        # The first plateau is the initial 0.10-mm baseline. The next 60 plateaus
        # comprise five ordered loading/unloading staircases (12 plateaus each).
        analysis_plateaus = plateaus.iloc[1:61].copy().reset_index(drop=True)
        if len(analysis_plateaus) != 60:
            raise RuntimeError(f"Sample {sample}: expected 60 staircase plateaus, found {len(analysis_plateaus)}")
        analysis_plateaus["sample"] = sample
        analysis_plateaus["cycle"] = np.repeat(np.arange(1, 6), 12)
        analysis_plateaus["phase"] = np.tile(["Loading"] * 6 + ["Unloading"] * 6, 5)
        analysis_plateaus["alignment_offset_s"] = lag
        r0 = float(esp["R0_ohm"].dropna().iloc[0])

        medians = []
        mads = []
        counts = []
        for row in analysis_plateaus.itertuples(index=False):
            values = esp.loc[
                (esp["elapsed_s"] >= row.end_s + lag - 3.0)
                & (esp["elapsed_s"] <= row.end_s + lag),
                "R_ohm",
            ].to_numpy()
            median = float(np.median(values))
            medians.append(median)
            mads.append(float(np.median(np.abs(values - median))))
            counts.append(int(len(values)))
        analysis_plateaus["R_median_ohm"] = medians
        analysis_plateaus["R_MAD_ohm"] = mads
        analysis_plateaus["plateau_observations"] = counts
        analysis_plateaus["dR_R0_pct"] = 100.0 * (analysis_plateaus["R_median_ohm"] - r0) / r0
        analysis_plateaus["robust_noise_pct_R"] = 100.0 * 1.4826 * analysis_plateaus["R_MAD_ohm"] / analysis_plateaus["R_median_ohm"]
        all_plateaus.append(analysis_plateaus)

        for timing_shift in [-1.0, 0.0, 1.0]:
            timing_values = []
            for row in analysis_plateaus.itertuples(index=False):
                values = esp.loc[
                    (esp["elapsed_s"] >= row.end_s + lag + timing_shift - 3.0)
                    & (esp["elapsed_s"] <= row.end_s + lag + timing_shift),
                    "R_ohm",
                ].to_numpy()
                timing_values.append(float(np.median(values)) if len(values) else np.nan)
            timing_frame = analysis_plateaus[["cycle", "phase", "displacement_mm"]].copy()
            timing_frame["dR_R0_pct"] = 100.0 * (np.asarray(timing_values) - r0) / r0
            timing_cycle_values = []
            for _, group in timing_frame.groupby("cycle"):
                loading = group[group["phase"] == "Loading"].set_index("displacement_mm")["dR_R0_pct"]
                unloading = group[group["phase"] == "Unloading"].set_index("displacement_mm")["dR_R0_pct"]
                shared = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
                timing_cycle_values.append(float(np.mean(np.abs((loading.loc[shared] - unloading.loc[shared]).to_numpy()))))
            all_timing.append({
                "sample": sample,
                "timing_shift_from_selected_s": timing_shift,
                "selected_alignment_offset_s": lag,
                "specimen_median_MAH_pp": float(np.nanmedian(timing_cycle_values)),
            })

        cycle_rows = []
        for cycle, group in analysis_plateaus.groupby("cycle"):
            loading = group[group["phase"] == "Loading"].set_index("displacement_mm")["dR_R0_pct"]
            unloading = group[group["phase"] == "Unloading"].set_index("displacement_mm")["dR_R0_pct"]
            shared = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
            signed_gap = (loading.loc[shared] - unloading.loc[shared]).to_numpy()
            cycle_rows.append({
                "sample": sample,
                "cycle": int(cycle),
                "mean_absolute_hysteresis_pp": float(np.mean(np.abs(signed_gap))),
                "absolute_loop_area_pp_mm": float(np.trapezoid(np.abs(signed_gap), shared)),
                "signed_loop_area_pp_mm": float(np.trapezoid(signed_gap, shared)),
                "response_at_3mm_pct": float(loading.loc[3.0]),
                "recovery_at_0p1mm_pct": float(unloading.loc[0.1]),
                "median_plateau_noise_pct_R": float(group["robust_noise_pct_R"].median()),
            })
        cycle_frame = pd.DataFrame(cycle_rows)
        all_cycles.append(cycle_frame)

        first_peak = float(cycle_frame.loc[cycle_frame["cycle"] == 1, "response_at_3mm_pct"].iloc[0])
        last_peak = float(cycle_frame.loc[cycle_frame["cycle"] == 5, "response_at_3mm_pct"].iloc[0])
        all_specimens.append({
            "sample": sample,
            "alignment_offset_s": lag,
            "median_MAH_pp": float(cycle_frame["mean_absolute_hysteresis_pp"].median()),
            "median_absolute_area_pp_mm": float(cycle_frame["absolute_loop_area_pp_mm"].median()),
            "median_signed_area_pp_mm": float(cycle_frame["signed_loop_area_pp_mm"].median()),
            "median_3mm_response_pct": float(cycle_frame["response_at_3mm_pct"].median()),
            "cycle1_3mm_response_pct": first_peak,
            "cycle5_3mm_response_pct": last_peak,
            "change_3mm_cycle1_to_5_pp": last_peak - first_peak,
            "relative_change_3mm_cycle1_to_5_pct": 100.0 * (last_peak - first_peak) / abs(first_peak) if first_peak != 0 else np.nan,
            "median_recovery_0p1mm_pct": float(cycle_frame["recovery_at_0p1mm_pct"].median()),
            "median_plateau_noise_pct_R": float(cycle_frame["median_plateau_noise_pct_R"].median()),
        })

        # Mechanical loop descriptors are calculated from synchronized UTM data,
        # but remain QC-only because measured forces are very small relative to
        # the nominal 50-kN load-cell capacity.
        initial_low = plateaus[plateaus["displacement_mm"] == 0.1].reset_index(drop=True)
        upper = plateaus[plateaus["displacement_mm"] == 3.0].reset_index(drop=True)
        grid = np.linspace(0.1, 3.0, 291)
        for cycle in range(1, 6):
            load_start = float(initial_low.iloc[cycle - 1]["end_s"])
            load_end = float(upper.iloc[cycle - 1]["start_s"])
            unload_start = float(upper.iloc[cycle - 1]["end_s"])
            unload_end = float(initial_low.iloc[cycle]["start_s"])
            loading = utm[(utm["Time"] >= load_start) & (utm["Time"] <= load_end)].sort_values("Displacement").drop_duplicates("Displacement")
            unloading = utm[(utm["Time"] >= unload_start) & (utm["Time"] <= unload_end)].sort_values("Displacement").drop_duplicates("Displacement")
            force_loading = np.interp(grid, loading["Displacement"], loading["Force"] * 1000.0)
            force_unloading = np.interp(grid, unloading["Displacement"], unloading["Force"] * 1000.0)
            all_mechanical.append({
                "sample": sample,
                "cycle": cycle,
                "absolute_force_loop_area_mJ": float(np.trapezoid(np.abs(force_loading - force_unloading), grid)),
                "signed_force_loop_area_mJ": float(np.trapezoid(force_loading - force_unloading, grid)),
                "maximum_force_N": float(max(loading["Force"].max(), unloading["Force"].max()) * 1000.0),
            })
    (OUTPUT / "qc_audit.json").write_text(json.dumps(audit, indent=2))
    pd.DataFrame([{k: v for k, v in row.items() if k != "lag_candidates"} for row in audit]).to_csv(
        OUTPUT / "qc_audit.csv", index=False
    )
    plateau_frame = pd.concat(all_plateaus, ignore_index=True)
    cycle_frame = pd.concat(all_cycles, ignore_index=True)
    specimen_frame = pd.DataFrame(all_specimens)
    mechanical_frame = pd.DataFrame(all_mechanical)
    plateau_frame.to_csv(OUTPUT / "plateau_metrics.csv", index=False)
    cycle_frame.to_csv(OUTPUT / "cycle_metrics.csv", index=False)
    specimen_frame.to_csv(OUTPUT / "specimen_metrics.csv", index=False)
    mechanical_frame.to_csv(OUTPUT / "mechanical_qc_metrics.csv", index=False)
    timing_frame = pd.DataFrame(all_timing)
    timing_frame.to_csv(OUTPUT / "alignment_sensitivity.csv", index=False)

    numeric_columns = [
        "median_MAH_pp", "median_absolute_area_pp_mm", "median_signed_area_pp_mm",
        "median_3mm_response_pct", "cycle1_3mm_response_pct", "cycle5_3mm_response_pct",
        "change_3mm_cycle1_to_5_pp", "relative_change_3mm_cycle1_to_5_pct",
        "median_recovery_0p1mm_pct", "median_plateau_noise_pct_R",
    ]
    group_rows = []
    for metric in numeric_columns:
        values = specimen_frame[metric].dropna().to_numpy()
        group_rows.append({
            "metric": metric,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        })
    pd.DataFrame(group_rows).to_csv(OUTPUT / "group_summary.csv", index=False)


if __name__ == "__main__":
    main()
