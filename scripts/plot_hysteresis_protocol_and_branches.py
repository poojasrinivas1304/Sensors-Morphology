from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
UTM = ROOT / "raw_data" / "hysteresis" / "utm" / "hysteresis1_1.csv"
ANALYSIS = ROOT / "generated" / "hysteresis"
PLATEAUS = ANALYSIS / "sample_1_utm_plateaus.csv"
PLATEAU_METRICS = ANALYSIS / "plateau_metrics.csv"
SPECIMEN_METRICS = ANALYSIS / "specimen_metrics.csv"
OUTPUT = ROOT / "generated" / "figures"


LOADING_COLOR = "#5B8DB8"
UNLOADING_COLOR = "#D98264"
SUMMARY_COLOR = "#6F8F72"
SHARED_LEVELS = np.array([0.5, 1.0, 1.5, 2.0, 2.5])


def summarize_branch(frame, phase):
    branch = frame[
        (frame["phase"] == phase)
        & (frame["displacement_mm"].isin(SHARED_LEVELS))
    ]
    summary = (
        branch.groupby("displacement_mm")["dR_R0_pct"]
        .agg(median="median", q1=lambda x: np.percentile(x, 25), q3=lambda x: np.percentile(x, 75))
        .reindex(SHARED_LEVELS)
    )
    return summary


def draw_protocol(ax, utm, plateaus):
    start = float(plateaus.iloc[0]["end_s"])
    end = float(plateaus.iloc[12]["end_s"])
    trace = utm[(utm["Time"] >= start) & (utm["Time"] <= end)].copy()
    trace["relative_time_s"] = trace["Time"] - start

    loop_plateaus = plateaus.iloc[1:13].copy()
    loop_plateaus["start_rel_s"] = loop_plateaus["start_s"] - start
    loop_plateaus["end_rel_s"] = loop_plateaus["end_s"] - start

    loading_end = float(loop_plateaus.iloc[5]["end_rel_s"])
    unloading_end = float(loop_plateaus.iloc[10]["end_rel_s"])
    ax.axvspan(0, loading_end, color="#DCEBF7", alpha=0.75, linewidth=0)
    ax.axvspan(loading_end, unloading_end, color="#F8E3D7", alpha=0.75, linewidth=0)
    ax.axvspan(unloading_end, end - start, color="#E8E8E8", alpha=0.8, linewidth=0)

    for row in loop_plateaus.itertuples(index=False):
        ax.axvspan(row.end_rel_s - 3.0, row.end_rel_s, color="#2F4858", alpha=0.15, linewidth=0)

    ax.plot(trace["relative_time_s"], trace["Displacement"], color="#173F5F", lw=1.55)
    ax.set_xlim(0, end - start)
    ax.set_ylim(0, 3.18)
    ax.set_xlabel("Elapsed time within staircase loop (s)")
    ax.set_ylabel("Programmed displacement (mm)")
    ax.set_yticks([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.annotate("Loading", xy=(loading_end * 0.52, 3.08), ha="center", va="center",
                weight="bold", color="#315F7D", fontsize=9.5)
    ax.annotate("Unloading", xy=((loading_end + unloading_end) / 2, 3.08),
                ha="center", va="center", weight="bold", color="#A65F3C", fontsize=9.5)
    ax.annotate("Inter-loop recovery", xy=((unloading_end + end - start) / 2, 2.95),
                ha="center", va="center", weight="bold", color="#555555", fontsize=9)
    ax.annotate("Final 3 s used for plateau median", xy=(9.5, 0.52), xytext=(42, 0.28),
                ha="center", arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
                fontsize=8.5)
    ax.text(-0.08, 1.04, "(a)", transform=ax.transAxes, fontsize=11, weight="bold")


def draw_sensor_panel(ax, specimen_frame, sensor_number, panel_letter):
    sensor = specimen_frame[specimen_frame["sample"] == sensor_number]
    loading = summarize_branch(sensor, "Loading")
    unloading = summarize_branch(sensor, "Unloading")

    ax.plot(SHARED_LEVELS, loading["median"], color=LOADING_COLOR, marker="o", ms=4.2,
            lw=1.55, label="Loading")
    ax.fill_between(SHARED_LEVELS, loading["q1"], loading["q3"], color=LOADING_COLOR,
                    alpha=0.18, linewidth=0)
    ax.plot(SHARED_LEVELS, unloading["median"], color=UNLOADING_COLOR, marker="s", ms=4.0,
            lw=1.55, ls="--", label="Unloading")
    ax.fill_between(SHARED_LEVELS, unloading["q1"], unloading["q3"], color=UNLOADING_COLOR,
                    alpha=0.18, linewidth=0)
    ax.axhline(0, color="#777777", lw=0.65, zorder=0)
    ax.set_xlim(0.42, 2.58)
    ax.set_ylim(-15, 15)
    ax.set_xticks(SHARED_LEVELS)
    ax.set_title(f"Sensor {sensor_number}", fontsize=10.5, weight="bold", pad=5)
    ax.text(-0.14, 1.05, f"({panel_letter})", transform=ax.transAxes,
            fontsize=10.5, weight="bold")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    utm = pd.read_csv(UTM, skiprows=[1]).apply(pd.to_numeric, errors="coerce").dropna()
    plateaus = pd.read_csv(PLATEAUS)
    plateau_metrics = pd.read_csv(PLATEAU_METRICS)
    specimen_metrics = pd.read_csv(SPECIMEN_METRICS)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Calibri", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.linewidth": 0.85,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    fig = plt.figure(figsize=(13.2, 10.0))
    grid = GridSpec(3, 3, figure=fig, height_ratios=[0.88, 1.0, 1.0], hspace=0.38, wspace=0.24)
    protocol_ax = fig.add_subplot(grid[0, :])
    draw_protocol(protocol_ax, utm, plateaus)

    axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    axes.extend([fig.add_subplot(grid[2, i]) for i in range(2)])
    for ax, sensor_number, letter in zip(axes, range(1, 6), "bcdef"):
        draw_sensor_panel(ax, plateau_metrics, sensor_number, letter)

    for ax in axes[:3]:
        ax.set_xticklabels([])
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(r"Stabilized $\Delta R/R_0$ (%)")
    for ax in axes[3:]:
        ax.set_xlabel("Programmed displacement (mm)")
    for ax in (axes[1], axes[2], axes[4]):
        ax.set_yticklabels([])
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)

    summary_ax = fig.add_subplot(grid[2, 2])
    values = specimen_metrics["median_MAH_pp"].to_numpy()
    x = np.arange(1, 6)
    summary_ax.scatter(x, values, s=42, color=SUMMARY_COLOR, edgecolor="#334B36",
                       linewidth=0.6, zorder=3)
    summary_ax.errorbar(6.0, values.mean(), yerr=values.std(ddof=1), fmt="D", ms=5.5,
                        color="#222222", ecolor="#222222", capsize=4, lw=1.1,
                        label=r"Mean $\pm$ SD")
    summary_ax.set_xlim(0.5, 6.5)
    summary_ax.set_ylim(0, 5.0)
    summary_ax.set_xticks(np.arange(1, 7), ["1", "2", "3", "4", "5", "Mean"])
    summary_ax.set_xlabel("Sensor / group summary")
    summary_ax.set_ylabel("Median " + r"$H_{\mathrm{MAH}}$" + "\n(percentage points)")
    summary_ax.set_title("Specimen-level hysteresis", fontsize=10.5, weight="bold", pad=5)
    summary_ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    summary_ax.text(-0.10, 1.05, "(g)", transform=summary_ax.transAxes,
                    fontsize=10.5, weight="bold")

    for ax in [protocol_ax, *axes, summary_ax]:
        ax.grid(False)
        ax.tick_params(length=3.5, width=0.8)

    fig.savefig(OUTPUT / "Hysteresis_protocol_and_branches.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "Hysteresis_protocol_and_branches.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
