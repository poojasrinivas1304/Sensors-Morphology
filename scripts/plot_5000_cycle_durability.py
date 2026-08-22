from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "frozen_inputs" / "durability"
OUTPUT = ROOT / "generated" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUTPUT / "Extended_5000_cycle_durability.png"
OUT_PDF = OUTPUT / "Extended_5000_cycle_durability.pdf"

colors = ["#6FA8DC", "#F39C73", "#7FB77E", "#9A86C8", "#D6A84B"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})

traces = pd.read_csv(ANALYSIS / "aligned_traces_5000.csv")
metrics = pd.read_csv(ANALYSIS / "specimen_metrics_5000.csv")

records = {}
for i in range(1, 6):
    part = traces[traces["sample"] == i]
    records[i] = (part["time_s"].to_numpy(), part["cycle"].to_numpy(),
                  part["dR_R0_pct"].to_numpy(), part["segment"].to_numpy())

fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=False)
outer = GridSpec(1, 2, figure=fig, width_ratios=[1.68, 1.0], wspace=0.23,
                 left=0.065, right=0.985, top=0.935, bottom=0.085)
left = GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[0], hspace=0.08)
right = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1],
                                height_ratios=[1.25, 1, 1], hspace=0.40)

left_axes = []
for i in range(1, 6):
    ax = fig.add_subplot(left[i - 1], sharex=left_axes[0] if left_axes else None)
    left_axes.append(ax)
    t, cycle, y, segment = records[i]
    for seg in np.unique(segment):
        take = segment == seg
        if np.sum(take) > 1:
            ax.plot(cycle[take], y[take], color=colors[i - 1], lw=0.28,
                    rasterized=True)
    lo, hi = np.nanpercentile(y, [0.2, 99.8])
    pad = max(1.0, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(1, 5000)
    ax.text(0.015, 0.80, f"Sensor {i}", transform=ax.transAxes,
            color=colors[i - 1], fontweight="bold")
    ax.tick_params(labelsize=7.5, length=3)
    if i < 5:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Cycle number")
    if i == 3:
        ax.set_ylabel(r"Normalized resistance change, $\Delta R/R_0$ (\%)")
left_axes[0].text(-0.10, 1.16, "(a)", transform=left_axes[0].transAxes,
                  fontsize=11, fontweight="bold")
left_axes[0].set_title("Complete 5000-cycle electrical records", pad=7,
                      fontsize=10, fontweight="bold")

ax_b = fig.add_subplot(right[0])
for i in range(1, 6):
    _, cycle, y, _ = records[i]
    sel = (cycle >= 4995) & (cycle <= 5000)
    ax_b.plot(cycle[sel], y[sel], color=colors[i - 1], lw=0.85,
              label=f"Sensor {i}")
ax_b.set_xlim(4995, 5000)
ax_b.set_xlabel("Cycle number")
ax_b.set_ylabel(r"$\Delta R/R_0$ (\%)")
ax_b.set_title("Late-stage waveform", fontsize=10, fontweight="bold", pad=5)
ax_b.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02),
            fontsize=7.2, handlelength=1.5, columnspacing=0.8)
ax_b.text(-0.18, 1.08, "(b)", transform=ax_b.transAxes,
          fontsize=11, fontweight="bold")

ax_c = fig.add_subplot(right[1])
for j, row in metrics.iterrows():
    vals = [row["early_A80_pct_points"], row["late_A80_pct_points"]]
    ax_c.plot([0, 1], vals, color=colors[j], lw=1.0, alpha=0.85)
    ax_c.scatter([0, 1], vals, color=colors[j], s=27, zorder=3,
                 edgecolor="white", linewidth=0.45)
means = [metrics["early_A80_pct_points"].mean(), metrics["late_A80_pct_points"].mean()]
sds = [metrics["early_A80_pct_points"].std(ddof=1), metrics["late_A80_pct_points"].std(ddof=1)]
ax_c.errorbar([0, 1], means, yerr=sds, fmt="D", color="black", mfc="white",
              ms=5, lw=1.1, capsize=3, label=r"Mean $\pm$ SD")
ax_c.set_xticks([0, 1], ["Early\n(1--50)", "Late\n(4951--5000)"])
ax_c.set_ylabel(r"Median $A_{80}$ (percentage points)")
ax_c.set_title("Within-cycle response retention", fontsize=10,
               fontweight="bold", pad=5)
ax_c.legend(loc="upper right", fontsize=7.3)
ax_c.text(-0.18, 1.08, "(c)", transform=ax_c.transAxes,
          fontsize=11, fontweight="bold")

ax_d = fig.add_subplot(right[2])
for j, row in metrics.iterrows():
    vals = [row["early_centre_pct"], row["late_centre_pct"]]
    ax_d.plot([0, 1], vals, color=colors[j], lw=1.0, alpha=0.85)
    ax_d.scatter([0, 1], vals, color=colors[j], s=27, zorder=3,
                 edgecolor="white", linewidth=0.45)
means = [metrics["early_centre_pct"].mean(), metrics["late_centre_pct"].mean()]
sds = [metrics["early_centre_pct"].std(ddof=1), metrics["late_centre_pct"].std(ddof=1)]
ax_d.errorbar([0, 1], means, yerr=sds, fmt="D", color="black", mfc="white",
              ms=5, lw=1.1, capsize=3)
ax_d.axhline(0, color="#999999", lw=0.7)
ax_d.set_xticks([0, 1], ["Early\n(1--50)", "Late\n(4951--5000)"])
ax_d.set_ylabel(r"Median cycle centre (\%)")
ax_d.set_title("Resistance-level evolution", fontsize=10,
               fontweight="bold", pad=5)
ax_d.text(-0.18, 1.08, "(d)", transform=ax_d.transAxes,
          fontsize=11, fontweight="bold")

for ax in [*left_axes, ax_b, ax_c, ax_d]:
    ax.grid(False)

fig.savefig(OUT_PNG, dpi=350, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(OUT_PNG)
print(OUT_PDF)
