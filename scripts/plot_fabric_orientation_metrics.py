"""Reproduce the descriptive fabric-orientation metric figure (main Fig. 7)."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "generated" / "fabric" / "fabric_metrics.json"
OUTPUT = ROOT / "generated" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

MORPHOLOGIES = [
    ("Grid 80%", "Grid80_standard", "Grid80_concave"),
    ("Rectilinear 80%", "Rect80_standard", "Rect80_concave"),
    ("Solid 100%", "Rect100_convex", "Rect100_concave"),
]
METRIC_SPECS = [
    ("late_median_a80_pct_points",
     "Late-stage median $A_{80}$\n(percentage points)"),
    ("resistance_evolution_pct_points",
     "Resistance evolution\n(percentage points)"),
    ("late_amplitude_cv_pct", "Late-stage $A_{80}$ CV (%)"),
]


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.tick_params(direction="in", top=True, right=True, width=0.8, length=4)
    ax.grid(False)


metrics = {item["record"]: item for item in json.loads(INPUT.read_text())}
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
colors = ["#4C78A8", "#59A14F", "#B279A2"]
for ax, (key, ylabel), panel in zip(axes, METRIC_SPECS, ["a", "b", "c"]):
    for (title, inward, outward), color in zip(MORPHOLOGIES, colors):
        ax.scatter(
            [0, 1],
            [metrics[inward][key], metrics[outward][key]],
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=title,
            zorder=2,
        )
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1], ["Inward", "Outward"])
    ax.set_ylabel(ylabel, fontsize=10)
    ax.text(-0.15, 1.04, f"({panel})", transform=ax.transAxes,
            fontsize=11, weight="bold")
    style_axis(ax)
axes[0].legend(frameon=False, fontsize=8.5, loc="best")
fig.suptitle(
    "Descriptive results for independent fabric-integrated sensors",
    fontsize=14,
    weight="semibold",
    y=1.01,
)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUTPUT / "Fabric_orientation_metrics.png",
            dpi=600, bbox_inches="tight")
plt.close(fig)
