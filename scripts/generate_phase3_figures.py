#!/usr/bin/env python3
"""Generate manuscript figures for Phase 3 confirmation and orientation pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "frozen_inputs" / "phase1_3"
FIGURES = PROJECT / "generated" / "figures"
TRACE_DATA = AUDIT / "phase3_trace_data.npz"

REPEAT_COLORS = ["#6BA6D1", "#ED8F6B", "#7AB07D"]
CONDITION_COLORS = {
    "Rectilinear 80%": "#6BA6D1",
    "Grid 80%": "#ED8F6B",
    "Solid 100%": "#7AB07D",
}
CONDITIONS = ["Rectilinear 80%", "Grid 80%", "Solid 100%"]


def phase3_inward_traces():
    source = np.load(TRACE_DATA)
    rect = [(source[f"rect_in_{i}_x"], source[f"rect_in_{i}_y"]) for i in range(1, 4)]
    grid = [(source[f"grid_in_{i}_x"], source[f"grid_in_{i}_y"]) for i in range(1, 4)]
    solid = [(source[f"solid_in_{i}_x"], source[f"solid_in_{i}_y"]) for i in range(1, 4)]
    return {"Rectilinear 80%": rect, "Grid 80%": grid, "Solid 100%": solid}


def outward_traces():
    source = np.load(TRACE_DATA)
    return {
        "Rectilinear 80%": [(source["rect_out_x"], source["rect_out_y"])],
        "Grid 80%": [(source["grid_out_x"], source["grid_out_y"])],
        "Solid 100%": [(source["solid_out_x"], source["solid_out_y"])],
    }


def style_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.grid(False)


def make_trace_figure(data, outward=False):
    title = (
        "Conductive-outward orientation pilot"
        if outward
        else "Conductive-inward confirmation and historical solid reference"
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), sharex="row")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)

    full_y = []
    mid_y = []
    for condition in CONDITIONS:
        for x, y in data[condition]:
            full_y.extend(y[np.isfinite(y)])
            mask = (x >= 301) & (x < 306)
            mid_y.extend(y[mask])
    full_limits = np.nanpercentile(full_y, [0.05, 99.95])
    full_pad = 0.08 * max(1, full_limits[1] - full_limits[0])
    mid_limits = np.nanpercentile(mid_y, [0.2, 99.8])
    mid_pad = 0.10 * max(1, mid_limits[1] - mid_limits[0])

    labels = list("abcdef")
    for col, condition in enumerate(CONDITIONS):
        for row in (0, 1):
            ax = axes[row, col]
            style_axes(ax)
            for index, (x, y) in enumerate(data[condition]):
                color = (
                    CONDITION_COLORS[condition]
                    if outward
                    else REPEAT_COLORS[index]
                )
                mask = np.ones_like(x, dtype=bool) if row == 0 else ((x >= 301) & (x < 306))
                ax.plot(x[mask], y[mask], color=color, linewidth=0.75, alpha=0.92)
            if row == 0:
                ax.set_xlim(1, 500)
                ax.set_ylim(full_limits[0] - full_pad, full_limits[1] + full_pad)
                ax.set_title(condition, fontsize=12, fontweight="bold")
            else:
                ax.set_xlim(301, 306)
                ax.set_ylim(mid_limits[0] - mid_pad, mid_limits[1] + mid_pad)
                ax.set_xlabel("Cycle number", fontweight="bold")
            ax.text(
                -0.04,
                1.03,
                f"({labels[row * 3 + col]})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
            )

    axes[0, 0].set_ylabel(r"$\Delta R/R_{0}$ (%)", fontweight="bold")
    axes[1, 0].set_ylabel(r"$\Delta R/R_{0}$ (%)", fontweight="bold")

    if outward:
        handles = [
            Line2D([0], [0], color=CONDITION_COLORS[c], lw=2.0, label=c)
            for c in CONDITIONS
        ]
    else:
        handles = [
            Line2D([0], [0], color=REPEAT_COLORS[i], lw=2.0, label=f"Specimen {i + 1}")
            for i in range(3)
        ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.91), w_pad=1.1, h_pad=1.4)

    stem = "Phase3_outward_responses" if outward else "Phase3_inward_confirmation"
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_metric_figure():
    inward_json = json.loads((AUDIT / "phase3_metrics.json").read_text())
    outward_json = json.loads((AUDIT / "outward_metrics.json").read_text())

    inward = {
        "Rectilinear 80%": [x for x in inward_json["metrics"] if x["condition"] == "Rectilinear 80%"],
        "Grid 80%": [x for x in inward_json["metrics"] if x["condition"] == "Grid 80%"],
        "Solid 100%": [
            {**x, "condition": "Solid 100%"} for x in inward_json["historical_solid"]
        ],
    }
    outward = {x["condition"]: x for x in outward_json["metrics"]}

    panels = [
        ("late_A80_pct", "(a) Late-stage cyclic amplitude", r"Late $A_{80}$ (%)"),
        ("resistance_evolution_pp", "(b) Resistance evolution", "Early-to-late evolution\n(percentage points)"),
        ("late_A80_CV_pct", "(c) Late-stage amplitude variability", r"Late $A_{80}$ CV (%)"),
    ]
    x = np.arange(1, 4)
    jitter = np.array([-0.13, 0.0, 0.13])

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
    for ax, (metric, title, ylabel) in zip(axes, panels):
        style_axes(ax)
        if metric == "resistance_evolution_pp":
            ax.axhline(0, color="#999999", linewidth=0.8, zorder=0)
        for index, condition in enumerate(CONDITIONS):
            values = np.asarray([row[metric] for row in inward[condition]], dtype=float)
            color = CONDITION_COLORS[condition]
            ax.scatter(
                x[index] + jitter,
                values,
                s=46,
                color=color,
                edgecolor=np.asarray(plt.matplotlib.colors.to_rgb(color)) * 0.65,
                linewidth=0.8,
                zorder=3,
            )
            ax.errorbar(
                x[index],
                values.mean(),
                yerr=values.std(ddof=1),
                color="#222222",
                linewidth=1.1,
                capsize=5,
                fmt="none",
                zorder=2,
            )
            ax.plot([x[index] - 0.18, x[index] + 0.18], [values.mean()] * 2, color="#222222", linewidth=1.2)
            ax.scatter(
                x[index] + 0.29,
                outward[condition][metric],
                marker="D",
                s=58,
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
                zorder=4,
            )
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xticks(x, ["Rect.\n80%", "Grid\n80%", "Solid\n100%"])
        ax.set_xlim(0.55, 3.55)

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#777777", markeredgecolor="#555555", label="Conductive-inward, individual sensors"),
            Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#555555", label="Conductive-outward pilot (n = 1)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.6)
    fig.savefig(FIGURES / "Phase3_orientation_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Phase3_orientation_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    make_trace_figure(phase3_inward_traces(), outward=False)
    make_trace_figure(outward_traces(), outward=True)
    make_metric_figure()


if __name__ == "__main__":
    main()
