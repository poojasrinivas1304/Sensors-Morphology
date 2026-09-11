#!/usr/bin/env python3
"""Generate the Phase 1 specimen-level manuscript figure without MATLAB."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "frozen_inputs" / "phase1_3" / "Phase1_specimen_level_metrics.csv"
OUTPUT_DIR = ROOT / "generated" / "figures"

ARCHITECTURE_ORDER = [
    "CL2-BL0",
    "CL2-BL1",
    "CL2-BL2",
    "CL3-BL0",
    "CL3-BL1",
    "CL3-BL2",
]

BL_COLORS = {
    "BL0": "#6BA6D1",
    "BL1": "#ED8F6B",
    "BL2": "#7AB07D",
}

CL_MARKERS = {
    "CL2": "o",
    "CL3": "s",
}

PANELS = [
    (
        "Late_A80_pct",
        "(a) Late-stage central 80% excursion",
        r"Late $A_{80}$ (percentage points)",
    ),
    (
        "Resistance_evolution_pp",
        "(b) Resistance evolution",
        "Early-to-late evolution (percentage points)",
    ),
    (
        "Late_A80_CV_pct",
        "(c) Late-stage excursion variability",
        r"Late $A_{80}$ CV (%)",
    ),
]


def load_metrics(path: Path) -> list[dict[str, object]]:
    """Load the audited specimen-level metric table."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "Sample": int(row["Sample"]),
                    "Architecture": row["Architecture"],
                    "Late_A80_pct": float(row["Late_A80_pct"]),
                    "Resistance_evolution_pp": float(
                        row["Resistance_evolution_pp"]
                    ),
                    "Late_A80_CV_pct": float(row["Late_A80_CV_pct"]),
                }
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metrics(METRICS_PATH)
    if len(rows) != 18:
        raise ValueError(f"Expected 18 specimens; found {len(rows)}.")

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

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 4.05))
    jitter = np.array([-0.13, 0.0, 0.13])

    for panel_index, (metric, title, ylabel) in enumerate(PANELS):
        axis = axes[panel_index]

        if panel_index == 1:
            axis.axhline(0, color="#8C8C8C", linewidth=0.8, zorder=0)

        for architecture_index, architecture in enumerate(
            ARCHITECTURE_ORDER, start=1
        ):
            group = [
                row for row in rows
                if row["Architecture"] == architecture
            ]
            group.sort(key=lambda row: int(row["Sample"]))
            if len(group) != 3:
                raise ValueError(
                    f"{architecture} should contain three specimens."
                )

            values = np.array(
                [float(row[metric]) for row in group],
                dtype=float,
            )
            cl_label, bl_label = architecture.split("-")
            color = BL_COLORS[bl_label]
            marker = CL_MARKERS[cl_label]
            x_positions = architecture_index + jitter

            axis.scatter(
                x_positions,
                values,
                s=46,
                marker=marker,
                facecolor=color,
                edgecolor="#36414A",
                linewidth=0.7,
                zorder=3,
            )

            group_mean = float(np.mean(values))
            group_sd = float(np.std(values, ddof=1))
            axis.errorbar(
                architecture_index,
                group_mean,
                yerr=group_sd,
                fmt="none",
                ecolor="#222222",
                elinewidth=1.2,
                capsize=3.5,
                capthick=1.2,
                zorder=2,
            )
            axis.plot(
                [architecture_index - 0.18, architecture_index + 0.18],
                [group_mean, group_mean],
                color="#111111",
                linewidth=1.8,
                zorder=4,
            )

        axis.set_xlim(0.5, 6.5)
        axis.set_xticks(range(1, 7), ARCHITECTURE_ORDER)
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold", pad=7)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#B8B8B8",
            markeredgecolor="#36414A",
            markersize=6.5,
            label="CL2",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#B8B8B8",
            markeredgecolor="#36414A",
            markersize=6.5,
            label="CL3",
        ),
        Line2D([0], [0], color=BL_COLORS["BL0"], linewidth=4, label="BL0"),
        Line2D([0], [0], color=BL_COLORS["BL1"], linewidth=4, label="BL1"),
        Line2D([0], [0], color=BL_COLORS["BL2"], linewidth=4, label="BL2"),
    ]

    figure.suptitle(
        "Specimen-level Phase 1 response metrics",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2,
    )
    figure.tight_layout(rect=(0.015, 0.02, 0.995, 0.855))

    png_path = OUTPUT_DIR / "Phase1_specimen_level_metrics.png"
    pdf_path = OUTPUT_DIR / "Phase1_specimen_level_metrics.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
