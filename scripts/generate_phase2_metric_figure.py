#!/usr/bin/env python3
"""Calculate and plot measured-only Phase 2 screening metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "frozen_inputs" / "phase1_3" / "phase2_per_cycle_metrics.json"
OUTPUT_DIR = PROJECT_DIR / "generated" / "figures"

SAMPLE_MAP = {
    19: ("Rectilinear", 40),
    20: ("Rectilinear", 60),
    21: ("Rectilinear", 80),
    22: ("Grid", 40),
    23: ("Grid", 60),
    24: ("Grid", 80),
    25: ("Honeycomb", 40),
    26: ("Honeycomb", 60),
    27: ("Honeycomb", 80),
}

INFILL_COLORS = {
    40: "#6BA6D1",
    60: "#ED8F6B",
    80: "#7AB07D",
}

PATTERN_MARKERS = {
    "Rectilinear": "o",
    "Grid": "s",
    "Honeycomb": "D",
}

MINIMUM_OBSERVATIONS_PER_CYCLE = 19

PANELS = [
    ("Late_A80_pct", "(a) Late-stage central 80% excursion", r"Late $A_{80}$ (percentage points)"),
    (
        "Resistance_evolution_pp",
        "(b) Resistance evolution",
        "Early-to-late evolution\n(percentage points)",
    ),
    (
        "Late_A80_CV_pct",
        "(c) Late-stage within-sensor excursion variability",
        r"Late $A_{80}$ CV (%)",
    ),
]


def load_cycle_rows(path: Path) -> list[dict[str, float | int | str]]:
    with path.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    header = matrix[0]
    return [dict(zip(header, row, strict=True)) for row in matrix[1:]]


def calculate_metrics(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    output: list[dict[str, float | int | str]] = []

    for sample, (pattern, infill) in SAMPLE_MAP.items():
        sample_rows = [row for row in rows if int(row["Sample"]) == sample]
        sample_rows.sort(key=lambda row: int(row["CycleInteger"]))
        eligible_rows = [
            row for row in sample_rows
            if int(row["N_points"]) >= MINIMUM_OBSERVATIONS_PER_CYCLE
        ]

        early = [
            row for row in eligible_rows
            if 1 <= int(row["CycleInteger"]) <= 50
        ]
        late = [
            row for row in eligible_rows
            if 451 <= int(row["CycleInteger"]) <= 500
        ]
        if not early or not late:
            raise ValueError(
                f"Sample {sample} lacks an early or late measured-data window."
            )

        late_a80_values = np.asarray(
            [float(row["A80_pct"]) for row in late],
            dtype=float,
        )
        early_centres = np.asarray(
            [float(row["Median_pct"]) for row in early],
            dtype=float,
        )
        late_centres = np.asarray(
            [float(row["Median_pct"]) for row in late],
            dtype=float,
        )

        output.append(
            {
                "Sample": sample,
                "Pattern": pattern,
                "Infill_pct": infill,
                "Valid_cycles_total": len(eligible_rows),
                "Valid_cycles_early": len(early),
                "Valid_cycles_late": len(late),
                "Late_A80_pct": float(np.median(late_a80_values)),
                "Early_cycle_centre_pct": float(np.median(early_centres)),
                "Late_cycle_centre_pct": float(np.median(late_centres)),
                "Resistance_evolution_pp": float(
                    np.median(late_centres) - np.median(early_centres)
                ),
                "Late_A80_CV_pct": float(
                    100
                    * np.std(late_a80_values, ddof=1)
                    / np.mean(late_a80_values)
                ),
            }
        )

    return output


def write_metrics(rows: list[dict[str, float | int | str]]) -> Path:
    path = OUTPUT_DIR / "Phase2_specimen_level_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_figure(rows: list[dict[str, float | int | str]]) -> tuple[Path, Path]:
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

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.65))
    x_values = np.asarray([1, 2, 3, 5, 6, 7, 9, 10, 11])
    tick_labels = [f"{row['Infill_pct']}%" for row in rows]

    for axis, (metric, title, ylabel) in zip(axes, PANELS, strict=True):
        if metric == "Resistance_evolution_pp":
            axis.axhline(0, color="#8C8C8C", linewidth=0.8, zorder=0)

        for x_value, row in zip(x_values, rows, strict=True):
            pattern = str(row["Pattern"])
            infill = int(row["Infill_pct"])
            axis.scatter(
                x_value,
                float(row[metric]),
                s=55,
                marker=PATTERN_MARKERS[pattern],
                facecolor=INFILL_COLORS[infill],
                edgecolor="#36414A",
                linewidth=0.8,
                zorder=3,
            )

        axis.axvline(4, color="#D4D4D4", linewidth=0.7)
        axis.axvline(8, color="#D4D4D4", linewidth=0.7)
        axis.set_xlim(0.4, 11.6)
        axis.set_xticks(x_values, tick_labels)
        axis.tick_params(axis="x", labelsize=8.5)
        for centre, pattern in zip(
            [2, 6, 10],
            ["Rectilinear", "Grid", "Honeycomb"],
            strict=True,
        ):
            axis.text(
                centre,
                -0.17,
                pattern,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold", pad=7)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)

    legend_handles = [
        Line2D([0], [0], color=INFILL_COLORS[40], linewidth=4, label="40%"),
        Line2D([0], [0], color=INFILL_COLORS[60], linewidth=4, label="60%"),
        Line2D([0], [0], color=INFILL_COLORS[80], linewidth=4, label="80%"),
    ]

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )
    figure.tight_layout(rect=(0.015, 0.08, 0.995, 0.88))

    png_path = OUTPUT_DIR / "Phase2_specimen_level_metrics.png"
    pdf_path = OUTPUT_DIR / "Phase2_specimen_level_metrics.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cycle_rows = load_cycle_rows(INPUT_PATH)
    metrics = calculate_metrics(cycle_rows)
    csv_path = write_metrics(metrics)
    png_path, pdf_path = make_figure(metrics)
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
