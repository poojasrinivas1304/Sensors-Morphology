# Replication-Aware Design Selection and Validation of 3D-Printed Conductive TPU Bending Sensors

> Data, analysis code and reproducibility records accompanying the manuscript
> *Replication-aware design selection and validation of 3D-printed conductive
> TPU bending sensors*.

---

## Overview

This repository contains the frozen analysis inputs, specimen-level results,
quality-control records and scripts used to study fused-filament-fabricated
conductive-TPU bending sensors. The experiments examined two complementary
design scales:

1. **Through-thickness architecture:** two or three conductive TPU layers
   combined with zero, one or two nonconductive backing TPU layers.
2. **In-plane morphology:** rectilinear, grid and honeycomb infill at 40%,
   60% and 80% nominal infill.

The work followed a sequential design comprising architecture screening,
exploratory morphology screening, independent confirmation, bending-orientation
and textile feasibility tests, displacement-programme validation,
quasi-static electrical hysteresis and 5000-cycle durability testing.

Each independently fabricated sensor was treated as the experimental unit.
Cycles within a sensor are repeated observations and were not treated as
independent replicates.

---

## Study Design

| Stage | Design | Independent sensors | Purpose |
|---|---|---:|---|
| Architecture | CL2/CL3 crossed with BL0/BL1/BL2 | 3 per architecture | Balanced architecture comparison |
| Morphology screen | Rectilinear, grid and honeycomb at 40%, 60% and 80% | 1 per condition | Exploratory candidate nomination |
| Finalist confirmation | Rectilinear 80% and grid 80% | 3 per condition | Independent replication |
| Outward-bending pilot | Rectilinear 80%, grid 80% and solid 100% | 1 per condition | Descriptive orientation feasibility |
| Textile integration | Three designs, inward and outward | 1 per design-orientation combination | Feasibility evidence |
| Progressive displacement | Selected grid 80% design | 5 retained sensors | Ordered low-medium-high programme |
| Fixed displacement | Low, medium and high cohorts | 5 per condition | Independent condition summaries |
| Quasi-static hysteresis | Staircase loading-unloading protocol | 5 sensors | Stabilized branch separation |
| Extended durability | Medium programme, 5000 cycles | 5 sensors | Response retention and resistance evolution |

`CL` denotes the number of conductive TPU layers and `BL` denotes the number
of nonconductive backing TPU layers.

---

## Main Findings

- **CL2-BL1** was selected as the baseline architecture because it combined a
  moderate late-stage response with the lowest mean within-sensor response
  variability and a nominal thickness of 0.60 mm.
- **Grid 80%** was selected for subsequent validation because the documented
  engineering hierarchy prioritized repeatability over maximum response. The
  selection is not presented as statistical or universal superiority.
- The replicated grid 80% condition had a mean late-stage central 80%
  excursion of **5.63 ± 3.08 percentage points** and a mean within-sensor
  excursion coefficient of variation of **15.87 ± 3.77%**.
- Quasi-static electrical hysteresis, expressed as the mean absolute
  loading-unloading difference, was **2.38 ± 0.92 percentage points** across
  five independently fabricated sensors.
- All five durability sensors retained periodic electrical responses through
  5000 cycles. Mean response retention was **54.2 ± 17.3%**, indicating
  continued responsiveness together with substantial conditioning and
  attenuation.
- Six independently fabricated fabric-integrated sensors provided feasibility
  evidence of bending responsiveness in both tested orientations.

The study does not report a gauge factor or claim a universal optimum because
local strain or curvature was not directly measured and some validation
programmes covaried displacement range and cycle frequency.

---

## Repository Structure

```text
conductive-tpu-sensor-optimization/
├── README.md
├── LICENSE
├── requirements.txt
├── audit/
│   └── Phase1_factorial_and_reproducibility_audit.xlsx
├── frozen_inputs/
│   ├── phase1_3/
│   ├── progressive/
│   ├── fixed_conditions/
│   ├── hysteresis/
│   └── durability/
├── raw_data/
│   ├── phase1_3/                   # Canonical architecture, morphology, confirmation and fabric workbooks
│   ├── progressive/                # ESP32 and retained UTM progressive records
│   ├── fixed_conditions/           # Low-, medium- and high-condition records
│   ├── hysteresis/                 # Paired ESP32 and UTM staircase records
│   └── durability/                 # Five-sensor 5000-cycle workbook
└── scripts/
    ├── phase1_factorial_analysis.py
    ├── analyze_hysteresis.py
    ├── plot_hysteresis_protocol_and_branches.py
    ├── plot_5000_cycle_durability.py
    ├── generate_phase1_metric_figure.py
    ├── generate_phase2_metric_figure.py
    ├── generate_phase3_figures.py
    └── calculate_fabric_metrics.py
```

---

## Data Dictionary and Primary Metrics

The specimen-specific normalized resistance response is

```text
dR_R0 = (R - R0) / R0
```

where one reference resistance, `R0`, was retained throughout each complete
recording. The principal within-cycle response metric is the central 80%
excursion:

```text
A80 = P90 - P10
```

Primary 500-cycle endpoint windows:

- Early: Cycles 1-50
- Late: Cycles 451-500

Primary 5000-cycle durability windows:

- Early: Cycles 1-50
- Late: Cycles 4951-5000

The audit workbook contains the detailed data dictionary, canonical specimen
inventory, Phase 1 model coefficients, planned contrasts, model diagnostics
and leave-one-sensor-out sensitivity results.

---

## Setup

### 1. Download and unpack the repository archive

Download the archived deposit, unpack it, and change into the extracted
repository directory.

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```text
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Reproducing the Analysis

The Phase 1 factorial analysis uses one row per independently fabricated
sensor and applies ordinary least squares with HC3 heteroscedasticity-consistent
standard errors. Two-sided 95% confidence intervals use a Student-t critical
value with 12 residual degrees of freedom. The planned contrasts are
unadjusted for multiple comparisons. Run:

```bash
python scripts/phase1_factorial_analysis.py
```

The remaining scripts reproduce the corresponding specimen-level calculations
and manuscript figures. Scripts use repository-relative input and output paths;
generated results are written under `generated/`.

No smoothing, gap filling or silent specimen substitution should be introduced
when reproducing the quantitative analysis. Any alternative processing rule
should be reported as a sensitivity analysis.

---

## Data Provenance and Inclusion Rules

- The approved replacement record for Sample 13 is the canonical Phase 1
  record.
- The original Sample 4 record is retained as the canonical manuscript record.
- Phase 2 is exploratory because it contains one sensor per morphology-infill
  condition.
- Phase 2 screening specimens were not pooled with the independently printed
  Phase 3 confirmation specimens.
- One progressive-displacement record with documented incorrect specimen
  placement was excluded before the primary five-sensor analysis; the
  inclusion audit is retained in `frozen_inputs/progressive/`.
- Cycles failing the documented observation-completeness rule remain in the
  audit but are excluded from primary cycle-level summaries.

### Data coverage and known omissions

- Original timestamped pre-segmentation files and raw ADC streams were not
  retained.
- Mechanical force/displacement records were not retained for the original
  Phase 1--3 experiments.
- Electrical records are provided for all six progressive-displacement
  specimens and retained UTM exports for five; the primary analysis uses the
  five records meeting the documented specimen-placement criterion.
- Electrical records are provided for all 15 fixed-condition sensors. Paired
  UTM exports are available for the low- and high-displacement cohorts but
  were unavailable for the medium-displacement cohort.
- Paired ESP32 and UTM exports are provided for all five quasi-static
  hysteresis sensors, and the retained durability workbook contains all five
  5000-cycle resistance records.

---

## Citation

If you use these data or scripts, please cite the associated manuscript and
this repository. The final journal citation and DOI will be added after
publication.

```bibtex
@article{gurram_conductive_tpu_sensors,
  title   = {Replication-aware design selection and validation of 3D-printed
             conductive TPU bending sensors},
  author  = {Gurram, Pooja and Elgendi, Mohamed},
  journal = {To be updated},
  year    = {2026},
  doi     = {To be updated}
}
```

---

## Funding

This work was supported by Khalifa University under grant FSU-2025-001 and by
the Healthcare Engineering Innovation Group, Khalifa University of Science
and Technology.

---

## Authors

- **Pooja Gurram** - sensor fabrication, experiments, data collection,
  analysis, figures and manuscript preparation
- **Mohamed Elgendi** - supervision, conceptualization, interpretation and
  critical manuscript revision

---

## Licence

Analysis code is released under the MIT Licence. Data are released under the
Creative Commons Attribution 4.0 International licence unless otherwise
specified. The corresponding licence files are included in the repository
root.
