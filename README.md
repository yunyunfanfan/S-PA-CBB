# S-PA-CBB

Official research code, processed curve records, evaluation outputs, and manuscript source for:

> **From Physics to Selective Generation: Conditional Brownian Bridges for Sparse S–N Curve Reconstruction and Fatigue Decision-Making**

S-PA-CBB reconstructs a complete probabilistic stress–life (S–N) curve from only two to four observed fatigue tests. The method first fits an interpretable Basquin state, diagnoses whether that state is sufficient, and applies a ControlNet-style conditional Brownian-bridge residual only to curves that benefit from correction. The resulting posterior supports fatigue-strength inversion, service-life and damage calculations, and sequential experiment selection.

## Repository contents

- Root-level Python files — data conversion, model training, calibration, evaluation, ablation, downstream-task, and figure-generation code.
- `am2022_curves.json` and `external_data/` — harmonized curve-level JSON plus source metadata.
- `strict_splits/` — fixed AM2022 source/alloy-held-out splits and LODO summary.
- `final_protocol/` — compact, publication-facing metrics, tables, and selected plots.
- `paper_elsevier_draft/` — Elsevier CAS LaTeX source, bibliography, tables, figures, and a compiled PDF.
- [`DATA.md`](DATA.md) — official dataset links, licenses, conversion commands, and provenance notes.

## Environment

Python 3.10 or later is recommended. Install the common dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch should be installed with the wheel appropriate for the local accelerator. The training scripts run on CPU, CUDA, or Ascend NPU where the corresponding PyTorch backend is available.

## Data preparation

The repository already contains the harmonized curve records used for the reported experiments. To rebuild them from the official sources, download the source files listed in [`DATA.md`](DATA.md), then run:

```bash
python convert_am2022.py --help
python convert_external_datasets.py --help
python build_multidataset.py --help
python make_strict_splits.py --help
python make_lodo_manifests.py --help
```

Source identifiers are retained in every converted record. The conversion scripts normalize units, reject ineligible records, and create curve-level rather than point-level partitions to prevent leakage.

## Main experimental pipeline

The principal model and evaluation entry points are:

```bash
# Conditional Brownian-bridge model
python control_score_bridge.py --help

# Validation-gated selective correction
python selective_two_stage_correction.py --help
python promote_selective_main_model.py --help

# External-domain and baseline evaluation
python external_control_eval.py --help
python external_baseline_benchmark.py --help

# Fatigue-strength inversion and active experiment selection
python downstream_fatigue_strength.py --help
python downstream_active_selection.py --help
```

All stochastic experiments expose explicit seed arguments. The paper uses three reverse Brownian-bridge steps for the core comparison and reports independent sensitivity sweeps over reverse-step and ensemble counts.

## Reproduce publication assets

Publication plots and tables are generated from saved result JSON/CSV files:

```bash
python plot_final_protocol.py --help
python plot_mechanism_case_studies.py --help
python plot_active_selection_case_studies.py --help
python make_publication_figures.py --help
```

Compile the manuscript from `manuscript/` with a standard TeX Live installation:

```bash
cd paper_elsevier_draft
latexmk -pdf main_v2.tex
```

## Reproducibility notes

- Sparse masks are deterministic functions of curve identifiers, budgets, and declared seeds.
- Model selection and the selective-correction gate use validation data only.
- Test curves remain untouched until final reporting.
- Raw database archives and trained checkpoints are intentionally not mirrored in Git. Official data links and licenses are recorded in [`DATA.md`](DATA.md).

## Citation

The article is under preparation. Until bibliographic details are assigned, please cite this repository using [`CITATION.cff`](CITATION.cff).

## Contact

Corresponding author: Hubin Yang, School of Information Science and Engineering, Lanzhou University.
