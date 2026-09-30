# S-PA-CBB

Official research code, processed curve records, and evaluation outputs for:

> **From Physics to Selective Generation: Conditional Brownian Bridges for Sparse S–N Curve Reconstruction and Fatigue Decision-Making**

S-PA-CBB reconstructs a complete probabilistic stress–life (S–N) curve from only two to four observed fatigue tests. The method first fits an interpretable Basquin state, diagnoses whether that state is sufficient, and applies a ControlNet-style conditional Brownian-bridge residual only to curves that benefit from correction. The resulting posterior supports fatigue-strength inversion, service-life and damage calculations, and sequential experiment selection.

## Repository contents

- `main_model.py` — the single self-contained reference script for validation-gated selective S-PA-CBB correction and evaluation.
- `am2022_curves.json` and `external_data/` — harmonized curve-level JSON plus source metadata.
- `strict_splits/` — fixed AM2022 source/alloy-held-out splits and LODO summary.
- `final_protocol/` — compact, publication-facing metrics, tables, and selected plots.
- [`DATA.md`](DATA.md) — official dataset links, licenses, conversion commands, and provenance notes.

## Environment

Python 3.10 or later is recommended. Install the common dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch should be installed with the wheel appropriate for the local accelerator. The checkpoint loader includes compatibility handling for models produced on Ascend NPU systems.

## Data preparation

The repository contains the harmonized curve records used for the reported experiments. Official source downloads, licenses, eligibility rules and provenance are documented in [`DATA.md`](DATA.md). Source identifiers are retained in every converted record, and all provided partitions are curve-level rather than point-level.

## Main experimental pipeline

The public code release intentionally exposes one entry point:

```bash
python main_model.py --help
```

`main_model.py` reconstructs the physical state from sparse observations, fits the validation-only selective gate, applies the residual-mean correction while retaining posterior deviations, and writes the aligned metrics and optional LaTeX table. It is self-contained and does not import private project modules.

## Result assets

The repository contains compact numerical results and selected result assets under `final_protocol/`. Plot-generation scripts and the paper draft are intentionally excluded from the public release.

## Reproducibility notes

- Sparse masks are deterministic functions of curve identifiers, budgets, and declared seeds.
- Model selection and the selective-correction gate use validation data only.
- Test curves remain untouched until final reporting.
- Raw database archives and trained checkpoints are intentionally not mirrored in Git. Official data links and licenses are recorded in [`DATA.md`](DATA.md).

## Citation

The article is under preparation. Until bibliographic details are assigned, please cite this repository using [`CITATION.cff`](CITATION.cff).

## Contact

Corresponding author: Hubin Yang, School of Information Science and Engineering, Lanzhou University.
