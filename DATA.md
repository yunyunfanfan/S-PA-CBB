# Data sources and provenance

The experiments use three public fatigue resources. This repository includes harmonized curve-level JSON for reproducibility; the authoritative source files remain with their original publishers.

| Repository file | Public resource | Official record | License reported by host |
|---|---|---|---|
| `am2022_curves.json` | FatigueData-AM2022 | [Figshare DOI 10.6084/m9.figshare.22337629.v2](https://doi.org/10.6084/m9.figshare.22337629.v2) | See the Figshare record |
| `external_data/cma2022_curves.json` | FatigueData-CMA2022 | [Figshare DOI 10.6084/m9.figshare.23007362.v2](https://doi.org/10.6084/m9.figshare.23007362.v2) | CC BY 4.0 |
| `external_data/weld2025_disjoint_curves.json` | A Dataset of Fatigue Properties for Welded Joints | [Figshare DOI 10.6084/m9.figshare.29254265.v2](https://doi.org/10.6084/m9.figshare.29254265.v2) | CC BY 4.0 |

## Direct source downloads

Figshare may reject copied browser download URLs with HTTP 403. The stable API/file links below can be downloaded with `curl -L` or `wget`.

```bash
# CMA2022 JSON
curl -L 'https://ndownloader.figshare.com/files/41461437' -o FatigueData-CMA2022.json

# Welded-joint S–N JSON
curl -L 'https://ndownloader.figshare.com/files/55248755' -o Weld-SN.json

# Welded-joint metadata/parameters
curl -L 'https://ndownloader.figshare.com/files/55248752' -o Weld-parameters.json
```

For AM2022, use the file listing on the DOI landing page so the download always follows the current version of the record.

## Harmonization

The included processed files follow this harmonization pipeline:

1. keeps the original database and curve identifiers;
2. converts stress and cycle fields to a common representation;
3. groups points into complete curve records before splitting;
4. removes exact duplicates before cross-database evaluation;
5. preserves material, process, testing, and missingness metadata when available;
6. records source-specific eligibility decisions in the generated summaries.

The fixed AM2022 split manifests are in `strict_splits/`. The LODO split summary reports the seeds and database-level counts used in the reported experiments.

## Terms of use

Users are responsible for following the terms and attribution requirements of each source dataset. The original data providers retain ownership of their records. The processed copies are supplied to reproduce the transformations and experiments described in the manuscript and should be cited together with the original datasets.
