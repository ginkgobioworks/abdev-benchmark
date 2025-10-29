# Aggrescan3D Baseline

Structure-based aggregation propensity predictions using Aggrescan3D.

## Description

Aggrescan3D calculates aggregation propensity scores from protein structures. This baseline uses pre-computed Aggrescan3D features from external analysis (via Tamarind.bio) and maps them to relevant biophysical properties.

### Feature Variants

Four different aggregation scores are used as baselines:

1. **aggrescan_average_score**: Average aggregation score across the antibody → predicts HIC
2. **aggrescan_max_score**: Maximum aggregation score → predicts HIC, PR_CHO
3. **aggrescan_90_score**: 90th percentile aggregation score → predicts HIC
4. **aggrescan_cdrh3_average_score**: Average score in CDR-H3 region → predicts HIC

## Requirements

- Pre-computed Aggrescan3D features in `../../features/processed_features/`
  - `GDPa1/Aggrescan3D.csv`
  - `heldout_test/Aggrescan3D.csv`

These features were computed externally using Aggrescan3D on predicted antibody structures.

## Installation

```bash
# From this directory
pixi install
```

## Usage

### Generate Predictions

```bash
pixi run predict
```

This creates 4 prediction files (one per aggregation metric) in:
- `../../predictions/GDPa1/Aggrescan3D/`
- `../../predictions/heldout_test/Aggrescan3D/`

### Development

```bash
# Run tests
pixi run test

# Lint code
pixi run lint
```

## Output

Generates prediction files:
- `Aggrescan3D - aggrescan_average_score.csv`
- `Aggrescan3D - aggrescan_max_score.csv`
- `Aggrescan3D - aggrescan_90_score.csv`
- `Aggrescan3D - aggrescan_cdrh3_average_score.csv`

## Reference

Aggrescan3D: Zambrano R, et al. (2015). "AGGRESCAN3D (A3D): server for prediction of aggregation properties of protein structures." Nucleic Acids Research.

## Acknowledgements

Aggrescan3D features computed via Tamarind.bio.
