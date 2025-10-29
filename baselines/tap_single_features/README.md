# TAP Single Features Baseline

Individual TAP features used as direct predictors for specific properties.

## Description

This baseline uses correlations between individual TAP features and biophysical properties to generate simple predictions. Each TAP feature is mapped to one or more properties based on observed correlations (≥0.2).

### Feature Mappings

- **PNC** (Negative Patch Charge)
  - AC-SINS_pH7.4 (negative correlation)
  - PR_CHO (negative correlation)

- **SFvCSP** (Structural Fv Charge Symmetry Parameter)
  - AC-SINS_pH7.4 (positive correlation)
  - PR_CHO (positive correlation)
  - HIC (negative correlation)

- **PPC** (Positive Patch Charge)
  - AC-SINS_pH7.4 (positive correlation)
  - Titer (positive correlation)

- **CDR Length**
  - AC-SINS_pH7.4 (negative correlation)
  - HIC (positive correlation)

## Requirements

- Pre-computed TAP features in `../../features/processed_features/`
  - `GDPa1/TAP.csv`
  - `heldout_test/TAP.csv`

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

This creates multiple prediction files (one per TAP feature) in:
- `../../predictions/GDPa1/TAP/`
- `../../predictions/heldout_test/TAP/`

### Development

```bash
# Run tests
pixi run test

# Lint code
pixi run lint
```

## Output

Generates 4 prediction files per dataset:
- `TAP - PNC.csv`
- `TAP - SFvCSP.csv`
- `TAP - PPC.csv`
- `TAP - CDR Length.csv`

Each file contains predictions only for the properties that correlate with that feature.

## Notes

This is a simple baseline that demonstrates feature-property relationships. More sophisticated models (like `tap_linear`) combine multiple features for better predictions.
