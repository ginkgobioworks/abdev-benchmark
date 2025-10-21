# TAP Linear Baseline

Ridge regression model trained on TAP (Therapeutic Antibody Profiler) features.

## Description

This baseline uses 5 TAP descriptors to predict antibody developability properties:
- **SFvCSP**: Structural Fv Charge Symmetry Parameter
- **PSH**: Patch Surface Hydrophobicity
- **PPC**: Positive Patch Charge
- **PNC**: Negative Patch Charge  
- **CDR Length**: Combined CDR length

A Ridge regression model is trained separately for each biophysical property using 5-fold cross-validation.

## Requirements

- Pre-computed TAP features in `../../features/processed_features/`
  - `GDPa1/TAP.csv` (training features)
  - `heldout_test/TAP.csv` (test features)

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

This will:
1. Load TAP features and ground truth data
2. Train Ridge models with cross-validation
3. Generate predictions for both CV and heldout sets
4. Save to `../../predictions/`

### Evaluate Predictions

```bash
pixi run eval
```

This evaluates the cross-validation predictions and displays metrics.

### Development

```bash
# Run tests
pixi run test

# Lint code
pixi run lint
```

## Output

Predictions are written to:
- `../../predictions/GDPa1_cross_validation/tap_linear/tap_linear.csv`
- `../../predictions/heldout_test/tap_linear/tap_linear.csv`

Each file contains:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- Predicted values for: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`

## Reference

TAP features from: Raybould MIJ, et al. (2019). "Five computational developability guidelines for therapeutic antibody profiling." PNAS.

