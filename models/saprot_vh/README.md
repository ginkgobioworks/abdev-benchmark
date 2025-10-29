# Saprot_VH Baseline

Protein language model predictions using Saprot on VH (variable heavy) sequences.

## Description

Saprot (Structure-aware Protein Language Model) generates predictions for protein properties using sequence and structure information. This baseline uses pre-computed Saprot features from external analysis (via Tamarind.bio) on VH sequences.

### Features

Two Saprot features are used as model:

1. **solubility_probability**: Predicted solubility → predicts PR_CHO (polyreactivity)
2. **stability_score**: Predicted stability → predicts Tm2 (melting temperature)
   - Note: Negative correlation observed with Tm2

## Requirements

- Pre-computed Saprot features in `../../features/processed_features/`
  - `GDPa1/Saprot_VH.csv`

Note: Saprot features are currently only available for the GDPa1 training set, not for the heldout test set.

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

This creates 2 prediction files in:
- `../../predictions/GDPa1/Saprot_VH/`

### Development

```bash
# Run tests
pixi run test

# Lint code
pixi run lint
```

## Output

Generates prediction files:
- `Saprot_VH - solubility_probability.csv` (predicts PR_CHO)
- `Saprot_VH - stability_score.csv` (predicts Tm2)

## Reference

Saprot: Su J, et al. (2023). "SaProt: Protein Language Modeling with Structure-aware Vocabulary." bioRxiv.

## Acknowledgements

Saprot features computed via Tamarind.bio.

