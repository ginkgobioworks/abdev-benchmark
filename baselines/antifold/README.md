# AntiFold Baseline

Antibody stability predictions using AntiFold.

## Description

AntiFold predicts antibody stability by comparing native and non-native sequence-structure compatibility. This baseline uses pre-computed AntiFold scores from external analysis (via Tamarind.bio) and maps them to biophysical properties.

### Predictions

The AntiFold Score is used to predict:
- **Tm2** (second melting temperature)
- **Titer** (expression level)

Note: Negative correlations were observed between AntiFold scores and these properties in the training data, which may indicate that lower scores (better native fit) correlate with stability/expression issues in this dataset.

## Requirements

- Pre-computed AntiFold features in `../../features/processed_features/`
  - `GDPa1/AntiFold.csv`
  - `heldout_test/AntiFold.csv`

These features were computed externally using AntiFold on antibody sequences.

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

This creates prediction files in:
- `../../predictions/GDPa1/AntiFold/AntiFold.csv`
- `../../predictions/heldout_test/AntiFold/AntiFold.csv`

### Development

```bash
# Run tests
pixi run test

# Lint code
pixi run lint
```

## Output

Each prediction file contains:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- `Tm2` prediction
- `Titer` prediction

## Reference

AntiFold: Ruffolo JA, et al. (2022). "Antibody structure prediction using interpretable deep learning." Patterns.

## Acknowledgements

AntiFold features computed via Tamarind.bio.
