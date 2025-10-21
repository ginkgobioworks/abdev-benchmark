# DeepViscosity Baseline

Viscosity predictions using DeepViscosity.

## Description

DeepViscosity predicts antibody solution viscosity using sequence-based deep learning. This baseline uses pre-computed DeepViscosity predictions from external analysis (via Tamarind.bio) and maps them to HIC (Hydrophobic Interaction Chromatography).

### Prediction

The predicted viscosity is used as a proxy for:
- **HIC**: Hydrophobic interaction chromatography retention time
  - Positive correlation observed (higher viscosity → higher HIC)

## Requirements

- Pre-computed DeepViscosity features in `../../features/processed_features/`
  - `GDPa1/DeepViscosity.csv`

Note: DeepViscosity features are currently only available for the GDPa1 training set.

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

This creates a prediction file in:
- `../../predictions/GDPa1/DeepViscosity/DeepViscosity.csv`

### Development

```bash
# Run tests
pixi run test

# Lint code
pixi run lint
```

## Output

The prediction file contains:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- `HIC` prediction

## Reference

DeepViscosity: Sharma V, et al. (2023). "In silico prediction of antibody viscosity using deep learning." mAbs.

## Acknowledgements

DeepViscosity features computed via Tamarind.bio.

