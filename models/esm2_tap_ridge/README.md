# ESM2 + TAP Ridge Model

Ridge regression model on PCA-reduced ESM2 embeddings (640D → 50D) combined with TAP biophysical features and antibody subtype information.

## Features
- **ESM2-PCA**: 50 dimensions (from ESM2-t6-8M embeddings, retains ~93% variance)
- **TAP**: 5 biophysical features (SFvCSP, PSH, PPC, PNC, CDR Length)
- **Antibody subtypes**: 5 dimensions
  - hc_subtype (one-hot): 3 features (IgG1, IgG2, IgG4)
  - lc_subtype (one-hot): 2 features (Kappa, Lambda)

## Model Architecture

This model uses the same feature set as esm2_tap_rf but with Ridge regression instead of Random Forest:
- **PCA dimensionality reduction**: ESM2 embeddings (640D → 50D)
- **Ridge regression**: L2 regularization (alpha=1.0)
- **Feature combination**: ESM2-PCA + TAP + Subtypes

## Usage
```bash
pixi run python -m esm2_tap_ridge train --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run
pixi run python -m esm2_tap_ridge predict --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run --out-dir ./outputs
```
