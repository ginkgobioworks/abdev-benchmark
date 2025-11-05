# ESM2 + TAP Random Forest Model

Random Forest model on PCA-reduced ESM2 embeddings (640D → 50D) combined with TAP biophysical features and antibody subtype information.

## Features
- **ESM2-PCA**: 50 dimensions (from ESM2-t6-8M embeddings, retains ~93% variance)
- **TAP**: 5 biophysical features (SFvCSP, PSH, PPC, PNC, CDR Length)
- **Antibody subtypes**: 5 dimensions
  - hc_subtype (one-hot): 3 features (IgG1, IgG2, IgG4)
  - lc_subtype (one-hot): 2 features (Kappa, Lambda)
- **Total**: 60 features (vs ~197 training samples = 0.30:1 ratio)

## Anti-overfitting Hyperparameters
- `n_estimators=100` - Moderate number of trees
- `max_depth=5` - Moderate tree depth
- `min_samples_split=30` - Moderate split requirements
- `min_samples_leaf=10` - Moderate leaf size
- `PCA_components=50` - Dimensionality reduction retaining ~93% variance
- `max_features='sqrt'` - Feature bagging

**Performance**: Achieves good Tm2 performance 

## Usage
```bash
pixi run python -m esm2_tap_rf train --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run
pixi run python -m esm2_tap_rf predict --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run --out-dir ./outputs
```
