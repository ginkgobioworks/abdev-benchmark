# ESM2 + TAP XGBoost Model

XGBoost model on PCA-reduced ESM2 embeddings (640D → 50D) combined with TAP biophysical features and antibody subtype information.

## Features
- **ESM2-PCA**: 50 dimensions (from ESM2-t6-8M embeddings, retains ~94% variance)
- **TAP**: 5 biophysical features (SFvCSP, PSH, PPC, PNC, CDR Length)
- **Antibody subtypes**: 5 dimensions
  - hc_subtype (one-hot): 3 features (IgG1, IgG2, IgG4)
  - lc_subtype (one-hot): 2 features (Kappa, Lambda)
- **Total**: 60 features (vs ~197 training samples = 0.30:1 ratio)

## Anti-overfitting Hyperparameters 
- `n_estimators=50` - Moderate boosting rounds (sweet spot between 30 and 100)
- `max_depth=3` - Moderate tree depth (balanced complexity)
- `learning_rate=0.075` - Moderate learning rate (balanced learning speed)
- `min_child_weight=7` - Moderate sample requirement per child node
- `subsample=0.75` - Use 75% of training data per tree
- `colsample_bytree=0.75` - Use 75% of features per tree
- `early_stopping_rounds=10` - Stop if no validation improvement for 10 rounds
- `PCA_components=50` - Dimensionality reduction retaining ~93% variance
- `reg_lambda=30` - Strong L2 regularization

**Rationale**: V1 (100 rounds, depth 4) severely overfit. V2 (30 rounds, depth 2) prevented overfitting but was too conservative. V3 finds the sweet spot with moderate settings + early stopping for adaptive regularization.

## XGBoost vs Random Forest

XGBoost uses gradient boosting (sequential tree building) rather than Random Forest's bagging (parallel trees). Key advantages:
- Built-in regularization (L1/L2)
- More efficient with fewer trees
- Better handling of feature interactions
- Often superior performance on structured data

## Usage
```bash
pixi run python -m esm2_tap_xgb train --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run
pixi run python -m esm2_tap_xgb predict --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run --out-dir ./outputs
```
