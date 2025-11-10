# MOE Baseline

Ridge, LightGBM, and MLP regression models trained on MOE (Molecular Operating Environment) molecular descriptors with nested cross-validation.

## Description

This baseline uses pre-computed MOE molecular descriptors to predict five antibody biophysical properties. Each property uses an optimized model configuration with **per-fold feature selection** to prevent data leakage. Features are selected independently for each cross-validation fold using LASSO, XGBoost, and Consensus (union of LASSO and XGBoost) methodologies.

**Model configurations:**
- **HIC**: Ridge regression (~30 Consensus features per fold)
- **PR_CHO**: LightGBM (~30 Consensus features per fold)
- **AC-SINS_pH7.4**: Ridge regression (~30 Consensus features per fold)
- **Titer**: MLP neural network (~31 Consensus features per fold)
- **Tm2**: LightGBM (~30 Consensus features per fold)

## Expected Performance

Based on 5-fold nested cross-validation on GDPa1 (unbiased estimates with per-fold feature selection):

| Property | Model | Avg Features | Spearman ρ (test) |
|----------|-------|--------------|-------------------|
| HIC | Ridge | ~30 | 0.656 |
| PR_CHO | LightGBM | ~30 | 0.353 |
| AC-SINS_pH7.4 | Ridge | ~30 | 0.424 |
| Titer | MLP | ~31 | 0.184 |
| Tm2 | LightGBM | ~30 | 0.107 |

## Requirements

- Pre-computed MOE features in `../../data/features/processed_features/`
  - `GDPa1/MOE_properties.csv` (training features)
  - `heldout_test/MOE_properties.csv` (test features)

MOE molecular descriptors computed from predicted antibody structures by Nels Thorsteinsen.

## Installation

```bash
# From this directory
pixi install
```

## Usage

### CLI Interface

The baseline implements a standardized CLI interface. MOE features are loaded automatically from the centralized feature store.

#### Train

```bash
pixi run python -m moe_baseline train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run \
  [--seed 42]
```

Trains 5 optimized models (one per property) and saves to `run-dir/model_artifacts.pkl`.

#### Predict

```bash
# Training data
pixi run python -m moe_baseline predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/train

# Heldout test set
pixi run python -m moe_baseline predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/heldout
```

Generates predictions for all samples and writes to `out-dir/predictions.csv`.

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

Automatically runs all models with 5-fold cross-validation and evaluation.

## Method

### MOE Descriptors

MOE molecular descriptors capture structural, electrostatic, hydrophobic, geometric, and secondary structure properties computed from predicted antibody structures. The descriptor set includes ~246 features covering:
- Structural: radius of gyration, packing scores, surface areas
- Electrostatic: charge distribution, dipole moments, multipole moments
- Hydrophobic: patch hydrophobicity, hydrophobic moments
- Secondary structure: helicity, strand content

### Feature Selection

Features are selected independently for each cross-validation fold using a nested CV approach to prevent data leakage:

1. **Per-fold selection**: For each of the 5 folds, features are selected using only the training data (80% of samples) from that fold
2. **Three feature sets tested**:
   - **All_MOE**: All 246 MOE descriptors (baseline)
   - **LASSO**: Features selected by L1 regularization (alpha tuned via internal CV on training data)
   - **Consensus**: Union of LASSO and XGBoost-derived features (XGBoost features: top features covering 90% cumulative SHAP importance)
3. **Pre-computed features**: Selected features for each fold are stored in JSON files (`*_fold_features_updated_feature_selection.json`)

Best models selected Consensus for most properties (HIC, PR_CHO, Tm2), LASSO for AC-SINS, and All_MOE for Titer. This approach ensures test data never influences feature selection, producing unbiased performance estimates.

### Model Selection

For each property, Ridge, XGBoost, LightGBM, and MLP models were compared across the three feature sets (All_MOE, LASSO, Consensus). Best configurations were selected based on 5-fold nested cross-validation performance:
- **Ridge**: Best for HIC and AC-SINS (linear relationships)
- **LightGBM**: Best for PR_CHO and Tm2 (captures non-linear patterns)
- **MLP**: Best for Titer (complex interactions, single hidden layer)

Models use Consensus features by default, which combine LASSO and XGBoost-derived selections for robust feature coverage.

### Prediction

The model automatically detects which fold is being predicted based on the fold column in the input data:
1. **Fold detection**: Identifies test samples by missing values in the fold column
2. **Feature loading**: Loads the appropriate pre-selected features for that fold from JSON files
3. **Standardization**: Features are standardized using training set statistics from that fold
4. **Model application**: Applies the trained model (Ridge, LightGBM, or MLP) to generate predictions

This ensures each prediction uses only features that were selected from its corresponding training data, maintaining the integrity of nested cross-validation.

## Implementation

This baseline implements the `BaseModel` interface from `abdev_core` with nested cross-validation support:

```python
from abdev_core import BaseModel, load_features

class MoeBaselineModel(BaseModel):
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        # Load MOE features from centralized store
        moe_features = load_features("MOE_properties")
        
        # Detect current fold from data
        fold_id = self._get_fold_id(df)
        
        # Load pre-selected features for this fold
        fold_features = self._get_fold_features(property_name, fold_id)
        
        # Train model (Ridge/LightGBM/MLP) with fold-specific features
        # ...
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        # Load models and MOE features
        # Automatically detect fold and use corresponding features
        # Generate predictions for all 5 properties
        # ...
```

**Key files:**
- `model.py`: Main implementation with fold-aware feature loading
- `*_fold_features_updated_feature_selection.json`: Pre-computed per-fold features (5 files, one per property)

Features are managed centrally by `abdev_core`. See the [abdev_core documentation](../../libs/abdev_core/README.md) for details.

## Output

Predictions are written to `<out-dir>/predictions.csv` with columns:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- Predicted values for: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`

## References

- **MOE descriptors**: Nels Thorsteinsen
- **GDPa1 dataset**: [ginkgo-datapoints/GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1)
