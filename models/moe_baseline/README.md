# MOE Baseline

Ridge and MLP regression models trained on MOE (Molecular Operating Environment) molecular descriptors.

## Description

This baseline uses pre-computed MOE molecular descriptors to predict five antibody biophysical properties. Each property uses an optimized model configuration (Ridge or MLP) with property-specific feature sets selected through LASSO and Stabl methodologies.

**Model configurations:**
- **HIC**: Ridge regression (11 features, α=79.1)
- **PR_CHO**: Ridge regression (10 features, α=59.6)
- **AC-SINS_pH7.4**: Ridge regression (23 features, α=1.5)
- **Titer**: Ridge regression (4 features, α=59.6)
- **Tm2**: MLP neural network (8 features, 1 hidden layer)

## Expected Performance

Based on 5-fold cross-validation on GDPa1:

| Property | Model | Features | Spearman ρ (mean ± std) |
|----------|-------|----------|-------------------------|
| HIC | Ridge | 11 | 0.684 ± 0.091 |
| PR_CHO | Ridge | 10 | 0.436 ± 0.116 |
| AC-SINS_pH7.4 | Ridge | 23 | 0.514 ± 0.066 |
| Titer | Ridge | 4 | 0.308 ± 0.248 |
| Tm2 | MLP | 8 | 0.184 ± 0.181 |

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

Features were selected using two complementary methods:
- **LASSO**: L1 regularization for sparse feature selection
- **Stabl**: Bootstrap LASSO with FDR control for stable selection (Hédou et al., 2024)

Final feature sets range from 4 features (Titer) to 23 features (AC-SINS_pH7.4), balancing predictive power with model simplicity.

### Model Selection

For each property, Ridge, XGBoost, LightGBM, and MLP models were compared across multiple feature sets. Best configurations were selected based on 5-fold cross-validation performance.

### Prediction

Features are standardized using training set statistics. Ridge models apply linear regression; MLP models use a single hidden layer with early stopping for Tm2.

## Implementation

This baseline implements the `BaseModel` interface from `abdev_core`:

```python
from abdev_core import BaseModel, load_features

class MoeBaselineModel(BaseModel):
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        # Load MOE features from centralized store
        moe_features = load_features("MOE_properties")
        # Train 5 separate models with optimized configs
        # ...
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        # Load models and MOE features
        # Generate predictions for all 5 properties
        # ...
```

Features are managed centrally by `abdev_core`. See the [abdev_core documentation](../../libs/abdev_core/README.md) for details.

## Output

Predictions are written to `<out-dir>/predictions.csv` with columns:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- Predicted values for: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`

## References

- **MOE descriptors**: Nels Thorsteinsen
- **Stabl selection**: Hédou et al. (2024), "Discovery of sparse, reliable omic biomarkers with Stabl", Nature Biotechnology
- **GDPa1 dataset**: [ginkgo-datapoints/GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1)
