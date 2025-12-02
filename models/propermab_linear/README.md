# PROPERMAB Linear Baseline

Ridge regression model trained on PROPERMAB features.

## Description

This baseline uses 5 PROPERMAB descriptors to predict antibody developability properties:
- **hyd_patch_area_cdr**: Hydrophobic surface patches
- **pos_patch_area**: Positively charged surface patches
- **dipole_moment**: Dipole moment of Fv domain
- **aromatic_asa**: Total solvent accessible surface area  
- **exposed_net_charge**: Total charge of CDR atoms that are solvent-exposed

A Ridge regression model is trained separately for each biophysical property using 5-fold cross-validation.

## Requirements

- Pre-computed PROPERMAB training and test features in `feature_store_top5.csv`

## Installation

```bash
# From this directory
pixi install
```

## Usage

### CLI Interface

The baseline implements a standardized CLI interface with only required arguments. PROPERMAB features are loaded from the csv file `feature_store_top5.csv`.

#### Train Models

```bash
# From the baseline directory
pixi run python -m tap_linear train \
  --data <path-to-training-csv> \
  --run-dir <directory-to-save-models> \
  [--seed 42]

# Example
pixi run python -m tap_linear train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./outputs/run_001
```

This will:
1. Load training data from `--data`
2. Load PROPERMAB features automatically from csv file
3. Train Ridge models for each property using 5-fold cross-validation
4. Save trained models to `run-dir/models.pkl`
5. Save cross-validation predictions to `run-dir/cv_predictions.csv`

#### Generate Predictions

```bash
# From the baseline directory
pixi run python -m tap_linear predict \
  --data <path-to-input-csv> \
  --run-dir <directory-with-trained-models> \
  --out-dir <directory-to-write-predictions>

# Example: CV predictions
pixi run python -m tap_linear predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./outputs/run_001 \
  --out-dir ../../predictions/cv_run_001

# Example: Heldout predictions
pixi run python -m tap_linear predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./outputs/run_001 \
  --out-dir ../../predictions/heldout_run_001
```

Behavior:
- PROPERMAB features are loaded automatically from csv file
- For **training data** (with fold column): Uses CV predictions from training
- For **heldout data**: Uses final models trained on all data
- Writes predictions to `out-dir/predictions.csv`

### Development

```bash
# Run tests (requires dev environment)
pixi run -e dev test

# Lint code (requires dev environment)
pixi run -e dev lint
```

## Implementation

This baseline implements the `BaseModel` interface from `abdev_core`:

```python
from abdev_core import BaseModel, load_features

class TapLinearModel(BaseModel):
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int) -> None:
        # Load features from centralized store
        tap_features = load_features("TAP", dataset="GDPa1")
        # Train models and generate CV predictions
        ...
    
    def predict(self, df: pd.DataFrame, run_dir: Path, out_dir: Path) -> None:
        # Load features from centralized store
        tap_features = load_features("TAP", dataset="heldout_test")
        # Generate predictions from trained models
        ...
```

Features are managed centrally by `abdev_core` - models simply import what they need. See the [abdev_core documentation](../../libs/abdev_core/README.md) for details.

## Output

Predictions are written to `<out-dir>/predictions.csv` with columns:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- Predicted values for: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`

## Reference

PROPERMAB features from: Li B, et al. (2025). "PROPERMAB: an integrative framework for in silico prediction of antibody developability using machine learning
" mAbs.
