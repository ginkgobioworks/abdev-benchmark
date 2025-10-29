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

### CLI Interface

The baseline implements a standardized CLI interface with only required arguments. TAP features are loaded automatically from the centralized feature store.

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
2. Load TAP features automatically from centralized feature store
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
- TAP features are loaded automatically from centralized feature store
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

Features are managed centrally by `abdev_core` - model simply import what they need. See the [abdev_core documentation](../../libs/abdev_core/README.md) for details.

## Output

Predictions are written to `<out-dir>/predictions.csv` with columns:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- Predicted values for: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`

## Reference

TAP features from: Raybould MIJ, et al. (2019). "Five computational developability guidelines for therapeutic antibody profiling." PNAS.

