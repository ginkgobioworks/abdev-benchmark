# Antibody Developability Benchmark

A benchmark suite for evaluating machine learning models on antibody biophysical property prediction.

## Overview

This repository provides:
- **Baseline models** for predicting antibody developability properties
- **Standardized evaluation** framework with consistent metrics
- **Pre-computed features** from various computational tools
- **Benchmark dataset** (GDPa1) with measured biophysical properties

Each baseline is an isolated [Pixi](https://prefix.dev/) project with its own dependencies and lockfile, ensuring reproducibility.

## Quick Start

### Installation

1. Install Pixi (if not already installed):
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. Clone the repository:
```bash
git clone <repository-url>
cd abdev-benchmark
```

### Running a Baseline

Each baseline follows a standard train/predict workflow. For example, with TAP Linear:

```bash
cd baselines/tap_linear
pixi install

# Train the model
pixi run python -m tap_linear train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run

# Generate predictions
pixi run python -m tap_linear predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/train

# Predict on heldout data
pixi run python -m tap_linear predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/heldout
```

All baselines implement the same `BaseModel` interface with `train()` and `predict()` methods.

**Note:** Models train on ALL provided data. The orchestrator handles data splitting for cross-validation.

### Running All Baselines

To train, predict, and evaluate all baselines:

```bash
pixi run all
```

This orchestrator will:
- Automatically discover all baselines (directories with `pixi.toml` in `baselines/`)
- Install dependencies for each baseline
- Train models with 5-fold cross-validation on GDPa1
- Generate predictions for both CV and heldout test sets
- Evaluate predictions and compute metrics (Spearman, Top 10% Recall)
- Display summary tables with results
- Save artifacts to `outputs/models/`, `outputs/predictions/`, `outputs/evaluation/`

### Example Output

After running all baselines, you'll see a summary table like this:

**Spearman ρ (Test, Average Fold)**

| Baseline | AC-SINS_pH7.4 | HIC | PR_CHO | Titer | Tm2 |
|---|---:|---:|---:|---:|---:|
| esm2_ridge | 0.420 | 0.416 | 0.420 | 0.180 | -0.098 |
| piggen | 0.388 | 0.346 | 0.424 | 0.238 | -0.119 |
| tap_single_features | 0.327 | 0.231 | 0.074 | 0.126 | — |
| tap_linear | 0.294 | 0.222 | 0.136 | 0.113 | -0.115 |
| aggrescan3d | — | 0.404 | 0.112 | — | — |
| saprot_vh | — | — | 0.289 | — | 0.162 |
| antifold | — | — | — | 0.194 | 0.084 |
| deepviscosity | — | 0.176 | — | — | — |
| random_predictor | -0.026 | 0.002 | -0.081 | 0.068 | -0.000 |

Options:
```bash
pixi run all                    # Full workflow (train + predict + eval)
pixi run all-skip-train         # Skip training (use existing models)
pixi run all-skip-eval          # Skip evaluation step
python run_all_baselines.py --help  # See all options
```

You can customize behavior via config files in `configs/`:
```bash
python run_all_baselines.py --config configs/custom.toml
```

## Repository Structure

```
abdev-benchmark/
├── baselines/              # Baseline models (each is a Pixi project)
│   └── random_predictor/  # E.g. Random baseline (performance floor)
├── libs/
│   └── abdev_core/       # Shared utilities, base classes, and evaluation
├── configs/              # Configuration files for orchestrator
├── data/                 # Benchmark datasets and precomputed features
├── outputs/              # Generated outputs (models, predictions, evaluation)
│   ├── models/          # Trained model artifacts
│   ├── predictions/     # Generated predictions
│   └── evaluation/      # Evaluation metrics
└── pixi.toml            # Root environment with orchestrator dependencies
```

## Available Baselines

| Baseline | Description | Trains Model | Data Source |
|----------|-------------|--------------|-------------|
| **tap_linear** | Ridge regression on TAP descriptors | Yes | TAP features |
| **tap_single_features** | Individual TAP features as predictors | No | TAP features |
| **esm2_ridge** | Ridge regression on ESM2 embeddings | Yes | Sequences (ESM2 model) |
| **piggen** | Ridge regression on p-IgGen embeddings | Yes | Sequences (p-IgGen model) |
| **aggrescan3d** | Aggregation propensity from structure | No | Tamarind |
| **antifold** | Antibody stability predictions | No | Tamarind |
| **saprot_vh** | Protein language model features | No | Tamarind |
| **deepviscosity** | Viscosity predictions | No | Tamarind |
| **random_predictor** | Random predictions (baseline floor) | No | None |

All baselines implement the `BaseModel` interface with standardized `train()` and `predict()` commands. See individual baseline READMEs for details.

## Predicted Properties

The benchmark evaluates predictions for 5 biophysical properties:

- **HIC**: Hydrophobic Interaction Chromatography retention time (lower is better)
- **Tm2**: Second melting temperature in °C (higher is better)
- **Titer**: Expression titer in mg/L (higher is better)
- **PR_CHO**: Polyreactivity CHO (lower is better)
- **AC-SINS_pH7.4**: Self-interaction at pH 7.4 (lower is better)

## Evaluation Metrics

For each property:
- **Spearman correlation**: Rank correlation between predicted and true values
- **Top 10% recall**: Fraction of true top 10% captured in predicted top 10%

For cross-validation datasets, metrics are averaged across 5 folds.

## Data Format

### Predictions

Prediction CSVs must contain:
- `antibody_name` (required)
- One or more property columns (HIC, Tm2, Titer, PR_CHO, AC-SINS_pH7.4)

See `data/schema/README.md` for detailed format specifications.

Prediction format validation is handled automatically by the orchestrator using `abdev_core.validate_prediction_format()`.

## Adding a New Baseline

All baselines must implement the `BaseModel` interface with `train()` and `predict()` methods.

### Steps

1. **Create directory structure:**
   ```bash
   mkdir -p baselines/your_baseline/src/your_baseline
   ```

2. **Create `pixi.toml`** with dependencies:
   ```toml
   [workspace]
   name = "your-baseline"
   version = "0.1.0"
   channels = ["conda-forge"]
   platforms = ["linux-64", "osx-64", "osx-arm64"]
   
   [dependencies]
   python = "3.11.*"
   pandas = ">=2.0"
   typer = ">=0.9"
   
   [pypi-dependencies]
   abdev-core = { path = "../../libs/abdev_core", editable = true }
   your-baseline = { path = ".", editable = true }
   ```

3. **Create `pyproject.toml`** for package metadata.

4. **Implement `src/your_baseline/model.py`:**
   ```python
   from pathlib import Path
   import pandas as pd
   from abdev_core import BaseModel
   
   class YourModel(BaseModel):
       def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
           """Train model on ALL provided data and save artifacts to run_dir."""
           # Train on ALL samples in df (no internal CV)
           # Your training logic here
           pass
       
       def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
           """Generate predictions for ALL provided samples.
           
           Returns:
               DataFrame with predictions. Orchestrator handles saving to file.
           """
           # Predict on ALL samples in df
           # Your prediction logic here
           # Return DataFrame (don't save to disk - orchestrator handles I/O)
           return df_with_predictions
   ```

5. **Create `src/your_baseline/run.py`:**
   ```python
   from abdev_core import create_cli_app
   from .model import YourModel
   
   app = create_cli_app(YourModel, "Your Model")
   
   if __name__ == "__main__":
       app()
   ```

6. **Create `src/your_baseline/__main__.py`:**
   ```python
   from .run import app
   if __name__ == "__main__":
       app()
   ```

7. **Add `README.md`** documenting your approach.

8. **Test your baseline:**
   ```bash
   # From repository root
   python tests/test_baseline_contract.py --baseline your_baseline
   
   # Or test train/predict manually
   cd baselines/your_baseline
   pixi install
   pixi run python -m your_baseline train --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./test_run
   pixi run python -m your_baseline predict --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./test_run --out-dir ./test_out
   ```

See `baselines/random_predictor/` for a complete minimal example.

## Development

### Testing Baseline Contract Compliance

Validate that all baselines implement the train/predict contract correctly:

```bash
# Install dev environment dependencies (includes pytest)
pixi install -e dev

# Test all baselines
pixi run -e dev test-contract

# Or run with options
pixi run -e dev python tests/test_baseline_contract.py --baseline tap_linear  # Test specific baseline
pixi run -e dev python tests/test_baseline_contract.py --skip-train           # Skip training step
pixi run -e dev python tests/test_baseline_contract.py --help                 # See all options
```

This test script validates:
- Train command executes successfully and creates artifacts
- Predict command works on both training and heldout data
- Output predictions follow the required CSV format
- All required columns are present

**Note:** The test script uses `pixi run` to activate each baseline's environment, matching how the orchestrator runs baselines.

## Citation

If you use this benchmark, please cite:

```
[Citation information to be added]
```

## Acknowledgements

- **Tamarind.bio**: Computed features for Aggrescan3D, AntiFold, BALM_Paired, DeepSP, DeepViscosity, Saprot, TEMPRO, TAP
- **Nels Thorsteinsen**: MOE structure predictions
- Contributors to individual baseline methods (see baseline READMEs)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** Datasets and individual baseline implementations may have their own licenses and terms of use. Please refer to the specific documentation in each baseline directory and the `data/` directory for details.
