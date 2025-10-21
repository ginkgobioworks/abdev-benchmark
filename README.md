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

Each baseline can be run independently. For example, to run the TAP Linear baseline:

```bash
cd baselines/tap_linear
pixi install
pixi run predict
```

This will generate predictions in `../../predictions/`.

### Running All Baselines

To run all baselines sequentially:

```bash
./run_all_baselines.sh
```

This script will:
- Install dependencies for each baseline (if needed)
- Run predictions for all 6 baselines
- Display a summary of successes and failures
- Save predictions to `predictions/`

### Evaluating Predictions

To evaluate predictions:

```bash
cd evaluation
pixi install
pixi run score \
  --pred ../predictions/GDPa1_cross_validation/tap_linear/tap_linear.csv \
  --truth ../data/GDPa1_v1.2_20250814.csv \
  --dataset GDPa1_cross_validation
```

## Repository Structure

```
abdev-benchmark/
├── baselines/              # Baseline models (each is a Pixi project)
│   ├── tap_linear/        # Ridge regression on TAP features
│   ├── tap_single_features/ # Individual TAP features
│   ├── aggrescan3d/       # Aggregation propensity
│   ├── antifold/          # Stability predictions
│   ├── saprot_vh/         # Protein language model
│   └── deepviscosity/     # Viscosity predictions
├── evaluation/            # Standardized evaluation (Pixi project)
├── libs/
│   └── abdev_core/       # Shared constants and utilities
├── data/                 # Benchmark datasets
│   ├── schema/          # I/O contracts and format specifications
│   ├── GDPa1_v1.2_20250814.csv  # Main dataset
│   └── heldout-set-sequences.csv # Test set (labels withheld)
├── features/             # Pre-computed features
│   └── processed_features/
│       ├── GDPa1/       # Training features
│       └── heldout_test/ # Test features
├── predictions/          # Generated predictions (output)
├── results/             # Evaluation results (output)
└── tests/
    └── baseline_results/ # Reference predictions for regression testing
```

## Available Baselines

| Baseline | Description | Trains Model | Data Source |
|----------|-------------|--------------|-------------|
| **tap_linear** | Ridge regression on TAP descriptors | Yes | TAP features |
| **tap_single_features** | Individual TAP features as predictors | No | TAP features |
| **aggrescan3d** | Aggregation propensity from structure | No | Tamarind |
| **antifold** | Antibody stability predictions | No | Tamarind |
| **saprot_vh** | Protein language model features | No | Tamarind |
| **deepviscosity** | Viscosity predictions | No | Tamarind |

See individual baseline READMEs for details.

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

### Validation

Validate prediction format:
```bash
cd evaluation
pixi run validate --pred path/to/predictions.csv
```

## Adding a New Baseline

1. Create a new directory under `baselines/your_baseline/`
2. Add `pixi.toml` with dependencies and tasks
3. Create `src/your_baseline/predict.py` with prediction logic
4. Add `README.md` documenting the approach
5. Ensure predictions follow the standard format (see `data/schema/`)

Example `pixi.toml`:
```toml
[project]
name = "your-baseline"
channels = ["conda-forge"]

[dependencies]
python = "3.11.*"
pandas = ">=2.0"

[pypi-dependencies]
abdev-core = { path = "../../libs/abdev_core", editable = true }

[tasks]
predict = "python -m your_baseline.predict"
lint = "ruff check src"
test = "pytest tests -v"
```

## Development

### Running Tests

Each project has its own tests:
```bash
cd baselines/tap_linear
pixi run test
```

### Regression Testing

Compare new predictions against baseline results:
```bash
python tests/test_regression.py
```

### Linting

```bash
cd baselines/tap_linear
pixi run lint
```

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

[License information to be added]
