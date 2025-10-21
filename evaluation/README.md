# Evaluation Module

Standardized evaluation module for the antibody developability benchmark.

## Installation

This project uses [Pixi](https://prefix.dev/docs/pixi/overview) for dependency management.

```bash
# Install Pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Initialize the environment (from this directory)
pixi install
```

## Usage

### Score Predictions

Score a prediction CSV against ground truth:

```bash
pixi run score --pred path/to/predictions.csv --truth ../data/GDPa1_v1.2_20250814.csv --dataset GDPa1
```

Options:
- `--pred`: Path to predictions CSV (required)
- `--truth`: Path to ground truth CSV (required)
- `--model-name`: Model name for results (defaults to filename)
- `--dataset`: Dataset name, e.g., "GDPa1" or "GDPa1_cross_validation"
- `--output`: Path to save results CSV

### Validate Predictions

Check if a prediction file has the correct format:

```bash
pixi run validate --pred path/to/predictions.csv
```

## I/O Contract

### Prediction Format

Prediction CSVs must have:
- `antibody_name` column (required, unique values, no NaN)
- At least one property column from: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`
- Each property column contains predicted values (numeric)

Example:
```csv
antibody_name,HIC,Tm2
antibody-001,2.5,75.3
antibody-002,3.1,72.8
```

### Ground Truth Format

Ground truth CSVs should have:
- `antibody_name` column
- Property columns matching prediction columns
- `hierarchical_cluster_IgG_isotype_stratified_fold` column (for cross-validation evaluation)

### Evaluation Metrics

For each property, the following metrics are computed:
- **Spearman correlation**: Rank correlation between predicted and true values
- **Top 10% recall**: Fraction of true top 10% captured in predicted top 10%
  - For properties where higher is better (Tm2, Titer), computed directly
  - For properties where lower is better (HIC, PR_CHO, AC-SINS_pH7.4), values are negated

### Cross-Validation Evaluation

When `dataset_name` is "GDPa1_cross_validation", metrics are computed per fold and averaged.

## Development

### Running Tests

```bash
pixi run test
```

### Linting

```bash
pixi run lint
```

## Using from Other Projects

Other baselines can call the evaluation module:

```bash
# From a baseline directory
pixi run -C ../../evaluation score --pred ../../predictions/my_baseline/pred.csv --truth ../../data/GDPa1_v1.2_20250814.csv
```

Or import directly in Python (if evaluation is installed):

```python
from evaluation.metrics import evaluate_model

results = evaluate_model(
    preds_path="predictions.csv",
    target_path="ground_truth.csv",
    model_name="my_model",
    dataset_name="GDPa1"
)
```

