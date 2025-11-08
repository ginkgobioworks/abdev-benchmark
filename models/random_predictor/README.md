# Random Predictor Baseline

A simple baseline that generates random predictions uniformly distributed within the observed ranges from training data.

## Purpose

This baseline serves several important purposes:
- **Performance Floor**: Establishes a baseline that any meaningful model should outperform
- **Pipeline Testing**: Useful for testing evaluation and orchestration pipelines
- **Sanity Check**: Validates that other models perform better than random guessing

## How It Works

1. **Training**: Computes the min/max ranges for each property from the training data
2. **Prediction**: Generates uniform random values within those ranges for each antibody

## Usage

### Training

```bash
cd model/random_predictor
pixi install
python -m random_predictor train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/test_run
```

### Prediction

```bash
# For training data (with cross-validation)
python -m random_predictor predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/test_run \
  --out-dir ./outputs/gdpa1

# For held-out test data
python -m random_predictor predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./runs/test_run \
  --out-dir ./outputs/heldout
```

## Features

- **Reproducible**: Uses a fixed random seed for consistent results
- **Property-aware**: Generates predictions within realistic ranges based on training data
- **Standard Interface**: Implements the BaseModel contract like all other models

## Performance Expectations

By design, this baseline should have:
- **Poor correlations**: Near-zero Pearson/Spearman correlations with true values
- **High errors**: Large MAE/RMSE relative to other models
- **Random rankings**: No ability to distinguish good vs bad antibodies

Any model that doesn't significantly outperform this baseline should be reconsidered.

## Dependencies

- Python 3.11+
- NumPy
- Pandas
- abdev-core

## License

Same as parent repository

