# Tests

Regression tests for the antibody developability benchmark baselines.

## Structure

```
tests/
├── baseline_results/       # Reference predictions and results from original implementation
│   ├── predictions/       # Original prediction CSVs
│   └── results/           # Original evaluation metrics
└── test_regression.py     # Regression test script
```

## Baseline Results

The `baseline_results/` directory contains predictions and evaluation results from the original monolithic implementation. These serve as reference data for regression testing to ensure the refactored Pixi-based baselines produce equivalent results.

### Usage

After generating new predictions with the refactored baselines, run regression tests to verify consistency:

```bash
# From project root
python tests/test_regression.py
```

## Validation

Regression tests check:
- Prediction file format matches
- Antibody names align between old and new predictions
- Predicted values are numerically close (within tolerance for floating point differences)
- Evaluation metrics match within tolerance

## Tolerance

Small numeric differences are expected due to:
- Different package versions
- Floating point precision
- Random seed differences (if applicable)

Default tolerance: 1e-6 for predictions, 1e-4 for evaluation metrics.

