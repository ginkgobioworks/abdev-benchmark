# Tests

Testing framework for validating baseline implementations in the antibody developability benchmark.

## Overview

This directory contains tests to ensure model correctly implement the train/predict interface and reference data for validation.

## Structure

```
tests/
├── baseline_results/       # Reference predictions and results for validation
│   ├── predictions/       # Reference prediction CSVs by dataset and model
│   └── results/           # Reference evaluation metrics
├── test_baseline_contract.py  # Contract compliance tests
└── README.md              # This file
```

## Baseline Contract Testing

The primary test validates that all model correctly implement the `BaseModel` interface with train/predict methods.

### Running Contract Tests

```bash
# Test all model
python tests/test_baseline_contract.py

# Test specific baseline
python tests/test_baseline_contract.py --baseline tap_linear

# Skip training step (faster, assumes models already trained)
python tests/test_baseline_contract.py --skip-train
```

### What is Tested

The contract test validates:
- **Train command** executes successfully and creates artifacts in `run_dir`
- **Predict command** works on both training and heldout data
- **Output format** follows the required CSV structure
- **Required columns** are present (antibody_name, vh_protein_sequence, vl_protein_sequence)
- **Row counts** match between input and output
- **Property predictions** are generated (at least one property column)

### Usage with Pytest

The root environment includes pytest:

```bash
pixi install
pixi run test           # Run all tests
pixi run test-contract  # Run contract tests specifically
```

## Reference Data

The `baseline_results/` directory contains reference predictions and evaluation results from validated baseline implementations. This data serves as:

- **Ground truth** for validation
- **Reference outputs** for comparing new implementations
- **Historical record** of baseline performance

### Structure

```
baseline_results/
├── predictions/
│   ├── GDPa1/                    # Full dataset predictions
│   ├── GDPa1_cross_validation/   # Cross-validation predictions
│   └── heldout_test/             # Held-out test set predictions
└── results/                      # Evaluation metrics
    ├── GDPa1/
    ├── GDPa1_cross_validation/
    └── metrics_all.csv
```

### Maintaining Reference Data

Reference data should be treated as stable. If updates are needed:
1. Document the reason for the update
2. Verify the new predictions are correct
3. Update all affected files consistently

## Adding Tests for New Model

When adding a new baseline:

1. Implement the baseline following `BaseModel` interface
2. Run contract tests to validate:
   ```bash
   python tests/test_baseline_contract.py --baseline your_baseline
   ```
3. Once validated, optionally add reference predictions to `baseline_results/`

## Best Practices

1. **Always run contract tests** before committing new model
2. **Keep reference data in version control** for reproducibility
3. **Test with both training and heldout data** to ensure generalization
4. **Use appropriate test data** - tests use full GDPa1 dataset and heldout sequences

## Integration with Orchestrator

The contract tests validate individual baseline behavior. The orchestrator (`run_all_models.py`) integrates these model for:
- Cross-validation workflows
- Batch prediction generation
- Automated evaluation

Test model individually first, then run via orchestrator:
```bash
# Individual baseline testing
python tests/test_baseline_contract.py --baseline tap_linear

# Full orchestrated workflow
pixi run all
```

## Troubleshooting

### All Tests Fail with "File Not Found"

**Problem:** Baseline hasn't been installed or data files are missing.

**Solution:**
```bash
cd model/your_baseline
pixi install
```

### Train/Predict Commands Fail

**Problem:** Baseline implementation doesn't follow `BaseModel` interface.

**Solution:**
- Check that `train()` and `predict()` methods match the interface signature
- Ensure `train()` saves artifacts to `run_dir`
- Ensure `predict()` loads from `run_dir` and returns DataFrame
- Review `libs/abdev_core/base.py` for interface definition

### Missing Required Columns in Predictions

**Problem:** Predictions don't include required columns.

**Solution:**
- Predictions must include: `antibody_name`, `vh_protein_sequence`, `vl_protein_sequence`
- At least one property column (HIC, Tm2, Titer, PR_CHO, AC-SINS_pH7.4)
- Review `data/schema/README.md` for format specification

## Summary

This testing framework provides validation that model correctly implement the train/predict interface required by the orchestrator. Use `test_baseline_contract.py` to validate implementations before integration.
