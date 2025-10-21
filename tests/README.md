# Tests

Comprehensive regression testing framework for the antibody developability benchmark baselines.

## Overview

This testing framework ensures that the refactored Pixi-based baselines produce results consistent with the reference implementation. It provides both standalone scripts and pytest-based test suites.

## Structure

```
tests/
├── baseline_results/       # Reference predictions from original implementation
│   ├── predictions/       # Original prediction CSVs by dataset and model
│   └── results/           # Original evaluation metrics
├── test_regression.py     # Standalone regression test script
├── test_baselines.py      # Pytest-based test suite
├── test_config.json       # Configuration for baseline mappings
└── README.md             # This file
```

## Quick Start

### Option 1: Standalone Script (Recommended for CI/Local Testing)

```bash
# Test all baselines
python tests/test_regression.py

# Test specific baselines
python tests/test_regression.py --baselines tap_linear aggrescan3d

# Use higher tolerance for numeric comparisons
python tests/test_regression.py --tolerance 1e-4

# Show verbose output
python tests/test_regression.py --verbose

# List available baselines
python tests/test_regression.py --list-baselines
```

### Option 2: Pytest Suite (Recommended for Development)

```bash
# Test all baselines
pytest tests/test_baselines.py -v

# Test specific baseline
pytest tests/test_baselines.py -v -k tap_linear

# Test with custom tolerance
pytest tests/test_baselines.py -v --tolerance 1e-4

# Test only structure (fast smoke tests)
pytest tests/test_baselines.py -v -m structure

# Test only data quality
pytest tests/test_baselines.py -v -m quality

# Generate HTML report
pytest tests/test_baselines.py -v --html=test_report.html --self-contained-html
```

## Before Running Tests

You must first generate predictions by running the baselines:

```bash
# Run all baselines
./run_all_baselines.sh

# Or run individual baselines
cd baselines/tap_linear && pixi run predict
cd baselines/tap_single_features && pixi run predict
# ... etc
```

## Test Types

### 1. Regression Tests

Compares new predictions against reference predictions to ensure numerical consistency.

**What is tested:**
- Antibody names match between old and new predictions
- Property values are numerically close (within tolerance)
- File formats match specification
- No unexpected missing data

**Tolerance:**
- Default: 1e-6 (relative and absolute)
- Can be adjusted via `--tolerance` flag
- Per-baseline tolerances can be configured in `test_config.json`

### 2. Structure Tests

Validates that the output directory structure is correct.

**What is tested:**
- Output directories exist
- Expected files are present
- No unexpected files in output

**Usage:**
```bash
pytest tests/test_baselines.py -v -m structure
```

### 3. Data Quality Tests

Ensures prediction data meets quality standards.

**What is tested:**
- No all-NaN property columns
- Antibody names are present and non-null
- Data types are correct
- Value ranges are reasonable

**Usage:**
```bash
pytest tests/test_baselines.py -v -m quality
```

## Baseline Configuration

The test framework uses `BASELINE_CONFIG` (in `test_regression.py`) and `test_config.json` to map baseline names to their output locations and expected files.

### Configuration Structure

```python
{
    "baseline_name": {
        "output_dirs": {
            "dataset_name": "output_directory_name"
        },
        "files": {
            "dataset_name": ["file1.csv", "file2.csv"]
        }
    }
}
```

### Example: TAP Linear

```python
"tap_linear": {
    "output_dirs": {
        "GDPa1_cross_validation": "tap_linear",
        "heldout_test": "tap_linear"
    },
    "files": {
        "GDPa1_cross_validation": ["tap_linear.csv"],
        "heldout_test": ["tap_linear.csv"]
    }
}
```

## Understanding Test Results

### Successful Test

```
Testing tap_linear...
  ✓ PASS (2/2 files)
```

### Failed Test

```
Testing tap_linear...
  ✗ FAIL (1/2 files)

tap_linear/GDPa1_cross_validation/tap_linear.csv:
  - Column HIC: values differ (max diff: 1.23e-05, mean diff: 3.45e-07, tolerance: 1.00e-06)
```

### Common Failure Reasons

1. **File Not Found**
   ```
   New predictions not found: predictions/GDPa1/TAP/TAP - PNC.csv
   ```
   **Solution:** Run the baseline: `cd baselines/tap_single_features && pixi run predict`

2. **Values Differ**
   ```
   Column HIC: values differ (max diff: 1.23e-05, tolerance: 1.00e-06)
   ```
   **Solution:** 
   - If difference is small, increase tolerance: `--tolerance 1e-4`
   - If difference is large, investigate the baseline implementation

3. **Missing Antibodies**
   ```
   Missing 5 antibodies in new predictions
   ```
   **Solution:** Check data loading and filtering logic in the baseline

## Tolerance Guidelines

Different baselines may require different tolerances due to:

1. **Feature-based predictions** (no model): Very strict (1e-6)
   - `tap_single_features`
   - `aggrescan3d`
   - `antifold`
   - `saprot_vh`
   - `deepviscosity`

2. **Trained models**: More lenient (1e-4 to 1e-6)
   - `tap_linear` (Ridge regression)
   
Small numeric differences are expected due to:
- Floating point precision
- Different package versions
- Random initialization (if applicable)
- Numerical optimization differences

## Integration with CI/CD

The test framework is designed for CI/CD integration. Add to your workflow:

```yaml
- name: Run baselines
  run: ./run_all_baselines.sh

- name: Run regression tests
  run: python tests/test_regression.py

- name: Run pytest suite
  run: pytest tests/test_baselines.py -v
```

## Baseline Results

The `baseline_results/` directory contains predictions and evaluation results from the original monolithic implementation before the Pixi refactoring. These serve as ground truth for regression testing.

### Structure

```
baseline_results/
├── predictions/
│   ├── GDPa1/                    # Main dataset predictions
│   │   ├── TAP/
│   │   ├── Aggrescan3D/
│   │   ├── AntiFold/
│   │   ├── Saprot_VH/
│   │   └── DeepViscosity/
│   ├── GDPa1_cross_validation/   # Cross-validation predictions
│   │   └── TAP/
│   └── heldout_test/             # Held-out test set predictions
│       ├── TAP/
│       ├── Aggrescan3D/
│       └── AntiFold/
└── results/                      # Evaluation metrics
    ├── GDPa1/
    ├── GDPa1_cross_validation/
    └── metrics_all.csv
```

### Do Not Modify

These files should be treated as immutable reference data. If you need to update them:
1. Document why the update is needed
2. Archive the old version
3. Update with the new reference implementation results

## Troubleshooting

### All Tests Fail with "File Not Found"

**Problem:** Predictions haven't been generated yet.

**Solution:**
```bash
./run_all_baselines.sh
```

### Specific Baseline Fails with Large Differences

**Problem:** Implementation may have changed or there's a bug.

**Solution:**
1. Check the baseline's predict.py for changes
2. Review any dependency updates in pixi.toml
3. Compare against reference implementation logic
4. If intentional change, update reference data

### Tests Pass Locally but Fail in CI

**Problem:** Environment differences (package versions, random seeds, etc.)

**Solution:**
1. Ensure pixi.lock files are committed
2. Use same tolerance in CI as locally
3. Check for platform-specific differences (Linux vs macOS)

## Advanced Usage

### Custom Test Configuration

Create a custom config file:

```json
{
  "baseline_dir": "my_custom_baseline",
  "new_dir": "my_predictions",
  "default_tolerance": 1e-4,
  "baselines": {
    "my_baseline": {
      "output_dirs": {...},
      "files": {...}
    }
  }
}
```

### Programmatic Usage

```python
from test_regression import compare_predictions

ref_path = Path("tests/baseline_results/predictions/GDPa1/TAP/TAP - PNC.csv")
new_path = Path("predictions/GDPa1/TAP/TAP - PNC.csv")

passed, issues = compare_predictions(ref_path, new_path, tolerance=1e-6)
if not passed:
    for issue in issues:
        print(f"Issue: {issue}")
```

### Adding Tests for New Baselines

1. Add entry to `BASELINE_CONFIG` in `test_regression.py`:
```python
"my_baseline": {
    "output_dirs": {
        "GDPa1": "MyBaseline"
    },
    "files": {
        "GDPa1": ["MyBaseline.csv"]
    }
}
```

2. Run the baseline to generate predictions
3. Run tests to verify they work
4. Once validated, copy predictions to `tests/baseline_results/predictions/`

## Best Practices

1. **Always run baselines before testing**
   - Tests cannot pass without generated predictions

2. **Use appropriate tolerance**
   - Start with default (1e-6)
   - Increase only if needed
   - Document why higher tolerance is required

3. **Commit reference data**
   - Keep `baseline_results/` in version control
   - Update only when necessary and document changes

4. **Test before committing**
   - Run regression tests before pushing changes
   - Ensure all baselines still pass

5. **Use pytest markers**
   - Test specific baselines during development
   - Run full suite before release

## Summary

This testing framework provides comprehensive regression testing to ensure the refactored Pixi-based baselines maintain consistency with the reference implementation. Use the standalone script for CI and quick checks, and pytest for detailed development testing.

