# Configuration Files

This directory contains TOML configuration files for the baseline orchestration system.

## Files

- **`default.toml`** - Standard configuration for running all model with 5-fold CV

## Creating Custom Configs

### Quick Test Config

For rapid iteration during development:

```toml
# configs/quick_test.toml
[model]
include = ["random_predictor"]  # Single fast baseline

[cross_validation]
num_folds = 2  # Faster than 5-fold

[execution]
verbose = true
skip_eval = true  # Skip evaluation to save time
```

Usage:
```bash
python run_all_models.py --config configs/quick_test.toml
```

### Debug Config

For debugging a specific baseline:

```toml
# configs/debug_tap_linear.toml
[model]
include = ["tap_linear"]

[execution]
verbose = true  # Show all output
```

### Production Config

For final evaluation runs:

```toml
# configs/production.toml
[model]
# Run all except experimental ones
exclude = ["experimental_baseline"]

[paths]
# Store artifacts on mounted storage
run_dir = "/mnt/storage/runs"
predictions_dir = "/mnt/storage/predictions"

[execution]
skip_train = false
skip_eval = false
verbose = false
```

## Configuration Reference

All configuration sections and their options:

### `[data]`
Data file paths (relative to repository root):
- `train_file` - Training dataset CSV (default: `data/GDPa1_v1.2_20250814.csv`)
- `test_file` - Heldout test set CSV (default: `data/heldout-set-sequences.csv`)

### `[cross_validation]`
Cross-validation settings:
- `num_folds` - Number of CV folds (default: 5)
- `seed` - Random seed for reproducibility (default: 42)
- `fold_col` - Column name for fold assignments (empty string = generate random folds)

### `[paths]`
Output directories (relative to repository root unless absolute):
- `run_dir` - Model artifacts directory (default: `outputs/models`)
- `predictions_dir` - Prediction outputs directory (default: `outputs/predictions`)
- `evaluation_dir` - Evaluation results directory (default: `outputs/evaluation`)
- `temp_dir` - Temporary files, auto-cleaned (default: `.tmp_cv_splits`)

### `[model]`
Model selection:
- `model_dir` - Directory containing model (default: `model`)
- `include` - List of model names to run (empty = discover all)
- `exclude` - List of model names to skip (default: empty)

### `[execution]`
Execution control:
- `skip_train` - Skip training step, use existing models (default: false)
- `skip_eval` - Skip evaluation step (default: false)
- `verbose` - Show detailed output for debugging (default: false)

### `[evaluation]`
Evaluation settings:
- `cv_dataset_name` - Dataset identifier for cross-validation metrics (default: `GDPa1_cross_validation`)

## CLI Overrides

Config values can be overridden via command line:

```bash
# Override skip_train
python run_all_models.py --config configs/my_config.toml --skip-train

# Override run_dir
python run_all_models.py --run-dir /tmp/test_runs

# Override verbose
python run_all_models.py --verbose
```

CLI arguments take precedence over config file values.

## Best Practices

1. **Version Control**: Commit configs used for published results
2. **Naming**: Use descriptive names (e.g., `paper_figure_3.toml`)
3. **Documentation**: Add comments explaining non-obvious choices
4. **Paths**: Use relative paths when possible for portability
5. **Defaults**: Extend `default.toml` rather than replacing it

## Examples by Use Case

### Benchmarking New Baseline
```toml
[model]
include = ["my_new_baseline", "random_predictor"]  # Compare to baseline
```

### Reproducing Paper Results
```toml
[model]
include = ["tap_linear", "tap_single_features", "random_predictor"]

[cross_validation]
seed = 42  # Match paper

[paths]
run_dir = "paper_results/runs"
predictions_dir = "paper_results/predictions"
```

### Quick Iteration
```toml
[model]
include = ["my_baseline"]

[cross_validation]
num_folds = 2

[execution]
skip_eval = true
verbose = true
```

## Troubleshooting

**Config not found:**
```bash
# Use absolute path or run from repo root
python run_all_models.py --config /full/path/to/config.toml
```

**Want to see what config was loaded:**
```bash
# Verbose mode shows config at startup
python run_all_models.py --verbose
```

**Config syntax error:**
```python
# Test config loading
import toml
config = toml.load("configs/my_config.toml")
print(config)
```


