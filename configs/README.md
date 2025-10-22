# Configuration Files

This directory contains TOML configuration files for the baseline orchestration system.

## Files

- **`default.toml`** - Standard configuration for running all baselines with 5-fold CV

## Creating Custom Configs

### Quick Test Config

For rapid iteration during development:

```toml
# configs/quick_test.toml
[baselines]
include = ["random_predictor"]  # Single fast baseline

[cross_validation]
num_folds = 2  # Faster than 5-fold

[execution]
verbose = true
skip_eval = true  # Skip evaluation to save time
```

Usage:
```bash
python run_all_baselines.py --config configs/quick_test.toml
```

### Debug Config

For debugging a specific baseline:

```toml
# configs/debug_tap_linear.toml
[baselines]
include = ["tap_linear"]

[execution]
verbose = true  # Show all output
```

### Production Config

For final evaluation runs:

```toml
# configs/production.toml
[baselines]
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

### `[data]`
- `train_file` - Path to training CSV (relative to repo root)
- `heldout_file` - Path to heldout test CSV

### `[cross_validation]`
- `num_folds` - Number of CV folds (default: 5)
- `seed` - Random seed for reproducibility (default: 42)

### `[paths]`
All paths are relative to repository root unless absolute:
- `run_dir` - Model artifacts directory
- `predictions_dir` - Prediction outputs directory
- `evaluation_dir` - Evaluation results directory
- `temp_dir` - Temporary files (auto-cleaned)

### `[baselines]`
- `baselines_dir` - Directory containing baselines
- `include` - List of baseline names to run (empty = all)
- `exclude` - List of baseline names to skip

### `[execution]`
- `skip_train` - Skip training, use existing models
- `skip_eval` - Skip evaluation step
- `verbose` - Show detailed output for debugging

### `[evaluation]`
- `cv_dataset_name` - Dataset identifier for metrics
- `per_fold_metrics` - Compute per-fold metrics (future feature)

## CLI Overrides

Config values can be overridden via command line:

```bash
# Override skip_train
python run_all_baselines.py --config configs/my_config.toml --skip-train

# Override run_dir
python run_all_baselines.py --run-dir /tmp/test_runs

# Override verbose
python run_all_baselines.py --verbose
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
[baselines]
include = ["my_new_baseline", "random_predictor"]  # Compare to baseline
```

### Reproducing Paper Results
```toml
[baselines]
include = ["tap_linear", "tap_single_features", "random_predictor"]

[cross_validation]
seed = 42  # Match paper

[paths]
run_dir = "paper_results/runs"
predictions_dir = "paper_results/predictions"
```

### Quick Iteration
```toml
[baselines]
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
python run_all_baselines.py --config /full/path/to/config.toml
```

**Want to see what config was loaded:**
```bash
# Verbose mode shows config at startup
python run_all_baselines.py --verbose
```

**Config syntax error:**
```python
# Test config loading
import toml
config = toml.load("configs/my_config.toml")
print(config)
```

