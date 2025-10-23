# Contributing to Antibody Developability Benchmark

Thank you for your interest in contributing! This guide will help you understand the repository structure and how to add new baselines or improvements.

## Getting Started

### Prerequisites

1. Install Pixi:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd abdev-benchmark
   ```

3. Test that Pixi works:
   ```bash
   cd evaluation
   pixi install
   pixi run validate --help
   ```

## Repository Structure

The repository uses a multi-project Pixi architecture:

- **`baselines/`**: Each baseline is an independent Pixi project
- **`libs/abdev_core/`**: Shared utilities, base classes, and evaluation
- **`configs/`**: Configuration files for orchestrator
- **`data/`**: Benchmark datasets and schema documentation
- **`outputs/`**: Generated outputs (models, predictions, evaluation)
- **`tests/`**: Baseline contract tests and reference data
- **`run_all_baselines.py`**: Main orchestrator script

## Adding a New Baseline

### 1. Create Directory Structure

```bash
mkdir -p baselines/your_baseline/src/your_baseline
cd baselines/your_baseline
```

### 2. Create `pixi.toml`

```toml
[project]
name = "your-baseline"
version = "0.1.0"
description = "Brief description of your baseline"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64"]

[dependencies]
python = "3.11.*"
pandas = ">=2.0"
numpy = ">=1.24"
# Add other conda dependencies

[pypi-dependencies]
abdev-core = { path = "../../libs/abdev_core", editable = true }
# Add other PyPI dependencies

[tasks]
predict = "python -m your_baseline.predict"
lint = "ruff check src && ruff format --check src"
test = "pytest tests -v"

[feature.dev.dependencies]
pytest = ">=7.0"
ruff = ">=0.1"
```

### 3. Implement Model Class

Create `src/your_baseline/__init__.py`:
```python
"""Your baseline description."""

__version__ = "0.1.0"
```

Create `src/your_baseline/model.py`:
```python
"""Model implementation for your baseline."""

from pathlib import Path
import pandas as pd
from abdev_core import BaseModel, load_features


class YourModel(BaseModel):
    """Your model description.
    
    This baseline [describe approach].
    """
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train model on ALL provided data and save artifacts to run_dir.
        
        Args:
            df: Training dataframe with sequences and labels
            run_dir: Directory to save model artifacts
            seed: Random seed for reproducibility
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load features if needed
        features = load_features("YourFeatureSource", dataset="GDPa1")
        
        # Train your model on ALL samples in df
        # The orchestrator handles CV splitting externally
        # ... your training logic here ...
        
        # Save model artifacts
        # model_path = run_dir / "model.pkl"
        # pickle.dump(model, open(model_path, "wb"))
        
        print(f"Model saved to {run_dir}")
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for ALL provided samples.
        
        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing saved model artifacts
            
        Returns:
            DataFrame with predictions
        """
        # Load model artifacts
        # model = pickle.load(open(run_dir / "model.pkl", "rb"))
        
        # Load features if needed
        features = load_features("YourFeatureSource")
        
        # Generate predictions for ALL samples
        # ... your prediction logic here ...
        
        # Return predictions
        df_output = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        # df_output["HIC"] = predictions
        
        return df_output
```

Create `src/your_baseline/run.py`:
```python
"""CLI entry point."""

from abdev_core import create_cli_app
from .model import YourModel

app = create_cli_app(YourModel, "Your Model")

if __name__ == "__main__":
    app()
```

Create `src/your_baseline/__main__.py`:
```python
"""Allow running as python -m your_baseline."""

from .run import app

if __name__ == "__main__":
    app()
```

### 4. Create README

Create `README.md`:
```markdown
# Your Baseline Name

Brief description.

## Description

Detailed explanation of the method, what it does, and how it works.

## Requirements

- List data dependencies
- List feature dependencies

## Installation

\`\`\`bash
pixi install
\`\`\`

## Usage

### Train Model

\`\`\`bash
pixi run python -m your_baseline train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./outputs/run_001 \
  --seed 42
\`\`\`

### Generate Predictions

\`\`\`bash
# On training data
pixi run python -m your_baseline predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./outputs/run_001 \
  --out-dir ./outputs/predictions

# On heldout data
pixi run python -m your_baseline predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./outputs/run_001 \
  --out-dir ./outputs/predictions_heldout
\`\`\`

### Run via Orchestrator

\`\`\`bash
# From repository root
pixi run all
\`\`\`

## Reference

Citation if applicable.
```

### 5. Install and Test

```bash
pixi install

# Test train/predict manually
pixi run python -m your_baseline train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./test_run

pixi run python -m your_baseline predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./test_run \
  --out-dir ./test_out
```

### 6. Validate Baseline Contract

```bash
# From repository root
python tests/test_baseline_contract.py --baseline your_baseline
```

This validates that your baseline correctly implements the `BaseModel` interface.

## Prediction Format Requirements

All predictions must follow the standard format (see `data/schema/README.md`):

### Required Columns
- `antibody_name` (string, unique, no NaN)

### Property Columns (at least one required)
- `HIC` (float)
- `Tm2` (float)
- `Titer` (float)
- `PR_CHO` (float)
- `AC-SINS_pH7.4` (float)

### Optional Columns
- `vh_protein_sequence` (string)
- `vl_protein_sequence` (string)

## Code Standards

### Style
- Follow PEP 8
- Use ruff for linting: `pixi run lint`
- Type hints encouraged
- Docstrings for public functions

### Documentation
- Clear README per baseline
- Inline comments for complex logic
- Citation for external methods

### Testing
- Add pytest tests in `tests/`
- Compare against reference predictions if available
- Test edge cases (missing values, etc.)

## Modifying Shared Components

### Core Library (`libs/abdev_core/`)

When adding shared constants, utilities, or evaluation functions:

1. Add to appropriate module:
   - `constants.py` - Shared constants (properties, datasets, etc.)
   - `utils.py` - Data manipulation utilities
   - `base.py` - BaseModel interface
   - `evaluation_metrics.py` - Evaluation metrics
   - `features.py` - Feature loading utilities
2. Export in `__init__.py`
3. Update docstrings
4. Test that existing baselines still work

### Orchestrator (`run_all_baselines.py`)

When modifying the orchestration logic:

1. Test with multiple baselines
2. Ensure config file compatibility
3. Update `configs/README.md` if adding new config options
4. Verify evaluation metrics are computed correctly

## Submitting Changes

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Documentation added/updated
- [ ] Tests added/passing
- [ ] CI pipeline passes
- [ ] Lockfile committed (`pixi.lock`)
- [ ] README updated if needed
- [ ] No breaking changes to shared components (or discussed)

### Commit Messages

Use clear, descriptive commit messages:
```
Add XYZ baseline with feature engineering

- Implement prediction module
- Add documentation
- Configure Pixi environment
- Update CI matrix
```

## Getting Help

- Check existing baselines for examples (e.g., `baselines/random_predictor/`, `baselines/tap_linear/`)
- Read `data/schema/README.md` for format specifications
- Review `libs/abdev_core/` for shared utilities and base classes
- See `configs/README.md` for orchestrator configuration options

## Questions?

Open an issue or reach out to maintainers.

