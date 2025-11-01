# Contributing to Antibody Developability Benchmark

Thank you for your interest in contributing! This guide will help you understand the repository structure and how to add new models or improvements.

In addition to this repo, feel free to reach out via slack in the [Bits in Bio](https://join.slack.com/t/bitsinbio/shared_invite/zt-3dqigle2b-e0dEkfPPzzWL055j_8N_eQ) community, specifically the channel `#competition-ginkgo-antibody-2025`.

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

3. Install root environment:
   ```bash
   pixi install
   ```

## Repository Structure

The repository uses a multi-project Pixi architecture:

- **`models/`**: Each model is an independent Pixi project
- **`libs/abdev_core/`**: Shared utilities, base classes, and evaluation
- **`configs/`**: Configuration files for orchestrator
- **`data/`**: Benchmark datasets and schema documentation
- **`outputs/`**: Generated outputs (models, predictions, evaluation)
- **`tests/`**: Model contract tests and reference data
- **`run_all_models.py`**: Main orchestrator script

## Adding a New Model

### 1. Create Directory Structure

```bash
mkdir -p models/your_model/src/your_model
cd models/your_model
```

### 2. Create `pixi.toml`

```toml
[workspace]
name = "your-model"
version = "0.1.0"
description = "Brief description of your model"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64"]

[dependencies]
python = "3.11.*"
pandas = ">=2.0"
numpy = ">=1.24"
# Add other conda dependencies

[pypi-dependencies]
abdev-core = { path = "../../libs/abdev_core", editable = true }
your-model = { path = ".", editable = true }
# Add other PyPI dependencies

[feature.dev.dependencies]
pytest = ">=7.0"
ruff = ">=0.1"
```

### 3. Implement Model Class

Create `src/your_model/__init__.py`:
```python
"""Your model description."""

__version__ = "0.1.0"
```

Create `src/your_model/model.py`:
```python
"""Model implementation for your model."""

from pathlib import Path
import pandas as pd
from abdev_core import BaseModel, load_features


class YourModel(BaseModel):
    """Your model description.
    
    This model [describe approach].
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

Create `src/your_model/run.py`:
```python
"""CLI entry point."""

from abdev_core import create_cli_app
from .model import YourModel

app = create_cli_app(YourModel, "Your Model")

if __name__ == "__main__":
    app()
```

Create `src/your_model/__main__.py`:
```python
"""Allow running as python -m your_model."""

from .run import app

if __name__ == "__main__":
    app()
```

### 4. Create README

Create `README.md`:
```markdown
# Your Model Name

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
pixi run python -m your_model train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./outputs/run_001 \
  --seed 42
\`\`\`

### Generate Predictions

\`\`\`bash
# On training data
pixi run python -m your_model predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./outputs/run_001 \
  --out-dir ./outputs/predictions

# On heldout data
pixi run python -m your_model predict \
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
pixi run python -m your_model train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./test_run

pixi run python -m your_model predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./test_run \
  --out-dir ./test_out
```

### 6. Validate Model Contract

```bash
# From repository root
python tests/test_model_contract.py --model your_model
```

This validates that your model correctly implements the `BaseModel` interface.

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

### Style Suggestions
- Follow PEP 8
- Type hints encouraged
- Docstrings for public functions

### Documentation
- Clear README per model
- Inline comments for complex logic
- Citation for external methods

### Testing Suggestions
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
4. Test that existing models still work

### Orchestrator (`run_all_models.py`)

When modifying the orchestration logic:

1. Test with multiple models
2. Ensure config file compatibility
3. Update `configs/README.md` if adding new config options
4. Verify evaluation metrics are computed correctly

## Submitting Changes

### Fork and Pull Request Workflow

We use the standard GitHub fork and pull request workflow:

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/abdev-benchmark.git
   cd abdev-benchmark
   ```
3. **Create a feature branch** (not main):
   ```bash
   git checkout -b add-your-model-name
   # or
   git checkout -b fix-issue-description
   ```
4. **Make your changes** following the guidelines in this document
5. **Commit your changes** with clear commit messages (see below)
6. **Push to your fork**:
   ```bash
   git push origin add-your-model-name
   ```
7. **Open a Pull Request** from your fork's branch to our `main` branch
8. **Address review feedback** - we appreciate your contribution and will attempt a timely review

### Pull Request Checklist

Before submitting your PR, please ensure:

- [ ] Code follows style guidelines
- [ ] Documentation added/updated (README, docstrings, etc.)
- [ ] Tests added/passing (`python tests/test_model_contract.py --model your_model`)
- [ ] Lockfile committed (`pixi.lock`) for new models
- [ ] README updated if needed
- [ ] No breaking changes to shared components (or discussed in PR description)
- [ ] For new models: Added entry to main README's model table
- [ ] For new models: Specified any external dependencies, data sources, and licensing
- [ ] CI checks pass (if applicable)

### Commit Messages

Use clear, descriptive commit messages:
```
Add XYZ model with feature engineering

- Implement prediction module
- Add documentation
- Configure Pixi environment
- Add tests and validation
```

## Getting Help

- Check existing models for examples (e.g., `models/random_predictor/`, `models/tap_linear/`)
- Read `data/schema/README.md` for format specifications
- Review `libs/abdev_core/` for shared utilities and base classes
- See `configs/README.md` for orchestrator configuration options

## Questions?

Open an issue or reach out to maintainers.

