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
- **`evaluation/`**: Shared evaluation framework (Pixi project)
- **`libs/abdev_core/`**: Shared constants and utilities
- **`data/`**: Benchmark datasets and schema documentation
- **`features/`**: Pre-computed features from external tools
- **`predictions/`**: Generated predictions (output)
- **`results/`**: Evaluation results (output)
- **`tests/`**: Regression tests and reference data

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

### 3. Create Prediction Module

Create `src/your_baseline/__init__.py`:
```python
"""Your baseline description."""

__version__ = "0.1.0"
```

Create `src/your_baseline/predict.py`:
```python
"""Prediction module for your baseline."""

import argparse
from pathlib import Path
import pandas as pd

from abdev_core import PROPERTY_LIST


def main():
    """Main entry point for predictions."""
    parser = argparse.ArgumentParser(description="Your baseline predictions")
    parser.add_argument("--data-dir", type=Path, default=Path("../../data"))
    parser.add_argument("--features-dir", type=Path, default=Path("../../features/processed_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("../../predictions"))
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data_dir / "GDPa1_v1.2_20250814.csv")
    
    # Generate predictions
    # ... your model logic here ...
    
    # Save predictions in standard format
    output_dir = args.output_dir / "GDPa1" / "your_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_predictions = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"] + predicted_properties]
    df_predictions.to_csv(output_dir / "your_baseline.csv", index=False)
    
    print("✓ Predictions complete")


if __name__ == "__main__":
    main()
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

### Generate Predictions

\`\`\`bash
pixi run predict
\`\`\`

### Evaluate

\`\`\`bash
pixi run -C ../../evaluation score --pred ../../predictions/GDPa1/your_baseline/your_baseline.csv --truth ../../data/GDPa1_v1.2_20250814.csv
\`\`\`

## Reference

Citation if applicable.
```

### 5. Install and Test

```bash
pixi install
pixi run predict
```

### 6. Validate Output

```bash
pixi run -C ../../evaluation validate --pred ../../predictions/GDPa1/your_baseline/your_baseline.csv
```

### 7. Update CI

Add your baseline to `.github/workflows/ci.yml`:
```yaml
matrix:
  project:
    - evaluation
    - baselines/tap_linear
    - baselines/your_baseline  # Add this line
```

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

When adding shared constants or utilities:

1. Add to appropriate module (`constants.py` or `utils.py`)
2. Export in `__init__.py`
3. Update docstrings
4. Test that existing baselines still work

### Evaluation Module

When modifying metrics or validation:

1. Update `evaluation/src/evaluation/`
2. Update documentation in `evaluation/README.md`
3. Ensure backward compatibility or update all baselines
4. Add tests

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

- Check existing baselines for examples
- Read `data/schema/README.md` for format specifications
- See `evaluation/README.md` for evaluation details
- Review `notes/` for design decisions and progress notes

## Questions?

Open an issue or reach out to maintainers.

