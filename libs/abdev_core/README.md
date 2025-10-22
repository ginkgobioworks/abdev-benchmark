# abdev_core

Shared core library for the Antibody Developability Benchmark.

## Overview

`abdev_core` provides:
- **Constants**: Property names, datasets, evaluation directionality
- **BaseModel Interface**: Standard interface for baseline models
- **Feature Loading**: Centralized access to pre-computed features
- **CLI Utilities**: Helpers for building command-line interfaces
- **Utility Functions**: Common operations for sequence processing

## Installation

This package is typically installed as an editable dependency in baseline projects:

```toml
# In baseline pixi.toml
[pypi-dependencies]
abdev-core = { path = "../../libs/abdev_core", editable = true }
```

## Feature Loading

Features are managed centrally and can be imported by any baseline. Baselines **do not** need to know feature file paths.

### Simple Usage

```python
from abdev_core import load_features

# Load features for a dataset (returns DataFrame indexed by antibody_name)
tap_features = load_features("TAP", dataset="GDPa1")

# Use as dictionary lookup
features_for_antibody = tap_features.loc["abagovomab"]

# Or merge with your data
df_with_features = df.merge(tap_features.reset_index(), on="antibody_name")
```

### Advanced Usage

```python
from abdev_core import FeatureLoader

# Create loader (uses standard feature location by default)
loader = FeatureLoader()

# List available features
available = loader.list_available_features(dataset="GDPa1")
# Returns: ['TAP', 'Aggrescan3D', 'AntiFold', 'Saprot', ...]

# Load multiple feature sets
tap = loader.load_features("TAP", dataset="GDPa1")
aggrescan = loader.load_features("Aggrescan3D", dataset="GDPa1")

# Get as dictionary
tap_dict = loader.get_feature_dict("TAP", dataset="GDPa1")
features = tap_dict["abagovomab"]  # Returns Series for one antibody
```

### Available Datasets

- `"GDPa1"`: Training data features
- `"heldout_test"`: Heldout test set features

## BaseModel Interface

All baseline models should implement the `BaseModel` abstract class:

```python
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class BaseModel(ABC):
    """Contract: train writes to run_dir; predict reads run_dir and writes predictions.csv."""
    
    @abstractmethod
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train the model and save artifacts to run_dir."""
        ...
    
    @abstractmethod
    def predict(self, df: pd.DataFrame, run_dir: Path, out_dir: Path) -> None:
        """Generate predictions using saved model artifacts."""
        ...
```

### Contract

1. **train()** writes all model artifacts (models, weights, checkpoints, CV predictions) to `run_dir`
2. **predict()** reads artifacts from `run_dir` and writes `predictions.csv` to `out_dir`
3. Both methods must create their output directories if they don't exist
4. Non-training baselines should implement a no-op `train()` method

## Implementing a Baseline

### Minimal Example

```python
# model.py
from pathlib import Path
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from abdev_core import BaseModel, PROPERTY_LIST, load_features

class MyModel(BaseModel):
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load features from centralized store
        features = load_features("TAP", dataset="GDPa1")
        df_merged = df.merge(features.reset_index(), on="antibody_name")
        
        # Train a simple model
        model = LinearRegression()
        X = df_merged[['SFvCSP', 'PSH']]  # TAP features
        y = df_merged['HIC']
        model.fit(X, y)
        
        # Save model
        with open(run_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
    
    def predict(self, df: pd.DataFrame, run_dir: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load features (auto-detects dataset)
        dataset = "heldout_test" if "fold" not in df.columns else "GDPa1"
        features = load_features("TAP", dataset=dataset)
        df_merged = df.merge(features.reset_index(), on="antibody_name")
        
        # Load model
        with open(run_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Generate predictions
        X = df_merged[['SFvCSP', 'PSH']]
        df_output = df[['antibody_name', 'vh_protein_sequence', 'vl_protein_sequence']].copy()
        df_output['HIC'] = model.predict(X)
        
        # Write output
        df_output.to_csv(out_dir / "predictions.csv", index=False)


# run.py
from pathlib import Path
import typer
import pandas as pd
from .model import MyModel

app = typer.Typer(add_completion=False)

@app.command()
def train(
    data: Path = typer.Option(..., help="Path to training data CSV"),
    run_dir: Path = typer.Option(..., help="Directory to save model artifacts"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Train the model."""
    df = pd.read_csv(data)
    MyModel().train(df, run_dir, seed=seed)

@app.command()
def predict(
    data: Path = typer.Option(..., help="Path to input data CSV"),
    run_dir: Path = typer.Option(..., help="Directory with model artifacts"),
    out_dir: Path = typer.Option(..., help="Directory to write predictions"),
):
    """Generate predictions."""
    df = pd.read_csv(data)
    MyModel().predict(df, run_dir, out_dir)

if __name__ == "__main__":
    app()


# __main__.py
from .run import app

if __name__ == "__main__":
    app()
```

### Using CLI Helper

Alternatively, use the provided CLI helper for standard behavior:

```python
# run.py
from abdev_core.cli import create_cli_app
from .model import MyModel

app = create_cli_app(MyModel, "my_model")

if __name__ == "__main__":
    app()
```

## Constants

```python
from abdev_core import (
    PROPERTY_LIST,           # ['HIC', 'Tm2', 'Titer', 'PR_CHO', 'AC-SINS_pH7.4']
    ASSAY_HIGHER_IS_BETTER, # Dict mapping property -> bool
    FOLD_COL,               # Cross-validation fold column name
    DATASETS,               # ['GDPa1', 'GDPa1_cross_validation', 'heldout_test']
)
```

## API Reference

### Feature Loading
```python
load_features(feature_name, dataset="GDPa1")  # Quick access
FeatureLoader()                                # Advanced usage
```

### Base Classes
```python
BaseModel  # Abstract interface for baselines
```

### CLI Utilities
```python
create_cli_app(model_class, model_name)  # Create Typer app
validate_data_path(path)                  # Validate CSV path
validate_dir_path(path, must_exist=False) # Validate directory
```

### Legacy Utilities
```python
get_indices(seq_with_gaps)              # Get sequence indices
extract_region(scores, indices, region) # Extract CDR/FW regions
load_from_tamarind(filepath, ...)       # Load Tamarind outputs
```

## Output Format

All baselines must write `predictions.csv` with the following structure:

```csv
antibody_name,vh_protein_sequence,vl_protein_sequence,HIC,Tm2,Titer,PR_CHO,AC-SINS_pH7.4
ab_001,QVQL...,DIQM...,2.5,82.3,245.1,0.15,3.2
ab_002,EVQL...,SSEV...,2.8,81.9,230.5,0.18,4.5
```

Required columns:
- `antibody_name` (str)
- `vh_protein_sequence` (str)
- `vl_protein_sequence` (str)

Optional property columns (include only those you predict):
- `HIC` (float)
- `Tm2` (float)
- `Titer` (float)
- `PR_CHO` (float)
- `AC-SINS_pH7.4` (float)

## Examples

See existing baselines for full implementations:
- [`baselines/tap_linear`](../../baselines/tap_linear/): Training baseline with CV
- [`baselines/aggrescan3d`](../../baselines/aggrescan3d/): Non-training baseline (feature passthrough)

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy src/
```

