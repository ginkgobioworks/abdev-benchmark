# Notebooks

Tutorial notebooks for the antibody developability benchmark.

## Migration Note

**Important**: This repository has been refactored to use Pixi for dependency management. The notebooks may reference old import paths or structures.

### Using Notebooks with New Structure

To run notebooks with the new repository structure:

1. Install Pixi:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Set up a notebook environment in one of the baseline directories, or create a dedicated notebooks environment:
   ```bash
   # Option 1: Use an existing baseline environment
   cd ../baselines/tap_linear
   pixi install
   pixi run jupyter notebook
   
   # Option 2: Create a notebooks-specific environment (recommended)
   # Add a pixi.toml in notebooks/ with jupyter dependencies
   ```

3. Update import paths in notebooks:
   - Old: `from utils import PROPERTY_LIST`
   - New: `from abdev_core import PROPERTY_LIST`
   
   - Old: `from evaluate import evaluate_model`
   - New: `from abdev_core import evaluate_model`
   
   - Old: `from evaluation.metrics import evaluate`
   - New: `from abdev_core import evaluate`

### Available Notebooks

- **tutorial_pIgGen.ipynb**: How to train an antibody developability model with foundation model embeddings

### Contributing Notebooks

When adding new notebooks:
1. Use relative paths for data access: `../data/`, `../features/`, etc.
2. Import from `abdev_core` for shared constants and evaluation functions
3. Document any additional dependencies needed

### Setting Up Jupyter with Pixi

To create a Pixi environment for notebooks, create `pixi.toml` in this directory:

```toml
[workspace]
name = "abdev-notebooks"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64"]

[dependencies]
python = "3.11.*"
jupyter = ">=1.0"
pandas = ">=2.0"
numpy = ">=1.24"
matplotlib = ">=3.7"
scikit-learn = ">=1.3"

[pypi-dependencies]
abdev-core = { path = "../libs/abdev_core", editable = true }

[tasks]
notebook = "jupyter notebook"
lab = "jupyter lab"
```

Then run:
```bash
pixi install
pixi run notebook
```

