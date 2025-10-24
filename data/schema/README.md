# Data Schema and I/O Contracts

This directory defines the data formats and contracts for the antibody developability benchmark.

## Prediction Format

All baselines must output predictions in the following CSV format:

### Required Columns
- `antibody_name` (string): Unique identifier for the antibody
  - Must match names in ground truth dataset
  - No duplicates
  - No missing values

### Property Columns
At least one of the following biophysical property columns:
- `HIC` (float): Hydrophobic Interaction Chromatography retention time (lower is better)
- `Tm2` (float): Second melting temperature in °C (higher is better)
- `Titer` (float): Expression titer (higher is better)
- `PR_CHO` (float): Polyreactivity CHO (lower is better)
- `AC-SINS_pH7.4` (float): Affinity-Capture Self-Interaction Nanoparticle Spectroscopy at pH 7.4 (lower is better)

### Optional Columns
- `vh_protein_sequence` (string): Variable heavy chain sequence
- `vl_protein_sequence` (string): Variable light chain sequence

### Example

```csv
antibody_name,HIC,Tm2,Titer
antibody-001,2.545,80.33,193.31
antibody-002,2.705,85.03,114.75
antibody-003,2.565,75.93,327.32
```

## Ground Truth Format

The ground truth dataset (`data/GDPa1_v1.2_20250814.csv`) contains:

### Key Columns
- `antibody_name`: Unique antibody identifier
- `antibody_id`: Numeric ID
- Property measurements: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`
- Additional measurements: `Purity`, `SEC %Monomer`, `SMAC`, etc.
- Sequences: `vh_protein_sequence`, `vl_protein_sequence`, `hc_protein_sequence`, `lc_protein_sequence`
- Fold assignments: `hierarchical_cluster_IgG_isotype_stratified_fold`, `random_fold`
- Alignment: `light_aligned_aho`, `heavy_aligned_aho`

### Cross-Validation Folds

The `hierarchical_cluster_IgG_isotype_stratified_fold` column defines 5-fold cross-validation splits:
- Values: 0, 1, 2, 3, 4
- Stratified by IgG isotype and hierarchical clustering
- Used for `GDPa1_cross_validation` dataset evaluation

## Dataset Types

### 1. GDPa1
Full dataset evaluation (all data used for predictions).

### 2. GDPa1_cross_validation
Cross-validation evaluation:
- Models trained on 4 folds, predict on held-out fold
- Metrics averaged across 5 folds
- Predictions file should contain predictions for all samples using their respective held-out fold model

### 3. heldout_test
Held-out test set with sequences only (no labels):
- File: `data/heldout-set-sequences.csv`
- Contains: `antibody_name`, `vh_protein_sequence`, `vl_protein_sequence`
- Used for final evaluation/competition submission

## Output Directory Structure

Baselines should write predictions to:
```
predictions/
  {dataset_name}/        # e.g., "GDPa1", "GDPa1_cross_validation", "heldout_test"
    {baseline_name}/     # e.g., "tap_linear", "aggrescan3d"
      {variant}.csv      # e.g., "tap_linear.csv", "aggrescan_average.csv"
```

Examples:
- `predictions/GDPa1/tap_linear/tap_linear.csv`
- `predictions/GDPa1_cross_validation/tap_linear/tap_linear.csv`
- `predictions/GDPa1/aggrescan3d/aggrescan_average.csv`

## Feature Format

Pre-computed features are stored in:
```
features/processed_features/
  {dataset_name}/       # e.g., "GDPa1", "heldout_test"
    {feature_source}.csv  # e.g., "TAP.csv", "Aggrescan3D.csv"
```

Feature files must contain:
- `antibody_name` column
- One or more feature columns

Example feature file (`TAP.csv`):
```csv
antibody_name,SFvCSP,PSH,PPC,PNC,CDR Length
antibody-001,0.123,0.456,0.789,0.234,15
antibody-002,0.145,0.467,0.801,0.256,14
```

## Validation

Prediction format validation is handled automatically by the orchestrator using `abdev_core.validate_prediction_format()`.

You can also validate predictions programmatically:
```python
from abdev_core import validate_prediction_file

is_valid, errors = validate_prediction_file("path/to/predictions.csv")
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

Validation checks:
- Required columns present
- No missing antibody_name values
- No duplicate antibody_name values
- At least one property column present
- Numeric values in property columns

