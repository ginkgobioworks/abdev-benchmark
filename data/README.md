# Data Directory

This directory contains the antibody developability benchmark datasets and structural data.

## Datasets

### GDPa1_v1.2_20250814.csv
Primary benchmark dataset containing:
- 247 antibodies
- 5 biophysical properties: HIC, Tm2, Titer, PR_CHO, AC-SINS_pH7.4
- Additional measurements and metadata
- VH/VL sequences and alignments
- Cross-validation fold assignments

### heldout-set-sequences.csv
Held-out test set with sequences only (labels withheld):
- Antibody sequences for final evaluation
- Use for generating competition submissions

## Structures

Predicted antibody structures from different tools:

### structures/AntiBodyBuilder3/
Structures predicted using AntiBodyBuilder3:
- `GDPa1/`: Structures for training set
- `heldout_test/`: Structures for test set

### structures/MOE_structures/
Structures predicted using Molecular Operating Environment (MOE):
- `GDPa1/`: Structures for training set
- `heldout_test/`: Structures for test set

**Acknowledgement**: MOE structures computed by Nels Thorsteinsen.

## Schema

See `schema/README.md` for detailed documentation of:
- Prediction format requirements
- Ground truth format
- Cross-validation structure
- Feature format
- Output directory conventions

## Feature Provenance

Pre-computed features in `../features/processed_features/` were generated using:

- **TAP**: Therapeutic Antibody Profiler descriptors
- **Aggrescan3D**: Aggregation propensity from structure
- **AntiFold**: Stability predictions
- **DeepSP**: Deep learning structural predictions
- **DeepViscosity**: Viscosity predictions
- **Saprot**: Protein language model features
- **BALM_Paired**: Antibody-specific language model
- **TEMPRO**: Thermostability predictions
- **MOE_properties**: Molecular descriptors from structures

Most features were computed externally (e.g., via Tamarind.bio) and imported into the benchmark format.

## Usage

Model should:
1. Read ground truth from `GDPa1_v1.2_20250814.csv`
2. Load features from `../features/processed_features/{dataset}/`
3. Generate predictions for appropriate dataset splits
4. Write predictions to `../predictions/{dataset}/{baseline}/`

For cross-validation (`GDPa1_cross_validation`):
- Use `hierarchical_cluster_IgG_isotype_stratified_fold` column
- Train on 4 folds, predict on the 5th
- Repeat for all 5 folds
- Output contains predictions for all samples using their held-out model

