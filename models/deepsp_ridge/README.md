# DeepSP Ridge

Ridge regression on **DeepSP** spatial features computed on-the-fly from antibody sequences.

## Overview

This model uses the DeepSP deep learning framework to generate 30 spatial properties from VH and VL antibody sequences, then trains simple Ridge regression models on top to predict antibody developability properties.

### Why DeepSP?

- **Spatial properties**: DeepSP predicts structure-based spatial properties without requiring 3D structures
- **Sequence-only input**: Computes features directly from amino acid sequences via ANARCI alignment
- **Interpretable features**: 30 spatial descriptors covering charge distributions and aggregation propensity across CDRs and variable regions
- **Pre-trained models**: Uses pre-trained Conv1D neural networks from the original DeepSP publication

## Method

### 1. Sequence Alignment

For each antibody, VH and VL chains are aligned to IMGT numbering scheme using ANARCI:

```
VH_seq → ANARCI (IMGT, heavy) → Aligned VH (145 positions)
VL_seq → ANARCI (IMGT, light)  → Aligned VL (127 positions)
Combined → Aligned sequence (272 positions)
```

### 2. Feature Generation

The aligned sequence is one-hot encoded and passed through three pre-trained Conv1D models to predict:

- **SAP_pos**: Spatial Aggregation Propensity (positive charges) for 10 regions
- **SCM_neg**: Spatial Charge Map (negative charges) for 10 regions
- **SCM_pos**: Spatial Charge Map (positive charges) for 10 regions

**Regions**: CDRH1, CDRH2, CDRH3, CDRL1, CDRL2, CDRL3, CDR, Hv, Lv, Fv

**Total**: 30 spatial features per antibody

### 3. Ridge Regression Training

For each antibody property (HIC, Tm2, Titer, etc.):

- Extract 30 DeepSP features and labels for all samples with valid measurements
- Train a Ridge regression model (alpha=1.0) on features → property values
- Save trained models for prediction time

### 4. Prediction

At prediction time:
- Generate DeepSP features for new samples using pre-trained Conv1D models
- Use trained Ridge models to predict each property
- Return predictions as DataFrame

## Usage

### Train

From the repository root:

```bash
cd models/deepsp_ridge
pixi install

# Train on GDPa1 dataset
pixi run python -m deepsp_ridge train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Predict

```bash
# Predict on training data
pixi run python -m deepsp_ridge predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/predictions

# Predict on heldout data
pixi run python -m deepsp_ridge predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/predictions_heldout
```

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

This automatically discovers and runs all models, including DeepSP Ridge, with 5-fold cross-validation.

## Implementation Notes

- **On-the-fly computation**: DeepSP features are always recomputed from sequences (no caching)
- **Lazy loading**: Pre-trained Conv1D models are loaded only on first use to save memory
- **ANARCI dependency**: Requires ANARCI for sequence alignment to IMGT numbering
- **Batch processing**: All sequences are processed together for efficiency
- **NaN handling**: Samples with missing property values are automatically excluded during training for that property
- **No internal CV**: Training data is not split internally; the orchestrator handles cross-validation

## Features Used

- **DeepSP spatial features**: 30 features covering spatial charge distributions and aggregation propensity
- **Two-chain integration**: Features computed from aligned VH+VL concatenated sequences
- **Pre-trained models**: Three Conv1D neural networks trained on antibody structure data

## Performance Considerations

### Training Time

- ~10-30 seconds per 100 samples for feature generation (depends on ANARCI and TensorFlow)
- Each sequence requires ANARCI alignment and Conv1D forward passes
- Additional ~1 second for Ridge regression training

### Memory Usage

- Pre-trained models: ~50MB (three Conv1D models)
- Features: ~5MB for full GDPa1 dataset (30 features × ~400 samples × 4 bytes)

## Model Architecture

### DeepSP Conv1D Models

- **Input**: One-hot encoded aligned sequences (272 positions × 21 amino acids)
- **Architecture**: 1D Convolutional Neural Networks
- **Output**: 10 spatial features per model (30 total from 3 models)
- **Pre-trained**: Models trained on antibody structure database

### Ridge Regression

- **Regularization**: Alpha = 1.0 (L2 penalty)
- **Input**: 30 DeepSP spatial features
- **Output**: Scalar property predictions
- **Training**: One model per property

## Comparison to Other Models

### vs. ESM2 Ridge

- **ESM2 Ridge**: General protein language model embeddings (640D)
- **DeepSP Ridge**: Antibody-specific spatial features (30D), more interpretable

### vs. TAP Linear

- **TAP**: Hand-crafted biophysical and sequence descriptors
- **DeepSP**: Learned spatial properties from structure-based training

### vs. Pre-computed Features

Unlike models that use pre-computed DeepSP features, this implementation computes features on-the-fly, allowing predictions on any new sequences without pre-processing.

## References

- **DeepSP**: Kalejaye, L. et al. (2024), "DeepSP: Deep Learning-Based Spatial Properties to Predict Monoclonal Antibody Stability", *Computational and Structural Biotechnology Journal*, 23, 2220–2229
- **ANARCI**: Dunbar, J. & Deane, C.M. (2016), "ANARCI: antigen receptor numbering and receptor classification"
- **GDPa1 dataset**: [ginkgo-datapoints/GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1)

