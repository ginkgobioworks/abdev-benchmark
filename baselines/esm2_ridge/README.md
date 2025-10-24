# ESM2 Ridge Baseline

Ridge regression on embeddings from the **ESM2** general protein language model with two-chain encoding.

## Overview

This baseline uses the ESM2 protein language model to generate fixed-length embeddings for VH and VL antibody sequences separately, then concatenates these embeddings and trains simple Ridge regression models on top to predict antibody developability properties.

### Why ESM2 Two-Chain?

- **General protein model**: ESM2 is trained on broad evolutionary data, providing a non-antibody-specific baseline
- **Separate chain encoding**: VH and VL are embedded independently, avoiding padding contamination
- **No batching artifacts**: Each sequence is processed individually to ensure clean mean-pooling
- **Concatenated features**: Combined VH+VL representation captures both chains' contributions
- **Minimal fine-tuning**: Ridge regression is simple, fast, and provides interpretable baseline performance

## Method

### 1. Separate Chain Embedding

For each antibody, we embed the heavy and light chains independently:

**VH Embedding:**
```
VH_seq → ESM2 Tokenizer → ESM2 Model → Last Hidden State → Mean Pool → vh_embedding
```

**VL Embedding:**
```
VL_seq → ESM2 Tokenizer → ESM2 Model → Last Hidden State → Mean Pool → vl_embedding
```

Key features:
- **No batching**: Each sequence is processed individually to avoid padding tokens
- **Attention mask**: Mean pooling excludes padding tokens using the attention mask
- **Individual forward passes**: VH and VL are encoded separately for clean representations

### 2. Feature Concatenation

After generating embeddings for both chains, we concatenate them:

```
combined_embedding = [vh_embedding, vl_embedding]
```

For ESM2-t6 (8M parameters), the embedding dimension is 320, so:
- VH embedding: 320D
- VL embedding: 320D  
- Combined: 640D

### 3. Ridge Regression Training

For each antibody property (HIC, Tm2, Titer, etc.):

- Extract concatenated embeddings and labels for all samples with valid measurements
- Train a Ridge regression model (alpha=1.0) on embeddings → property values
- Save trained models for prediction time

### 4. Prediction

At prediction time:
- Generate VH and VL embeddings separately for new samples
- Concatenate embeddings
- Use trained Ridge models to predict each property
- Return predictions as DataFrame

## Usage

### Train

From the repository root:

```bash
cd baselines/esm2_ridge
pixi install

# Train on GDPa1 dataset
pixi run python -m esm2_ridge train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Predict

```bash
# Predict on training data
pixi run python -m esm2_ridge predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

This automatically discovers and runs all baselines, including ESM2 Ridge, with 5-fold cross-validation.

## Implementation Notes

- **Lazy loading**: The transformer model is loaded only on first use (train or predict) to save memory
- **GPU support**: Automatically detects and uses CUDA if available; falls back to CPU
- **Individual processing**: Each sequence is embedded separately to avoid padding token contamination
- **NaN handling**: Samples with missing property values are automatically excluded during training for that property
- **No internal CV**: Training data is not split internally; the orchestrator handles cross-validation

## Features Used

- **ESM2 embeddings**: 320-dimensional vectors from ESM2-t6 (8M) hidden layer
- **Two-chain concatenation**: 640D combined representation (320D VH + 320D VL)
- **No auxiliary features**: Unlike TAP baselines, this approach relies solely on sequence information encoded in the embeddings

## Performance Considerations

### Training Time

- ~2-5 seconds per 100 samples (CPU) / ~0.5-1 seconds (GPU) for embedding generation
- Each sequence requires a separate forward pass (no batching)
- Additional ~1 second for Ridge regression training

### Memory Usage

- Model: ~30MB (ESM2-t6 8M parameters)
- Embeddings: ~10MB for full GDPa1 dataset (640D × ~400K samples × 4 bytes)

## Model Architecture

### ESM2-t6-8M

- **Parameters**: 8 million
- **Layers**: 6 transformer layers
- **Embedding dimension**: 320
- **Training data**: UniRef50 (evolutionary sequences)
- **Model checkpoint**: [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D)

### Ridge Regression

- **Regularization**: Alpha = 1.0 (L2 penalty)
- **Input**: 640D concatenated embeddings (VH + VL)
- **Output**: Scalar property predictions
- **Training**: One model per property

## Comparison to Other Baselines

### vs. p-IgGen

- **p-IgGen**: Antibody-specific model, processes paired sequences jointly
- **ESM2 Ridge**: General protein model, processes chains separately then concatenates

### vs. TAP Features

- **TAP**: Hand-crafted biophysical and sequence features
- **ESM2 Ridge**: Learned representations from protein language model

### vs. Other ESM2 Variants

This baseline uses the smallest ESM2 model (8M parameters) for efficiency. Larger models (35M, 150M, 650M, 3B, 15B) are available but require significantly more compute and memory.

## References

- **ESM-2**: Lin et al. (2022), "Language models of protein sequences at the scale of evolution"
- **Model checkpoint**: [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
- **GDPa1 dataset**: [ginkgo-datapoints/GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1)

